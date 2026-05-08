"""
PPO training script for Procgen — CleanRL-style (single file, readable top-to-bottom).

Phase 2 additions:
  - augmentation_method: "none" | "rad" | "drac" | "ucb_drac"
  - level_selection: "uniform" | "plr"
  These can be combined freely (e.g., PLR + DrAC).

Usage:
    python src/train_ppo.py --env-id coinrun --seed 1
    python src/train_ppo.py --config configs/ppo_procgen_drac.yaml --env-id starpilot
    python src/train_ppo.py --config configs/ppo_procgen_plr_drac.yaml --env-id ninja --seed 2
    python src/train_ppo.py --env-id bigfish --resume checkpoints/bigfish/run_name/latest.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import wandb

from src.networks import ActorCritic
from src.env_utils import make_procgen_envs, make_eval_envs, RewardNormWrapper
from src.checkpoint import save_checkpoint, load_checkpoint
from src.utils import (
    set_seed,
    get_device,
    load_config,
    compute_derived_config,
    get_rng_states,
    set_rng_states,
    make_run_name,
)

EXPECTED_TEST_RETURNS: dict[str, tuple[float, float]] = {
    "coinrun":   (8.5,  9.5),
    "starpilot": (25.0, 35.0),
    "bigfish":   (10.0, 20.0),
    "ninja":     (6.0,  8.0),
}


def _build_augmentation_module(cfg: dict[str, Any], model: ActorCritic) -> Any:
    """Instantiate the augmentation helper from config."""
    method = cfg.get("augmentation_method", "none")
    if method == "none":
        return None
    if method == "rad":
        from src.rad import RAD
        aug = RAD(aug_type=cfg.get("aug_type", "crop"))
        print(f"[aug] RAD  aug_type={aug.aug_type}")
        return aug
    if method == "drac":
        from src.drac import DrAC
        aug = DrAC(
            model=model,
            aug_type=cfg.get("aug_type", "crop"),
            aug_coef=cfg.get("aug_coef", 0.1),
            mode=cfg.get("drac_mode", "full"),
        )
        print(f"[aug] DrAC  aug_type={cfg.get('aug_type','crop')}  aug_coef={cfg.get('aug_coef',0.1)}")
        return aug
    if method == "ucb_drac":
        from src.ucb_drac import UCBDrAC
        aug = UCBDrAC(
            model=model,
            aug_names=cfg.get("ucb_augmentations", None),
            aug_coef=cfg.get("aug_coef", 0.1),
            ucb_c=cfg.get("ucb_exploration", 0.1),
            drac_mode=cfg.get("drac_mode", "full"),
        )
        print(f"[aug] UCB-DrAC  arms={aug.aug_names}  ucb_c={aug.ucb_c}")
        return aug
    raise ValueError(f"Unknown augmentation_method '{method}'.")


def _build_plr(cfg: dict[str, Any]) -> Any:
    """Instantiate PLR if level_selection == 'plr', else return None."""
    if cfg.get("level_selection", "uniform") != "plr":
        return None
    from src.plr import PLR
    plr = PLR(
        num_levels=cfg["num_levels"],
        rho=cfg.get("plr_rho", 0.5),
        beta=cfg.get("plr_beta", 0.1),
        staleness_coef=cfg.get("plr_staleness_coef", 0.1),
        scoring=cfg.get("plr_scoring", "l1_value_loss"),
        seed=cfg["seed"],
    )
    print(f"[plr] PLR enabled  rho={plr.rho}  beta={plr.beta}  staleness_coef={plr.staleness_coef}")
    return plr


def _make_plr_env(cfg: dict[str, Any], level_seed: int) -> Any:
    """Create a training env pinned to a single PLR-selected level."""
    return make_procgen_envs(
        env_id=cfg["env_id"],
        num_envs=cfg["num_envs"],
        num_levels=1,
        start_level=level_seed,
        distribution_mode=cfg["distribution_mode"],
        gamma=cfg["gamma"],
        normalize_reward=cfg["normalize_reward"],
        seed=cfg["seed"] + level_seed,
        env_backend=cfg.get("env_backend", "auto"),
    )


def evaluate_policy(
    model: ActorCritic,
    env_id: str,
    num_eval_envs: int,
    num_levels: int,
    start_level: int,
    distribution_mode: str,
    num_episodes: int,
    device: torch.device,
    deterministic: bool = False,
    env_backend: str | None = None,
) -> tuple[float, float]:
    """Roll out policy on eval envs for num_episodes. Returns (mean, std)."""
    eval_env = make_eval_envs(
        env_id=env_id,
        num_envs=num_eval_envs,
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode=distribution_mode,
        env_backend=env_backend,
    )
    obs = eval_env.reset()
    episode_returns: list[float] = []
    current_returns = np.zeros(num_eval_envs, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        while len(episode_returns) < num_episodes:
            obs_t = torch.tensor(obs, dtype=torch.uint8, device=device)
            if deterministic:
                actions = model.encoder(obs_t)
                actions = model.policy_head(actions).argmax(dim=-1).cpu().numpy()
            else:
                actions, _, _, _ = model.get_action_and_value(obs_t)
                actions = actions.cpu().numpy()
            obs, rewards, dones, _ = eval_env.step(actions)
            current_returns += rewards
            for i, done in enumerate(dones):
                if done:
                    episode_returns.append(float(current_returns[i]))
                    current_returns[i] = 0.0
                    if len(episode_returns) >= num_episodes:
                        break
    model.train()
    eval_env.close()
    arr = np.array(episode_returns[:num_episodes])
    return float(arr.mean()), float(arr.std())


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments that override the YAML config."""
    parser = argparse.ArgumentParser(description="PPO on Procgen (thesis)")
    parser.add_argument("--config",              type=str, default="configs/ppo_procgen_baseline.yaml")
    parser.add_argument("--env-id",              type=str, default=None)
    parser.add_argument("--seed",                type=int, default=None)
    parser.add_argument("--encoder",             type=str, default=None, choices=["impala", "nature", "small"])
    parser.add_argument("--num-levels",          type=int, default=None)
    parser.add_argument("--device",              type=str, default=None)
    parser.add_argument("--total-timesteps",     type=int, default=None)
    parser.add_argument("--augmentation-method", type=str, default=None,
                        choices=["none", "rad", "drac", "ucb_drac"])
    parser.add_argument("--aug-type",            type=str, default=None)
    parser.add_argument("--level-selection",     type=str, default=None, choices=["uniform", "plr"])
    parser.add_argument("--wandb-tags",          type=str, default=None)
    parser.add_argument("--resume",              type=str, default=None)
    parser.add_argument("--no-wandb",            action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cli_overrides = {
        "env_id":               args.env_id,
        "seed":                 args.seed,
        "encoder":              args.encoder,
        "num_levels":           args.num_levels,
        "device":               args.device,
        "total_timesteps":      args.total_timesteps,
        "augmentation_method":  args.augmentation_method,
        "aug_type":             args.aug_type,
        "level_selection":      args.level_selection,
    }
    cfg = load_config(args.config, cli_overrides)
    cfg = compute_derived_config(cfg)
    cfg.setdefault("augmentation_method", "none")
    cfg.setdefault("level_selection", "uniform")

    device = get_device(cfg["device"])
    set_seed(cfg["seed"])

    run_name = make_run_name(cfg)
    checkpoint_dir = Path("checkpoints") / cfg["env_id"] / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # --- Resume ---
    resume_checkpoint: dict[str, Any] | None = None
    resumed_wandb_run_id: str | None = None
    global_step_start = 0

    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        raw_ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        saved_cfg = raw_ckpt["config"]
        for k in ["env_id", "seed", "encoder", "augmentation_method", "level_selection"]:
            if cli_overrides.get(k) is None:
                cfg[k] = saved_cfg.get(k, cfg.get(k))
        cfg = compute_derived_config(cfg)
        global_step_start = raw_ckpt["global_step"]
        resumed_wandb_run_id = raw_ckpt.get("wandb_run_id")
        resume_checkpoint = raw_ckpt
        print(f"[resume] Resuming from global_step={global_step_start:,}")

    # --- W&B tags: consistent structure for analysis queries ---
    method_tag = cfg.get("augmentation_method", "none")
    if cfg.get("level_selection") == "plr":
        method_tag = f"plr_{method_tag}" if method_tag != "none" else "plr"
    wandb_tags = [method_tag, cfg["env_id"], f"seed_{cfg['seed']}", cfg["encoder"]]
    if args.wandb_tags:
        wandb_tags += [t.strip() for t in args.wandb_tags.split(",")]

    use_wandb = not args.no_wandb
    wandb_run_id = resumed_wandb_run_id
    if use_wandb:
        wandb_kwargs: dict[str, Any] = dict(
            project=cfg["wandb_project"],
            entity=cfg.get("wandb_entity") or None,
            name=run_name, config=cfg, tags=wandb_tags, save_code=True,
        )
        if wandb_run_id is not None:
            wandb_kwargs["id"] = wandb_run_id
            wandb_kwargs["resume"] = "must"
        run = wandb.init(**wandb_kwargs)
        wandb_run_id = run.id
        print(f"[wandb] Run ID: {wandb_run_id}  tags={wandb_tags}")
    else:
        print("[wandb] Logging disabled.")
        wandb_run_id = "disabled"

    # --- Environment ---
    env_backend = cfg.get("env_backend", "auto")
    print(f"[env] {cfg['env_id']}  {cfg['num_envs']} envs  {cfg['num_levels']} levels  mode={cfg['distribution_mode']}  backend={env_backend}")
    train_env = make_procgen_envs(
        env_id=cfg["env_id"], num_envs=cfg["num_envs"],
        num_levels=cfg["num_levels"], start_level=cfg["start_level"],
        distribution_mode=cfg["distribution_mode"],
        gamma=cfg["gamma"], normalize_reward=cfg["normalize_reward"], seed=cfg["seed"],
        env_backend=env_backend,
    )
    obs_space = train_env.observation_space
    act_space = train_env.action_space
    num_actions: int = act_space.n
    print(f"[env] obs_space={obs_space.shape}  num_actions={num_actions}")

    # --- Model ---
    model = ActorCritic(encoder=cfg["encoder"], num_actions=num_actions).to(device)
    if device.type != "cpu":
        # torch.compile() bypasses MIOpen/cuDNN and generates Triton-based kernels.
        # On AMD ROCm, MIOpen's backward pass for IMPALA-CNN runs ~20x below theoretical
        # throughput; Triton-based kernels avoid this entirely.
        # reduce-overhead: minimises kernel launch latency (good for many small ops).
        model = torch.compile(model, mode="reduce-overhead")
        print("[model] torch.compile applied (mode=reduce-overhead)")
    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"], eps=1e-5)

    # --- Build augmentation and PLR after model is ready ---
    aug_module = _build_augmentation_module(cfg, model)
    plr = _build_plr(cfg)

    # --- Restore from checkpoint ---
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        if cfg["normalize_reward"] and resume_checkpoint.get("reward_normalizer_state"):
            assert isinstance(train_env, RewardNormWrapper)
            train_env.set_reward_normalizer_state(resume_checkpoint["reward_normalizer_state"])
        if resume_checkpoint.get("rng_states"):
            set_rng_states(resume_checkpoint["rng_states"])
        if plr is not None and resume_checkpoint.get("extra", {}).get("plr_state"):
            plr.set_state(resume_checkpoint["extra"]["plr_state"])
        print("[resume] Restored successfully.")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {cfg['encoder'].upper()}-CNN  params={total_params:,}")

    # --- Rollout buffer ---
    obs_buf  = torch.zeros((cfg["num_steps"], cfg["num_envs"]) + obs_space.shape, dtype=torch.uint8)
    act_buf  = torch.zeros((cfg["num_steps"], cfg["num_envs"]), dtype=torch.long)
    logp_buf = torch.zeros((cfg["num_steps"], cfg["num_envs"]))
    rew_buf  = torch.zeros((cfg["num_steps"], cfg["num_envs"]))
    done_buf = torch.zeros((cfg["num_steps"], cfg["num_envs"]))
    val_buf  = torch.zeros((cfg["num_steps"], cfg["num_envs"]))

    obs  = torch.from_numpy(train_env.reset())
    done = torch.zeros(cfg["num_envs"])

    global_step   = global_step_start
    num_updates   = cfg["num_updates"]
    updates_done  = global_step // cfg["batch_size"]

    ep_return_buf = np.zeros(cfg["num_envs"], dtype=np.float32)
    completed_ep_returns: list[float] = []

    next_checkpoint_step = (global_step // cfg["checkpoint_freq"] + 1) * cfg["checkpoint_freq"]
    next_eval_step       = (global_step // cfg["eval_freq"]       + 1) * cfg["eval_freq"]

    plr_current_level: int = 0
    plr_vl_accum: list[float] = []

    print(f"\n{'='*65}")
    print(f"  Starting PPO  |  env={cfg['env_id']}  aug={cfg['augmentation_method']}  lvl={cfg['level_selection']}")
    print(f"  total_steps={cfg['total_timesteps']:,}  updates={num_updates - updates_done:,}  device={device}")
    print(f"{'='*65}\n")

    start_time = time.time()
    # Phase timing accumulators (milliseconds, reset each print interval)
    _pt_rollout = _pt_bulk_xfer = _pt_gae = _pt_ppo_xfer = _pt_ppo_grad = 0.0
    _pt_updates = 0

    for update in range(updates_done + 1, num_updates + 1):

        # ==================================================================
        # 1. PLR level selection (reconstructs env if needed)
        # ==================================================================
        if plr is not None:
            plr_current_level = plr.sample_level()
            train_env.close()
            train_env = _make_plr_env(cfg, plr_current_level)
            obs  = torch.from_numpy(train_env.reset())
            done = torch.zeros(cfg["num_envs"])
            plr_vl_accum = []

        # ==================================================================
        # 2. Collect rollout
        # ==================================================================
        _t0 = time.perf_counter()

        # GPU-side buffers for log_prob and value (avoid per-step .cpu() calls)
        logp_buf_gpu = torch.zeros((cfg["num_steps"], cfg["num_envs"]), device=device)
        val_buf_gpu  = torch.zeros((cfg["num_steps"], cfg["num_envs"]), device=device)

        model.eval()
        with torch.no_grad():
            for step in range(cfg["num_steps"]):
                obs_buf[step]  = obs
                done_buf[step] = done

                obs_dev = obs.to(device, non_blocking=True)
                action, log_prob, _, value = model.get_action_and_value(obs_dev)

                # Store log_prob and value on GPU — no sync needed.
                logp_buf_gpu[step] = log_prob
                val_buf_gpu[step]  = value.squeeze(-1)

                # Only sync once: pull action to CPU for env.step().
                action_cpu = action.cpu()
                act_buf[step] = action_cpu

                raw_obs, reward, done_np, infos = train_env.step(action_cpu.numpy())

                rew_buf[step] = torch.from_numpy(reward)
                done = torch.from_numpy(done_np)
                obs  = torch.from_numpy(raw_obs)
                global_step += cfg["num_envs"]

                ep_return_buf += reward
                for i, d in enumerate(done_np):
                    if d:
                        completed_ep_returns.append(float(ep_return_buf[i]))
                        ep_return_buf[i] = 0.0

            next_value = model.get_value(obs.to(device)).squeeze(-1)

        _t1 = time.perf_counter()
        _pt_rollout += (_t1 - _t0) * 1000

        # Bulk transfer log_prob and value from GPU → CPU (one sync, not 256).
        logp_buf = logp_buf_gpu.cpu()
        val_buf  = val_buf_gpu.cpu()
        next_value = next_value.cpu()

        _t2 = time.perf_counter()
        _pt_bulk_xfer += (_t2 - _t1) * 1000

        # ==================================================================
        # 3. GAE
        # ==================================================================
        advantages = torch.zeros_like(rew_buf)
        last_gae = 0.0
        for t in reversed(range(cfg["num_steps"])):
            if t == cfg["num_steps"] - 1:
                next_non_terminal = 1.0 - done
                next_val = next_value
            else:
                next_non_terminal = 1.0 - done_buf[t + 1]
                next_val = val_buf[t + 1]
            delta = rew_buf[t] + cfg["gamma"] * next_val * next_non_terminal - val_buf[t]
            last_gae = delta + cfg["gamma"] * cfg["gae_lambda"] * next_non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + val_buf

        _t3 = time.perf_counter()
        _pt_gae += (_t3 - _t2) * 1000

        # ==================================================================
        # 4. PPO update
        # ==================================================================
        model.train()

        b_obs        = obs_buf.reshape((-1,) + obs_space.shape)
        b_actions    = act_buf.reshape(-1)
        b_logprobs   = logp_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns    = returns.reshape(-1)
        b_values     = val_buf.reshape(-1)

        # Pre-transfer entire batch to GPU once (not per-minibatch).
        # obs stays as uint8 until the network normalizes it.
        b_obs_dev  = b_obs.to(device, non_blocking=True)
        b_acts_dev = b_actions.to(device, non_blocking=True)
        b_logp_dev = b_logprobs.to(device, non_blocking=True)
        b_advs_dev = b_advantages.to(device, non_blocking=True)
        b_rets_dev = b_returns.to(device, non_blocking=True)
        b_vals_dev = b_values.to(device, non_blocking=True)
        if device.type != "cpu":
            torch.cuda.synchronize()

        _t4 = time.perf_counter()
        _pt_ppo_xfer += (_t4 - _t3) * 1000

        clip_fracs: list[float] = []
        pg_losses:  list[float] = []
        vf_losses:  list[float] = []
        ent_losses: list[float] = []
        approx_kls: list[float] = []
        aug_losses: list[float] = []
        selected_aug_name: str | None = None

        b_inds = np.arange(cfg["batch_size"])
        for _ in range(cfg["num_epochs"]):
            np.random.shuffle(b_inds)
            for start in range(0, cfg["batch_size"], cfg["minibatch_size"]):
                mb_inds = b_inds[start:start + cfg["minibatch_size"]]

                # Index into pre-transferred GPU tensors — no transfer here.
                mb_obs   = b_obs_dev[mb_inds]
                mb_acts  = b_acts_dev[mb_inds]
                mb_logps = b_logp_dev[mb_inds]
                mb_advs  = b_advs_dev[mb_inds]
                mb_rets  = b_rets_dev[mb_inds]
                mb_vals  = b_vals_dev[mb_inds]

                # RAD: augment observations before the PPO forward pass
                mb_obs_input = mb_obs
                if aug_module is not None and cfg["augmentation_method"] == "rad":
                    mb_obs_input = aug_module.augment(mb_obs)

                _, new_logp, entropy, new_value = model.get_action_and_value(mb_obs_input, mb_acts)
                new_value = new_value.squeeze(-1)

                if cfg.get("norm_adv", True):
                    mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-8)

                log_ratio = new_logp - mb_logps
                ratio = log_ratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > cfg["clip_coef"]).float().mean().item()

                pg_loss1 = -mb_advs * ratio
                pg_loss2 = -mb_advs * ratio.clamp(1 - cfg["clip_coef"], 1 + cfg["clip_coef"])
                pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

                v_loss_unclipped = (new_value - mb_rets) ** 2
                v_clipped = mb_vals + (new_value - mb_vals).clamp(-cfg["clip_coef"], cfg["clip_coef"])
                v_loss_clipped = (v_clipped - mb_rets) ** 2
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                ent_loss = entropy.mean()
                loss = pg_loss - cfg["ent_coef"] * ent_loss + cfg["vf_coef"] * v_loss

                # DrAC: add KL + MSE regularization on augmented observations
                if aug_module is not None and cfg["augmentation_method"] == "drac":
                    aug_loss = aug_module.compute_loss(mb_obs)
                    loss = loss + aug_loss
                    aug_losses.append(aug_loss.item())

                # UCB-DrAC: select aug via bandit, add reg loss
                elif aug_module is not None and cfg["augmentation_method"] == "ucb_drac":
                    aug_loss, selected_aug_name = aug_module.compute_loss_and_select(mb_obs)
                    loss = loss + aug_loss
                    aug_losses.append(aug_loss.item())

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
                optimizer.step()

                clip_fracs.append(clip_frac)
                pg_losses.append(pg_loss.item())
                vf_losses.append(v_loss.item())
                ent_losses.append(ent_loss.item())
                approx_kls.append(approx_kl)

                if plr is not None:
                    plr_vl_accum.append(v_loss.item())

        if device.type != "cpu":
            torch.cuda.synchronize()
        _t5 = time.perf_counter()
        _pt_ppo_grad += (_t5 - _t4) * 1000
        _pt_updates += 1

        # ==================================================================
        # 5. Post-update: PLR score update + UCB-DrAC bandit update
        # ==================================================================
        if plr is not None and plr_vl_accum:
            plr.update_score(plr_current_level, float(np.mean(plr_vl_accum)))

        if aug_module is not None and cfg["augmentation_method"] == "ucb_drac":
            recent = completed_ep_returns[-16:] if completed_ep_returns else [0.0]
            if selected_aug_name is not None:
                aug_module.update(selected_aug_name, float(np.mean(recent)))

        # ==================================================================
        # 6. Logging
        # ==================================================================
        elapsed = time.time() - start_time
        sps = global_step / elapsed

        log_dict: dict[str, Any] = {
            "charts/SPS":           sps,
            "losses/policy_loss":   np.mean(pg_losses),
            "losses/value_loss":    np.mean(vf_losses),
            "losses/entropy":       np.mean(ent_losses),
            "losses/approx_kl":     np.mean(approx_kls),
            "losses/clip_fraction": np.mean(clip_fracs),
            "charts/learning_rate": cfg["learning_rate"],
            "train/global_step":    global_step,
        }
        if aug_losses:
            log_dict["losses/aug_reg_loss"] = np.mean(aug_losses)
        if plr is not None:
            log_dict.update(plr.get_stats())
        if aug_module is not None and cfg["augmentation_method"] == "ucb_drac":
            log_dict.update(aug_module.get_stats())
        if completed_ep_returns:
            log_dict["charts/episodic_return_normalised"]     = np.mean(completed_ep_returns)
            log_dict["charts/episodic_return_normalised_std"] = np.std(completed_ep_returns)
            completed_ep_returns.clear()

        if np.mean(approx_kls) > 0.03:
            print(f"  ⚠  approx_kl={np.mean(approx_kls):.4f} > 0.03 at step {global_step:,}")

        if use_wandb:
            wandb.log(log_dict, step=global_step)

        if update % 10 == 0 or update == 1:
            aug_str = f"  aug={np.mean(aug_losses):.4f}" if aug_losses else ""
            print(
                f"  update {update:>5}/{num_updates}  step {global_step:>10,}  "
                f"SPS {sps:>7,.0f}  pg={np.mean(pg_losses):+.4f}  "
                f"vf={np.mean(vf_losses):.4f}  ent={np.mean(ent_losses):.4f}  "
                f"kl={np.mean(approx_kls):.4f}{aug_str}"
            )
            if _pt_updates > 0:
                n = _pt_updates
                print(
                    f"  [phase ms/update]  rollout={_pt_rollout/n:.0f}  "
                    f"bulk_xfer={_pt_bulk_xfer/n:.1f}  gae={_pt_gae/n:.1f}  "
                    f"ppo_xfer={_pt_ppo_xfer/n:.0f}  ppo_grad={_pt_ppo_grad/n:.0f}  "
                    f"total={(_pt_rollout+_pt_bulk_xfer+_pt_gae+_pt_ppo_xfer+_pt_ppo_grad)/n:.0f}"
                )
                _pt_rollout = _pt_bulk_xfer = _pt_gae = _pt_ppo_xfer = _pt_ppo_grad = 0.0
                _pt_updates = 0

        # ==================================================================
        # 7. Periodic evaluation
        # ==================================================================
        if global_step >= next_eval_step:
            print(f"\n[eval] step {global_step:,}...")
            train_mean, train_std = evaluate_policy(
                model=model, env_id=cfg["env_id"], num_eval_envs=min(cfg["num_envs"], 16),
                num_levels=cfg["num_levels"], start_level=cfg["start_level"],
                distribution_mode=cfg["distribution_mode"],
                num_episodes=cfg["eval_episodes"], device=device,
                env_backend=cfg.get("env_backend", "auto"),
            )
            test_mean, test_std = evaluate_policy(
                model=model, env_id=cfg["env_id"], num_eval_envs=min(cfg["num_envs"], 16),
                num_levels=0, start_level=cfg["num_levels"],
                distribution_mode=cfg["distribution_mode"],
                num_episodes=cfg["eval_episodes"], device=device,
                env_backend=cfg.get("env_backend", "auto"),
            )
            gen_gap = train_mean - test_mean
            print(f"  train={train_mean:.2f}±{train_std:.2f}  test={test_mean:.2f}±{test_std:.2f}  gap={gen_gap:.2f}")
            if use_wandb:
                wandb.log({
                    "eval/train_return": train_mean, "eval/train_return_std": train_std,
                    "eval/test_return": test_mean,   "eval/test_return_std": test_std,
                    "eval/generalization_gap": gen_gap,
                }, step=global_step)
            next_eval_step += cfg["eval_freq"]

        # ==================================================================
        # 8. Periodic checkpointing
        # ==================================================================
        if global_step >= next_checkpoint_step:
            rn_state = (
                train_env.get_reward_normalizer_state()
                if isinstance(train_env, RewardNormWrapper) else None
            )
            save_checkpoint(
                path=checkpoint_dir / f"step_{global_step}.pt",
                model=model, optimizer=optimizer,
                global_step=global_step, config=cfg,
                reward_normalizer_state=rn_state,
                rng_states=get_rng_states(), wandb_run_id=wandb_run_id,
                extra={"plr_state": plr.get_state() if plr is not None else None},
            )
            if use_wandb:
                wandb.log({"train/checkpoint_step": global_step}, step=global_step)
            next_checkpoint_step += cfg["checkpoint_freq"]

    # ==================================================================
    # 9. Final save + eval
    # ==================================================================
    rn_state = (
        train_env.get_reward_normalizer_state()
        if isinstance(train_env, RewardNormWrapper) else None
    )
    save_checkpoint(
        path=checkpoint_dir / f"step_{global_step}_final.pt",
        model=model, optimizer=optimizer,
        global_step=global_step, config=cfg,
        reward_normalizer_state=rn_state,
        rng_states=get_rng_states(), wandb_run_id=wandb_run_id,
        extra={"plr_state": plr.get_state() if plr is not None else None},
    )
    print(f"\n[done] Training complete.")

    print("\n[final eval]")
    train_mean, _ = evaluate_policy(
        model=model, env_id=cfg["env_id"], num_eval_envs=min(cfg["num_envs"], 16),
        num_levels=cfg["num_levels"], start_level=cfg["start_level"],
        distribution_mode=cfg["distribution_mode"],
        num_episodes=cfg["eval_episodes"], device=device,
        env_backend=cfg.get("env_backend", "auto"),
    )
    test_mean, _ = evaluate_policy(
        model=model, env_id=cfg["env_id"], num_eval_envs=min(cfg["num_envs"], 16),
        num_levels=0, start_level=cfg["num_levels"],
        distribution_mode=cfg["distribution_mode"],
        num_episodes=cfg["eval_episodes"], device=device,
        env_backend=cfg.get("env_backend", "auto"),
    )
    print(f"\n{'='*65}")
    print(f"  {cfg['env_id']} | {cfg['augmentation_method']}+{cfg['level_selection']} | seed={cfg['seed']}")
    print(f"  train={train_mean:.2f}  test={test_mean:.2f}  gap={train_mean - test_mean:.2f}")
    if cfg["env_id"] in EXPECTED_TEST_RETURNS:
        lo, hi = EXPECTED_TEST_RETURNS[cfg["env_id"]]
        print(f"  expected=[{lo},{hi}]  {'PASS' if lo <= test_mean <= hi else 'outside range'}")
    print(f"{'='*65}\n")

    if use_wandb:
        wandb.finish()
    train_env.close()


if __name__ == "__main__":
    main()
