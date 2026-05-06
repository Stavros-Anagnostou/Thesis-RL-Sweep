"""
DQN training script for Procgen — CleanRL-style.

Implements standard DQN with:
  - IMPALA-CNN encoder (same as PPO for fair comparison)
  - Experience replay buffer
  - Epsilon-greedy exploration with linear decay
  - Hard target network updates
  - Reward clipping to [-1, 1] (standard DQN convention)

NOTE: DQN is expected to perform WORSE than PPO on Procgen, especially on
generalization to unseen levels.  This is a key thesis finding — DQN's
off-policy nature and reliance on epsilon-greedy exploration interact poorly
with the diverse level distributions that PPO's on-policy rollouts handle
naturally.  The PPO vs DQN generalization comparison is itself a thesis result.

Usage:
    python src/train_dqn.py --env-id coinrun --seed 1
    python src/train_dqn.py --config configs/dqn_procgen_baseline.yaml --env-id starpilot
"""

from __future__ import annotations

import argparse
import random
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import wandb

from src.networks import QNetwork
from src.env_utils import make_procgen_envs, make_eval_envs, RewardNormWrapper
from src.checkpoint import save_checkpoint
from src.utils import (
    set_seed,
    get_device,
    load_config,
    make_run_name,
)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """
    Simple circular replay buffer storing (obs, action, reward, next_obs, done).

    Observations are stored as uint8 (B, H, W, C) to minimise RAM.
    At sample time they are converted to float32 on-device.
    """

    def __init__(self, capacity: int, obs_shape: tuple) -> None:
        self.capacity  = capacity
        self.obs_shape = obs_shape
        self.pos  = 0
        self.size = 0

        # Pre-allocate numpy arrays to avoid repeated allocation.
        self.obs      = np.zeros((capacity,) + obs_shape, dtype=np.uint8)
        self.next_obs = np.zeros((capacity,) + obs_shape, dtype=np.uint8)
        self.actions  = np.zeros(capacity, dtype=np.int64)
        self.rewards  = np.zeros(capacity, dtype=np.float32)
        self.dones    = np.zeros(capacity, dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: float,
    ) -> None:
        self.obs[self.pos]      = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos]  = action
        self.rewards[self.pos]  = reward
        self.dones[self.pos]    = done
        self.pos  = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(
        self,
        obs: np.ndarray,      # (N, H, W, C)
        actions: np.ndarray,  # (N,)
        rewards: np.ndarray,  # (N,)
        next_obs: np.ndarray, # (N, H, W, C)
        dones: np.ndarray,    # (N,)
    ) -> None:
        """Add a vectorised step (N transitions) in one call."""
        n = len(actions)
        for i in range(n):
            self.add(obs[i], int(actions[i]), float(rewards[i]), next_obs[i], float(dones[i]))

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        """Sample a random minibatch and move to device."""
        idxs = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs":      torch.tensor(self.obs[idxs],      dtype=torch.uint8,   device=device),
            "next_obs": torch.tensor(self.next_obs[idxs], dtype=torch.uint8,   device=device),
            "actions":  torch.tensor(self.actions[idxs],  dtype=torch.long,    device=device),
            "rewards":  torch.tensor(self.rewards[idxs],  dtype=torch.float32, device=device),
            "dones":    torch.tensor(self.dones[idxs],    dtype=torch.float32, device=device),
        }

    def __len__(self) -> int:
        return self.size


# ---------------------------------------------------------------------------
# Evaluation helper (identical logic to train_ppo.py)
# ---------------------------------------------------------------------------

def evaluate_policy(
    q_net: QNetwork,
    env_id: str,
    num_eval_envs: int,
    num_levels: int,
    start_level: int,
    distribution_mode: str,
    num_episodes: int,
    device: torch.device,
    env_backend: str | None = None,
) -> tuple[float, float]:
    """Evaluate with greedy policy (argmax Q). Returns (mean, std) episode return."""
    eval_env = make_eval_envs(
        env_id=env_id, num_envs=num_eval_envs,
        num_levels=num_levels, start_level=start_level,
        distribution_mode=distribution_mode,
        env_backend=env_backend,
    )
    obs = eval_env.reset()
    episode_returns: list[float] = []
    current_returns = np.zeros(num_eval_envs, dtype=np.float32)

    q_net.eval()
    with torch.no_grad():
        while len(episode_returns) < num_episodes:
            obs_t = torch.tensor(obs, dtype=torch.uint8, device=device)
            actions = q_net.get_q_values(obs_t).argmax(dim=-1).cpu().numpy()
            obs, rewards, dones, _ = eval_env.step(actions)
            current_returns += rewards
            for i, done in enumerate(dones):
                if done:
                    episode_returns.append(float(current_returns[i]))
                    current_returns[i] = 0.0
                    if len(episode_returns) >= num_episodes:
                        break
    q_net.train()
    eval_env.close()
    arr = np.array(episode_returns[:num_episodes])
    return float(arr.mean()), float(arr.std())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DQN on Procgen (thesis)")
    parser.add_argument("--config",          type=str, default="configs/dqn_procgen_baseline.yaml")
    parser.add_argument("--env-id",          type=str, default=None)
    parser.add_argument("--seed",            type=int, default=None)
    parser.add_argument("--encoder",         type=str, default=None, choices=["impala", "nature", "small"])
    parser.add_argument("--num-levels",      type=int, default=None)
    parser.add_argument("--device",          type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--no-wandb",        action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    cli_overrides = {
        "env_id":          args.env_id,
        "seed":            args.seed,
        "encoder":         args.encoder,
        "num_levels":      args.num_levels,
        "device":          args.device,
        "total_timesteps": args.total_timesteps,
    }
    cfg = load_config(args.config, {k: v for k, v in cli_overrides.items() if v is not None})

    # Derived values
    cfg.setdefault("train_freq",         4)
    cfg.setdefault("target_update_freq", 10_000)
    cfg.setdefault("learning_starts",    10_000)
    cfg.setdefault("eps_start",          1.0)
    cfg.setdefault("eps_end",            0.02)
    cfg.setdefault("eps_decay_steps",    1_000_000)
    cfg.setdefault("reward_clipping",    True)
    cfg.setdefault("buffer_size",        200_000)
    cfg.setdefault("batch_size",         32)

    device = get_device(cfg["device"])
    set_seed(cfg["seed"])

    run_name = make_run_name(cfg)
    checkpoint_dir = Path("checkpoints") / cfg["env_id"] / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # --- W&B ---
    use_wandb = not args.no_wandb
    wandb_tags = ["dqn", cfg["env_id"], f"seed_{cfg['seed']}", cfg.get("encoder", "impala")]
    wandb_run_id = "disabled"
    if use_wandb:
        run = wandb.init(
            project=cfg["wandb_project"],
            entity=cfg.get("wandb_entity") or None,
            name=run_name, config=cfg,
            tags=wandb_tags, save_code=True,
        )
        wandb_run_id = run.id
        print(f"[wandb] {run.url}")

    # --- Environment ---
    # DQN uses a single-env loop (num_envs for data collection speed).
    # We use a small num_envs (default from config, typically 8-16 for DQN)
    # since DQN doesn't use vectorised advantage estimation.
    num_envs = min(cfg.get("num_envs", 8), 16)   # cap at 16 for DQN

    train_env = make_procgen_envs(
        env_id=cfg["env_id"], num_envs=num_envs,
        num_levels=cfg["num_levels"], start_level=cfg["start_level"],
        distribution_mode=cfg["distribution_mode"],
        gamma=cfg["gamma"], normalize_reward=False,   # DQN uses reward clipping
        env_backend=cfg.get("env_backend", "auto"),
        seed=cfg["seed"],
    )

    obs_space = train_env.observation_space
    num_actions: int = train_env.action_space.n
    print(f"[env] {cfg['env_id']}  obs={obs_space.shape}  actions={num_actions}  envs={num_envs}")

    # --- Networks ---
    encoder_name = cfg.get("encoder", "impala")
    q_net      = QNetwork(encoder=encoder_name, num_actions=num_actions).to(device)
    target_net = QNetwork(encoder=encoder_name, num_actions=num_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()  # target net is never trained directly

    optimizer = optim.Adam(q_net.parameters(), lr=cfg["learning_rate"])

    total_params = sum(p.numel() for p in q_net.parameters() if p.requires_grad)
    print(f"[model] DQN {encoder_name.upper()}-CNN  params={total_params:,}")

    # --- Replay buffer ---
    replay_buffer = ReplayBuffer(
        capacity=cfg["buffer_size"],
        obs_shape=obs_space.shape,
    )

    # --- Training state ---
    obs = train_env.reset()          # (num_envs, H, W, C) uint8
    ep_returns = np.zeros(num_envs, dtype=np.float32)
    completed_returns: list[float] = []

    global_step = 0
    total_timesteps = cfg["total_timesteps"]

    next_checkpoint_step = cfg["checkpoint_freq"]
    next_eval_step       = cfg["eval_freq"]

    print(f"\n{'='*60}")
    print(f"  DQN  |  env={cfg['env_id']}  encoder={encoder_name}")
    print(f"  total_steps={total_timesteps:,}  buffer={cfg['buffer_size']:,}")
    print(f"  eps {cfg['eps_start']}→{cfg['eps_end']} over {cfg['eps_decay_steps']:,} steps")
    print(f"{'='*60}\n")

    start_time = time.time()
    losses: list[float] = []

    while global_step < total_timesteps:

        # --- Epsilon-greedy action selection ---
        progress = min(global_step / cfg["eps_decay_steps"], 1.0)
        epsilon = cfg["eps_start"] + progress * (cfg["eps_end"] - cfg["eps_start"])

        if random.random() < epsilon:
            actions = np.array([train_env.action_space.sample() for _ in range(num_envs)])
        else:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.uint8, device=device)
                actions = q_net.get_q_values(obs_t).argmax(dim=-1).cpu().numpy()

        next_obs, rewards, dones, infos = train_env.step(actions)

        # Reward clipping — standard DQN practice (prevents Q-value divergence)
        if cfg["reward_clipping"]:
            rewards = np.clip(rewards, -1.0, 1.0)

        # Add transitions to replay buffer
        replay_buffer.add_batch(obs, actions, rewards, next_obs, dones.astype(np.float32))

        ep_returns += rewards
        for i, done in enumerate(dones):
            if done:
                completed_returns.append(float(ep_returns[i]))
                ep_returns[i] = 0.0

        obs = next_obs
        global_step += num_envs

        # --- DQN update (after filling buffer past learning_starts) ---
        if (
            len(replay_buffer) >= cfg["learning_starts"]
            and global_step % cfg["train_freq"] == 0
        ):
            batch = replay_buffer.sample(cfg["batch_size"], device)

            with torch.no_grad():
                # Standard DQN (not Double DQN): max over target net Q-values
                next_q = target_net.get_q_values(batch["next_obs"]).max(dim=1).values
                td_target = batch["rewards"] + cfg["gamma"] * next_q * (1.0 - batch["dones"])

            current_q = q_net.get_q_values(batch["obs"])
            q_selected = current_q.gather(1, batch["actions"].unsqueeze(1)).squeeze(1)

            loss = nn.functional.smooth_l1_loss(q_selected, td_target)   # Huber loss

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping — important for DQN stability
            nn.utils.clip_grad_norm_(q_net.parameters(), 10.0)
            optimizer.step()
            losses.append(loss.item())

        # --- Hard target network update ---
        if global_step % cfg["target_update_freq"] == 0:
            target_net.load_state_dict(q_net.state_dict())

        # --- Logging every ~10k steps ---
        if global_step % 10_000 < num_envs:
            elapsed = time.time() - start_time
            sps = global_step / elapsed
            mean_loss = np.mean(losses[-100:]) if losses else 0.0
            log_dict: dict[str, Any] = {
                "charts/SPS":     sps,
                "charts/epsilon": epsilon,
                "losses/td_loss": mean_loss,
                "train/global_step": global_step,
                "replay/buffer_fill": len(replay_buffer) / cfg["buffer_size"],
            }
            if completed_returns:
                log_dict["charts/episodic_return"] = np.mean(completed_returns)
                completed_returns.clear()

            if use_wandb:
                wandb.log(log_dict, step=global_step)
            print(
                f"  step {global_step:>10,}  SPS {sps:>6,.0f}  "
                f"eps={epsilon:.3f}  loss={mean_loss:.4f}  "
                f"buf={len(replay_buffer):,}"
            )

        # --- Periodic evaluation ---
        if global_step >= next_eval_step:
            print(f"\n[eval] step {global_step:,}...")
            train_mean, train_std = evaluate_policy(
                q_net=q_net, env_id=cfg["env_id"],
                num_eval_envs=min(num_envs, 16),
                num_levels=cfg["num_levels"], start_level=cfg["start_level"],
                distribution_mode=cfg["distribution_mode"],
                num_episodes=cfg["eval_episodes"], device=device,
                env_backend=cfg.get("env_backend", "auto"),
            )
            test_mean, test_std = evaluate_policy(
                q_net=q_net, env_id=cfg["env_id"],
                num_eval_envs=min(num_envs, 16),
                num_levels=0, start_level=cfg["num_levels"],
                distribution_mode=cfg["distribution_mode"],
                num_episodes=cfg["eval_episodes"], device=device,
                env_backend=cfg.get("env_backend", "auto"),
            )
            print(f"  train={train_mean:.2f}±{train_std:.2f}  test={test_mean:.2f}±{test_std:.2f}")
            if use_wandb:
                wandb.log({
                    "eval/train_return": train_mean, "eval/train_return_std": train_std,
                    "eval/test_return": test_mean,   "eval/test_return_std": test_std,
                    "eval/generalization_gap": train_mean - test_mean,
                }, step=global_step)
            next_eval_step += cfg["eval_freq"]

        # --- Periodic checkpointing ---
        if global_step >= next_checkpoint_step:
            save_checkpoint(
                path=checkpoint_dir / f"step_{global_step}.pt",
                model=q_net, optimizer=optimizer,
                global_step=global_step, config=cfg,
                reward_normalizer_state=None,
                rng_states={},  # DQN replay buffer state not saved (too large)
                wandb_run_id=wandb_run_id,
            )
            next_checkpoint_step += cfg["checkpoint_freq"]

    # Final checkpoint + eval
    save_checkpoint(
        path=checkpoint_dir / f"step_{global_step}_final.pt",
        model=q_net, optimizer=optimizer,
        global_step=global_step, config=cfg,
        reward_normalizer_state=None, rng_states={},
        wandb_run_id=wandb_run_id,
    )
    print(f"\n[done] DQN training complete.")

    train_mean, train_std = evaluate_policy(
        q_net=q_net, env_id=cfg["env_id"], num_eval_envs=min(num_envs, 16),
        num_levels=cfg["num_levels"], start_level=cfg["start_level"],
        distribution_mode=cfg["distribution_mode"],
        num_episodes=cfg["eval_episodes"], device=device,
        env_backend=cfg.get("env_backend", "auto"),
    )
    test_mean, test_std = evaluate_policy(
        q_net=q_net, env_id=cfg["env_id"], num_eval_envs=min(num_envs, 16),
        num_levels=0, start_level=cfg["num_levels"],
        distribution_mode=cfg["distribution_mode"],
        num_episodes=cfg["eval_episodes"], device=device,
        env_backend=cfg.get("env_backend", "auto"),
    )
    print(f"\n{'='*60}")
    print(f"  DQN FINAL  |  {cfg['env_id']} seed={cfg['seed']}")
    print(f"  train={train_mean:.2f}±{train_std:.2f}  test={test_mean:.2f}±{test_std:.2f}")
    print(f"  gap={train_mean - test_mean:.2f}")
    print(f"{'='*60}\n")

    if use_wandb:
        wandb.finish()
    train_env.close()


if __name__ == "__main__":
    main()
