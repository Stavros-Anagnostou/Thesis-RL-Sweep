"""
Standalone evaluation script.

Loads a trained checkpoint and evaluates the policy on both training levels
and held-out test levels.  Optionally renders episodes to .mp4 video.

Usage:
    python src/evaluate.py --checkpoint checkpoints/coinrun/run_name/latest.pt
    python src/evaluate.py --checkpoint path/to/ckpt.pt --render-episodes 3
    python src/evaluate.py --checkpoint path/to/ckpt.pt --deterministic
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.networks import ActorCritic
from src.env_utils import make_eval_envs
from src.checkpoint import load_checkpoint
from src.utils import get_device, set_seed
from src.train_ppo import EXPECTED_TEST_RETURNS

import torch.optim as optim


# ---------------------------------------------------------------------------
# Evaluation core
# ---------------------------------------------------------------------------

def run_evaluation(
    model: ActorCritic,
    env_id: str,
    num_envs: int,
    num_levels: int,
    start_level: int,
    distribution_mode: str,
    num_episodes: int,
    device: torch.device,
    deterministic: bool,
    render_episodes: int = 0,
    video_dir: Path | None = None,
    env_backend: str | None = None,
) -> dict[str, float]:
    """
    Evaluate a policy and optionally save rendered episodes to video.

    Returns a dict with mean, std, min, max return.
    """
    env = make_eval_envs(
        env_id=env_id,
        num_envs=num_envs,
        num_levels=num_levels,
        start_level=start_level,
        distribution_mode=distribution_mode,
        env_backend=env_backend,
    )

    obs = env.reset()
    episode_returns: list[float] = []
    current_returns = np.zeros(num_envs, dtype=np.float32)

    # Video recording state.
    recording = render_episodes > 0 and video_dir is not None
    frames: list[np.ndarray] = []
    recorded_episodes = 0

    model.eval()
    with torch.no_grad():
        while len(episode_returns) < num_episodes:
            obs_t = torch.tensor(obs, dtype=torch.uint8, device=device)

            if deterministic:
                features = model.encoder(obs_t)
                logits = model.policy_head(features)
                actions = logits.argmax(dim=-1).cpu().numpy()
            else:
                actions, _, _, _ = model.get_action_and_value(obs_t)
                actions = actions.cpu().numpy()

            # Capture frame for video (first env only).
            if recording and recorded_episodes < render_episodes:
                frames.append(obs[0].copy())  # (H, W, C) uint8

            obs, rewards, dones, _ = env.step(actions)
            current_returns += rewards

            for i, done in enumerate(dones):
                if done:
                    episode_returns.append(float(current_returns[i]))
                    current_returns[i] = 0.0
                    if i == 0 and recording and recorded_episodes < render_episodes:
                        # Save this episode's frames to video.
                        if frames:
                            _save_video(frames, video_dir, recorded_episodes, env_id)
                        frames = []
                        recorded_episodes += 1
                    if len(episode_returns) >= num_episodes:
                        break

    env.close()

    arr = np.array(episode_returns[:num_episodes])
    return {
        "mean":   float(arr.mean()),
        "std":    float(arr.std()),
        "min":    float(arr.min()),
        "max":    float(arr.max()),
        "median": float(np.median(arr)),
    }


def _save_video(
    frames: list[np.ndarray],
    video_dir: Path,
    episode_idx: int,
    env_id: str,
) -> None:
    """Save a list of (H, W, C) uint8 frames as an .mp4 file using OpenCV.

    Uses opencv-python-headless (already a project dependency) instead of
    imageio so there is no ffmpeg plugin registration issue in WSL2.
    """
    import cv2  # opencv-python-headless is in pyproject.toml dependencies

    video_dir.mkdir(parents=True, exist_ok=True)
    out_path = video_dir / f"{env_id}_episode_{episode_idx:03d}.mp4"

    if not frames:
        print("  [video] No frames to save.")
        return

    h, w, _ = frames[0].shape
    # mp4v codec is universally available via OpenCV without extra system packages.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, 15, (w, h))

    for frame in frames:
        # OpenCV expects BGR; Procgen frames are RGB.
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"  [video] Saved {len(frames)} frames → {out_path}")


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(
    env_id: str,
    train_stats: dict[str, float],
    test_stats: dict[str, float],
) -> None:
    """Print a formatted comparison against published baselines."""
    gen_gap = train_stats["mean"] - test_stats["mean"]

    print(f"\n{'='*65}")
    print(f"  Evaluation Results — {env_id}")
    print(f"{'='*65}")
    print(f"  {'Metric':<28}  {'Train levels':>14}  {'Test levels':>14}")
    print(f"  {'-'*60}")
    for k in ["mean", "std", "median", "min", "max"]:
        print(f"  {k.capitalize() + ' return':<28}  {train_stats[k]:>14.3f}  {test_stats[k]:>14.3f}")
    print(f"  {'Generalization gap':<28}  {gen_gap:>14.3f}")

    if env_id in EXPECTED_TEST_RETURNS:
        lo, hi = EXPECTED_TEST_RETURNS[env_id]
        print(f"\n  Published baseline (test): [{lo:.1f}, {hi:.1f}]")
        if lo <= test_stats["mean"] <= hi:
            print(f"  Status: ✓ PASS  (within expected range)")
        elif test_stats["mean"] < lo * 0.8:
            pct = (lo - test_stats["mean"]) / lo * 100
            print(f"  Status: ✗ FAIL  ({pct:.0f}% below lower bound — check for bugs)")
        else:
            pct = (test_stats["mean"] - hi) / hi * 100
            print(f"  Status: ✓ ABOVE range  ({pct:.0f}% above upper bound — great!)")
    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO checkpoint")
    parser.add_argument("--checkpoint",        type=str,  required=True, help="Path to .pt checkpoint")
    parser.add_argument("--num-eval-envs",     type=int,  default=16,    help="Parallel eval envs")
    parser.add_argument("--num-episodes",      type=int,  default=None,  help="Override eval_episodes from config")
    parser.add_argument("--deterministic",     action="store_true",      help="Use argmax policy")
    parser.add_argument("--render-episodes",   type=int,  default=0,     help="Number of episodes to render to video")
    parser.add_argument("--video-dir",         type=str,  default="videos", help="Output directory for videos")
    parser.add_argument("--device",            type=str,  default="auto")
    parser.add_argument("--seed",              type=int,  default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    set_seed(args.seed)

    ckpt_path = Path(args.checkpoint)
    print(f"[eval] Loading checkpoint: {ckpt_path}")

    # Load checkpoint to get config and model weights.
    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = raw_ckpt["config"]
    global_step = raw_ckpt["global_step"]

    print(f"[eval] env_id={cfg['env_id']}  encoder={cfg['encoder']}  step={global_step:,}")

    # Determine num_actions from a temporary env.
    env_backend = cfg.get("env_backend", "auto")
    tmp_env = make_eval_envs(
        env_id=cfg["env_id"],
        num_envs=1,
        num_levels=1,
        start_level=0,
        distribution_mode=cfg["distribution_mode"],
        env_backend=env_backend,
    )
    num_actions = tmp_env.action_space.n
    tmp_env.close()

    # Build model and load weights.
    model = ActorCritic(encoder=cfg["encoder"], num_actions=num_actions).to(device)
    optimizer = optim.Adam(model.parameters())  # dummy optimiser for load API compatibility
    load_checkpoint(ckpt_path, model, optimizer, device)

    num_episodes = args.num_episodes or cfg.get("eval_episodes", 100)
    video_dir = Path(args.video_dir) if args.render_episodes > 0 else None

    # --- Train-level evaluation ---
    print(f"\n[eval] Train levels  (num_levels={cfg['num_levels']}, start_level={cfg['start_level']})")
    train_stats = run_evaluation(
        model=model,
        env_id=cfg["env_id"],
        num_envs=args.num_eval_envs,
        num_levels=cfg["num_levels"],
        start_level=cfg["start_level"],
        distribution_mode=cfg["distribution_mode"],
        num_episodes=num_episodes,
        device=device,
        deterministic=args.deterministic,
        render_episodes=args.render_episodes,
        video_dir=video_dir,
        env_backend=env_backend,
    )

    # --- Test-level evaluation ---
    print(f"[eval] Test levels   (num_levels=0, start_level={cfg['num_levels']})")
    test_stats = run_evaluation(
        model=model,
        env_id=cfg["env_id"],
        num_envs=args.num_eval_envs,
        num_levels=0,
        start_level=cfg["num_levels"],
        distribution_mode=cfg["distribution_mode"],
        num_episodes=num_episodes,
        device=device,
        deterministic=args.deterministic,
        env_backend=env_backend,
    )

    print_comparison_table(cfg["env_id"], train_stats, test_stats)


if __name__ == "__main__":
    main()
