"""
Checkpoint utilities: save, load, and resume training state.

Design:
  - Each checkpoint is a single .pt file containing EVERYTHING needed to
    resume: model weights, optimiser state, global step, config, RNG states,
    reward normaliser state, and the W&B run ID.
  - A "latest.pt" copy is maintained alongside named checkpoints for easy
    resume without knowing the exact step.
  - A version field guards against silent breakage when the checkpoint format
    changes in future phases of the project.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


# Bump this whenever the checkpoint schema changes in a breaking way.
CHECKPOINT_FORMAT_VERSION = 1


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    config: dict[str, Any],
    reward_normalizer_state: dict | None,
    rng_states: dict[str, Any],
    wandb_run_id: str | None,
    extra: dict[str, Any] | None = None,
) -> None:
    """
    Save a complete training checkpoint.

    Parameters
    ----------
    path                    : destination .pt file (parent dirs created automatically)
    model                   : the ActorCritic module
    optimizer               : the Adam optimiser
    global_step             : total environment steps completed so far
    config                  : flat config dict (so the checkpoint is self-contained)
    reward_normalizer_state : dict from RewardNormWrapper.get_reward_normalizer_state(),
                              or None if reward normalisation is disabled
    rng_states              : dict from utils.get_rng_states()
    wandb_run_id            : W&B run ID string (for resume), or None
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "_format_version": CHECKPOINT_FORMAT_VERSION,
        "global_step": global_step,
        "config": config,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "reward_normalizer_state": reward_normalizer_state,
        "rng_states": rng_states,
        "wandb_run_id": wandb_run_id,
        "extra": extra or {},
    }
    torch.save(payload, path)

    # Maintain a "latest.pt" in the same directory for easy resume.
    latest_path = path.parent / "latest.pt"
    shutil.copy2(path, latest_path)
    print(f"[checkpoint] Saved step {global_step:,} → {path}  (latest.pt updated)")


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, Any]:
    """
    Load a checkpoint and restore model + optimiser state in-place.

    Returns the full checkpoint payload dict so the caller can access
    global_step, config, reward_normalizer_state, rng_states, wandb_run_id.

    Raises
    ------
    FileNotFoundError  : if the checkpoint file does not exist
    RuntimeError       : if the checkpoint format version is incompatible
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    payload: dict[str, Any] = torch.load(path, map_location=device, weights_only=False)

    # Version check — raise immediately rather than silently misbehaving.
    fmt_ver = payload.get("_format_version", 0)
    if fmt_ver != CHECKPOINT_FORMAT_VERSION:
        raise RuntimeError(
            f"Checkpoint format version mismatch: file has version {fmt_ver}, "
            f"code expects version {CHECKPOINT_FORMAT_VERSION}.  "
            f"This checkpoint was created by a different code version."
        )

    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])

    # Move optimiser tensors to the right device (they are CPU by default after load).
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    print(
        f"[checkpoint] Loaded step {payload['global_step']:,} from {path}  "
        f"(W&B run: {payload.get('wandb_run_id', 'N/A')})"
    )
    return payload


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """
    Return the path to 'latest.pt' in checkpoint_dir, or None if it doesn't exist.

    Useful for automatic resume: check this before starting a run.
    """
    latest = Path(checkpoint_dir) / "latest.pt"
    return latest if latest.exists() else None
