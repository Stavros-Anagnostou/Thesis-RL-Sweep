"""
Utility functions: seeding, device detection, config loading, RNG state helpers.
"""

from __future__ import annotations

import random
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed Python random, NumPy, PyTorch CPU, and PyTorch CUDA/ROCm."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Deterministic ops where possible.  Note: this can slow down some ops.
    # Keeping it on for reproducibility; disable for max throughput if needed.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def get_device(device_str: str = "auto") -> torch.device:
    """
    Resolve a device string and return a torch.device.

    "auto"  → CUDA if available (works for both NVIDIA CUDA and AMD ROCm via HIP),
               otherwise CPU.
    "cuda"  → Force CUDA/ROCm (will error if unavailable).
    "cpu"   → Force CPU.

    Prints device name and available VRAM so WSL2/ROCm issues are immediately visible.
    """
    if device_str == "cpu":
        print("[device] Using CPU.")
        return torch.device("cpu")

    if device_str == "auto" or device_str == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            vram_gb = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
            print(f"[device] GPU detected: {name}  ({vram_gb:.1f} GB VRAM)")
            # Detect ROCm vs NVIDIA by checking for 'AMD' or 'Radeon' in name
            if "AMD" in name.upper() or "RADEON" in name.upper() or "GFX" in name.upper():
                print("[device] Backend: AMD ROCm (HIP translation layer)")
            else:
                print("[device] Backend: NVIDIA CUDA")
            return device
        else:
            if device_str == "cuda":
                raise RuntimeError(
                    "device='cuda' requested but torch.cuda.is_available() is False.\n"
                    "For ROCm: ensure you installed the ROCm build of PyTorch:\n"
                    "  pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/rocm5.7"
                )
            print("[device] WARNING: No GPU detected, falling back to CPU.  Training will be slow.")
            return torch.device("cpu")

    raise ValueError(f"Unknown device string '{device_str}'. Use 'auto', 'cuda', or 'cpu'.")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(yaml_path: str | Path, cli_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Load a YAML config file and apply CLI overrides on top.

    CLI override keys must match the YAML keys exactly (snake_case).
    Values of None in cli_overrides are ignored (i.e. CLI arg was not provided).

    Returns a flat dict.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path) as f:
        config: dict[str, Any] = yaml.safe_load(f)

    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is not None:
                # Convert hyphenated CLI arg names to snake_case config keys
                config_key = key.replace("-", "_")
                config[config_key] = value

    return config


# ---------------------------------------------------------------------------
# RNG state helpers (for checkpoint/resume)
# ---------------------------------------------------------------------------

def get_rng_states() -> dict[str, Any]:
    """Capture current RNG states for Python random, NumPy, and PyTorch."""
    states: dict[str, Any] = {
        "python_random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["torch_cuda"] = torch.cuda.get_rng_state()
    return states


def set_rng_states(states: dict[str, Any]) -> None:
    """Restore RNG states from a dict produced by get_rng_states()."""
    random.setstate(states["python_random"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in states:
        torch.cuda.set_rng_state(states["torch_cuda"])


# ---------------------------------------------------------------------------
# Derived config values (convenience)
# ---------------------------------------------------------------------------

def compute_derived_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Compute values derived from the raw config and add them to the dict.

    batch_size       = num_envs * num_steps
    minibatch_size   = batch_size // num_minibatches
    num_updates      = total_timesteps // batch_size
    """
    cfg = config.copy()
    cfg["batch_size"] = cfg["num_envs"] * cfg["num_steps"]
    cfg["minibatch_size"] = cfg["batch_size"] // cfg["num_minibatches"]
    cfg["num_updates"] = cfg["total_timesteps"] // cfg["batch_size"]
    return cfg


# ---------------------------------------------------------------------------
# Run name helper
# ---------------------------------------------------------------------------

def make_run_name(config: dict[str, Any]) -> str:
    """
    Produce a short, human-readable run name for checkpointing and W&B.

    Format: {env_id}__{encoder}__seed{seed}__{timestamp}
    """
    import time
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"{config['env_id']}__{config['encoder']}__seed{config['seed']}__{ts}"
