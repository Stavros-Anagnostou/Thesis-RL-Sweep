#!/usr/bin/env python3
"""
Pre-flight check script.

Run this BEFORE starting any training to catch environment / hardware issues
early.  Prints a clear PASS/FAIL for each check.

Usage:
    python scripts/validate_setup.py
"""

from __future__ import annotations

import sys
import importlib
import subprocess

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
WARN = "\033[93m⚠ WARN\033[0m"


def check(label: str, ok: bool, detail: str = "") -> bool:
    status = PASS if ok else FAIL
    line = f"  {status}  {label}"
    if detail:
        line += f"  →  {detail}"
    print(line)
    return ok


def section(title: str) -> None:
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


# ---------------------------------------------------------------------------
# 1. Python version
# ---------------------------------------------------------------------------
section("Python")
py_ver = sys.version_info
check(
    f"Python {py_ver.major}.{py_ver.minor}.{py_ver.micro}",
    py_ver >= (3, 11) and py_ver < (3, 13),
    "requires >=3.11, <3.13",
)


# ---------------------------------------------------------------------------
# 2. PyTorch + CUDA/ROCm
# ---------------------------------------------------------------------------
section("PyTorch & GPU")

try:
    import torch

    torch_ver = torch.__version__
    cuda_ok = torch.cuda.is_available()

    check(f"torch {torch_ver}", True)
    check("torch.cuda.is_available()", cuda_ok,
          "No GPU found — training will use CPU and be very slow" if not cuda_ok else "")

    if cuda_ok:
        idx  = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        props = torch.cuda.get_device_properties(idx)
        vram_gb = props.total_memory / (1024 ** 3)
        check(f"GPU: {name}", True, f"{vram_gb:.1f} GB VRAM")

        # Detect ROCm
        is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
        if is_rocm:
            check("Backend: AMD ROCm (HIP)", True, f"HIP version {torch.version.hip}")
        else:
            check("Backend: NVIDIA CUDA", True, f"CUDA version {torch.version.cuda}")

        # Quick tensor op on GPU
        try:
            t = torch.ones(1024, 1024, device="cuda")
            _ = (t @ t).sum()
            check("GPU matmul smoke test", True)
        except Exception as e:
            check("GPU matmul smoke test", False, str(e))
    else:
        print(f"  {WARN}  Skipping GPU checks (no GPU detected)")
        print(
            "\n  If you have an AMD GPU under WSL2, make sure:\n"
            "  1. ROCm drivers are installed on Windows host: https://rocm.docs.amd.com/\n"
            "  2. PyTorch ROCm build is installed:\n"
            "     pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/rocm5.7\n"
            "  3. /dev/kfd and /dev/dri are accessible inside WSL2\n"
            "     (add 'render' group: sudo usermod -aG render $USER)"
        )

except ImportError:
    check("import torch", False, "Run: pip install torch (see README for ROCm/CUDA instructions)")


# ---------------------------------------------------------------------------
# 3. EnvPool (primary backend)
# ---------------------------------------------------------------------------
section("EnvPool (primary env backend)")

try:
    import envpool

    check(f"envpool {envpool.__version__}", True)

    # Verify Procgen support — task IDs are "{Game}{Mode}-v0" e.g. "CoinrunEasy-v0"
    all_envs = envpool.list_all_envs()
    has_procgen = any("CoinrunEasy" in e for e in all_envs)
    check("EnvPool Procgen support", has_procgen,
          "Procgen tasks not found in envpool.list_all_envs()" if not has_procgen else "")

    if has_procgen:
        try:
            env = envpool.make(
                "CoinrunEasy-v0",
                env_type="gymnasium",
                num_envs=2,
                batch_size=2,
                num_levels=10,
                start_level=0,
                channel_first=False,
                seed=42,
            )
            obs, info = env.reset()
            check("EnvPool Procgen reset()", True, f"obs shape={obs.shape}, dtype={obs.dtype}")

            expected_shape = (2, 64, 64, 3)
            check("Observation shape == (N, 64, 64, 3)", obs.shape == expected_shape,
                  f"got {obs.shape}")
            check("Observation dtype == uint8", str(obs.dtype) == "uint8",
                  f"got {obs.dtype}")

            import numpy as np
            for _ in range(100):
                actions = np.random.randint(0, env.action_space.n, size=2)
                obs, reward, term, trunc, info = env.step(actions)
            check("100 steps without error", True)

        except Exception as e:
            check("EnvPool Procgen smoke test", False, str(e))

except ImportError:
    check("import envpool", False,
          "Run: pip install envpool  (this is the primary backend for faster training)")


# ---------------------------------------------------------------------------
# 3b. Procgen native (optional fallback)
# ---------------------------------------------------------------------------
section("Procgen native (fallback)")

try:
    import procgen
    from procgen import ProcgenEnv

    check(f"procgen {procgen.__version__} (fallback available)", True)

    try:
        env = ProcgenEnv(num_envs=2, env_name="coinrun", num_levels=10,
                         start_level=0, distribution_mode="easy")
        obs_dict = env.reset()
        obs = obs_dict["rgb"]
        check("ProcgenEnv smoke test", True, f"obs shape={obs.shape}")
        env.close()
    except Exception as e:
        check("ProcgenEnv smoke test", False, str(e))

except ImportError:
    print(f"  {WARN}  procgen not installed (optional -- EnvPool is the primary backend)")
    print(f"         Install as fallback: pip install procgen==0.10.7")


# ---------------------------------------------------------------------------
# 4. Gymnasium
# ---------------------------------------------------------------------------
section("Gymnasium")

try:
    import gymnasium
    check(f"gymnasium {gymnasium.__version__}", True)
except ImportError:
    check("import gymnasium", False, "Run: pip install gymnasium==0.29.1")


# ---------------------------------------------------------------------------
# 5. W&B
# ---------------------------------------------------------------------------
section("Weights & Biases")

try:
    import wandb

    check(f"wandb {wandb.__version__}", True)

    # Check login status without triggering a prompt.
    api_key_set = bool(wandb.api.api_key) if hasattr(wandb, "api") else False
    # Alternative: check environment variable
    import os
    api_key_env = bool(os.environ.get("WANDB_API_KEY"))
    logged_in = api_key_set or api_key_env

    if logged_in:
        check("W&B API key configured", True)
    else:
        # Try calling wandb login check silently.
        try:
            result = subprocess.run(
                ["wandb", "status"], capture_output=True, text=True, timeout=5
            )
            logged_in = "Logged in" in result.stdout or result.returncode == 0
        except Exception:
            logged_in = False
        status = check(
            "W&B login status",
            logged_in,
            "Run: wandb login" if not logged_in else "",
        )

except ImportError:
    check("import wandb", False, "Run: pip install wandb==0.16.6")


# ---------------------------------------------------------------------------
# 6. Other dependencies
# ---------------------------------------------------------------------------
section("Other dependencies")

for pkg, import_name in [
    ("numpy",                   "numpy"),
    ("pyyaml",                  "yaml"),
    ("matplotlib",              "matplotlib"),
    ("seaborn",                 "seaborn"),
    ("rliable",                 "rliable"),
    ("opencv-python-headless",  "cv2"),
    ("tqdm",                    "tqdm"),
    ("imageio",                 "imageio"),
]:
    try:
        mod = importlib.import_module(import_name)
        ver = getattr(mod, "__version__", "?")
        check(f"{pkg} {ver}", True)
    except ImportError:
        check(f"import {import_name}", False, f"Run: pip install {pkg}")


# ---------------------------------------------------------------------------
# 7. Project structure check
# ---------------------------------------------------------------------------
section("Project structure")

from pathlib import Path
expected_files = [
    "configs/ppo_procgen_baseline.yaml",
    "src/__init__.py",
    "src/train_ppo.py",
    "src/evaluate.py",
    "src/networks.py",
    "src/env_utils.py",
    "src/checkpoint.py",
    "src/utils.py",
]

root = Path(__file__).parent.parent
for rel_path in expected_files:
    p = root / rel_path
    check(rel_path, p.exists())


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*55}")
print("  Pre-flight check complete.")
print("  Fix any ✗ FAIL items before starting training.")
print(f"{'='*55}\n")
