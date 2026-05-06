# MSc Thesis RL — Phase 1: PPO Procgen Baseline

**Generalization and Robustness of RL Agents in Procedurally Generated Game Environments**

---

## Hardware & Software Stack

| Component | Spec |
|-----------|------|
| OS | Windows 11 + WSL2 (Ubuntu 22.04) |
| GPU | AMD Radeon RX 9070 XT (16 GB VRAM) |
| CPU | Ryzen 7 5800X (8c/16t) |
| RAM | 34 GB DDR4 (26 GB allocated to WSL2) |
| Framework | PyTorch (ROCm build) |
| Python | 3.10 or 3.11 |

---

## 1. WSL2 Setup (Windows side)

### 1.1 `.wslconfig` (in `C:\Users\<YourUser>\`)
```ini
[wsl2]
memory=26GB
processors=14
gpuSupport=true
```

### 1.2 ROCm GPU Passthrough
Ensure your AMD GPU is visible inside WSL2:
```bash
ls /dev/kfd /dev/dri/renderD*
# Should output: /dev/kfd  /dev/dri/renderD128

# Add your user to the render and video groups:
sudo usermod -aG render,video $USER
# Log out and back in (or: newgrp render)
```

---

## 2. Environment Setup (inside WSL2)

### 2.1 Create Python environment
```bash
# Using conda (recommended for ROCm PyTorch):
conda create -n thesis-rl python=3.10 -y
conda activate thesis-rl

# Or with venv:
python3.10 -m venv .venv
source .venv/bin/activate
```

### 2.2 Install PyTorch with ROCm support
```bash
# ROCm 5.7 build (matches RX 9070 XT / RDNA3):
pip install torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/rocm5.7

# Verify GPU is detected:
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True  AMD Radeon RX 9070 XT (or similar)
```

> **Note for NVIDIA users / Colab:** Replace the ROCm index URL with:
> `--index-url https://download.pytorch.org/whl/cu121`
> No code changes needed — the HIP translation is transparent.

### 2.3 Install project dependencies
```bash
# Clone / navigate to the project root
cd thesis_rl

# Install the project in editable mode (picks up pyproject.toml):
pip install -e .

# Procgen requires CMake and build tools:
sudo apt-get install -y cmake build-essential
pip install procgen==0.10.7
```

### 2.4 Weights & Biases login
```bash
wandb login
# Paste your API key from https://wandb.ai/authorize
```

---

## 3. Pre-flight Check

```bash
python scripts/validate_setup.py
```

All items should show ✓ PASS before you start training.

---

## 4. Running Experiments

### Single run (one game, one seed)
```bash
bash scripts/run_baseline.sh coinrun 1
# Or explicitly:
python src/train_ppo.py --env-id coinrun --seed 1
```

### Full sweep (4 games × 5 seeds)
```bash
bash scripts/run_sweep.sh
# ~20–40 hours total on RX 9070 XT depending on the game
```

### Resume from checkpoint
```bash
python src/train_ppo.py --resume checkpoints/coinrun/run_name/latest.pt
```

---

## 5. Evaluation

```bash
# Basic evaluation (stochastic policy):
python src/evaluate.py --checkpoint checkpoints/coinrun/run_name/latest.pt

# Deterministic policy + render 3 episodes to video:
python src/evaluate.py \
    --checkpoint checkpoints/coinrun/run_name/latest.pt \
    --deterministic \
    --render-episodes 3 \
    --video-dir videos/
```

---

## 6. Plotting (after sweep completes)

```bash
python analysis/plot_baselines.py \
    --project thesis-rl-baselines \
    --output-dir figures/
```

Figures are saved as PDFs in `figures/`.

---

## 7. Expected Baseline Performance

After 25 M steps on easy mode (200 training levels), expected **test** returns:

| Game | Expected Test Return |
|------|---------------------|
| CoinRun | 8.5 – 9.5 |
| StarPilot | 25 – 35 |
| BigFish | 10 – 20 |
| Ninja | 6 – 8 |

Source: Cobbe et al. (2020), CleanRL Procgen benchmarks.

---

## 8. Project Structure

```
thesis_rl/
├── pyproject.toml                # Dependencies
├── README.md                     # This file
├── configs/
│   └── ppo_procgen_baseline.yaml # All hyperparameters
├── src/
│   ├── train_ppo.py              # Main training loop (CleanRL-style)
│   ├── evaluate.py               # Standalone evaluation
│   ├── networks.py               # IMPALA-CNN + Nature-CNN + ActorCritic
│   ├── env_utils.py              # Procgen env creation + reward normalisation
│   ├── checkpoint.py             # Save / load / resume
│   └── utils.py                  # Seeding, device, config helpers
├── scripts/
│   ├── validate_setup.py         # Pre-flight checks
│   ├── run_baseline.sh           # Single run
│   └── run_sweep.sh              # Multi-game multi-seed sweep
├── analysis/
│   └── plot_baselines.py         # W&B → PDF figures
└── checkpoints/                  # Created at runtime
```

---

## 9. Troubleshooting

**`torch.cuda.is_available()` returns False on ROCm:**
- Check `/dev/kfd` exists: `ls /dev/kfd`
- Check render group: `groups | grep render`
- Try `HSA_OVERRIDE_GFX_VERSION=11.0.0` if your GPU is very new:
  ```bash
  export HSA_OVERRIDE_GFX_VERSION=11.0.0
  python -c "import torch; print(torch.cuda.is_available())"
  ```

**Procgen build fails:**
```bash
sudo apt-get install -y qtbase5-dev  # may be needed on some Ubuntu versions
pip install procgen==0.10.7 --no-build-isolation
```

**Low SPS (steps per second):**
- Normal range: 3000–6000 SPS on RX 9070 XT with IMPALA-CNN + 64 envs
- If <1000 SPS: check WSL2 memory allocation, ensure GPU is being used
- Try `rocm-smi` inside WSL2 to monitor GPU utilisation

**W&B connection issues inside WSL2:**
```bash
export WANDB_MODE=offline  # collect locally, sync later with: wandb sync
```
