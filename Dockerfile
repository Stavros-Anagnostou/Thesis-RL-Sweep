# =============================================================================
# Thesis-RL-Sweep — ROCm Docker Image
# Base: AMD ROCm 7.2 on Ubuntu 22.04 (matches your local WSL2 stack)
#
# PyTorch stack: AMD's official native ROCm 7.2.1 wheels (cp311)
#   sourced from repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/
#   — torch 2.9.1, torchvision 0.24.0, torchaudio 2.9.0, triton 3.5.1
#   — Python 3.10 required (wheels are cp311 only)
#   — No HSA_OVERRIDE_GFX_VERSION needed; gfx1201 (RX 9070 XT) is natively
#     supported in these builds.
#
# The target machine needs ROCm drivers installed on the HOST (not in Docker).
# Run the container with GPU access:
#   docker compose up  (see docker-compose.yml)
# or manually:
#   docker run --device=/dev/kfd --device=/dev/dri \
#              --group-add video --group-add render \
#              -it thesis-rl bash
# =============================================================================

FROM rocm/dev-ubuntu-22.04:7.2-complete

# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
LABEL maintainer="Stavros Anagnostou"
LABEL description="MSc Thesis RL Sweep — PyTorch ROCm training environment"
LABEL rocm.version="7.2"

# --------------------------------------------------------------------------- #
# Environment — set early so all subsequent RUN steps inherit them
# --------------------------------------------------------------------------- #
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Native ROCm 7.2.1 wheels support gfx1201 (RX 9070 XT / RDNA4) directly.
    # HSA_OVERRIDE_GFX_VERSION is NOT set — it would downgrade capability.
    ROCR_VISIBLE_DEVICES=0 \
    # Enables experimental AOTriton kernels (used by torch 2.9 + ROCm 7.2)
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    # W&B: switch to offline if you hit network issues inside the container
    # WANDB_MODE=offline \
    PATH="/usr/local/bin:$PATH"

# --------------------------------------------------------------------------- #
# System packages
# --------------------------------------------------------------------------- #
RUN apt-get update && apt-get install -y --no-install-recommends \
        # Build tools (required by procgen & envpool C extensions)
        cmake \
        build-essential \
        ninja-build \
        # Qt5 — occasionally needed by procgen on some Ubuntu versions
        qtbase5-dev \
        # OpenCV headless runtime deps
        libgl1 \
        libglib2.0-0 \
        # Video / ffmpeg (for imageio[ffmpeg] in evaluate.py)
        ffmpeg \
        # Misc utilities
        git \
        curl \
        wget \
        ca-certificates \
        unzip \
        # Python 3.11 — required by AMD's cp311 ROCm 7.2.1 wheels
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python / pip
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
 && python -m pip install --upgrade pip setuptools wheel

# --------------------------------------------------------------------------- #
# PyTorch — AMD's official native ROCm 7.2.1 wheels
# Pulled directly from repo.radeon.com (same source as your working WSL2 setup)
# These are cp311 wheels, which is why Python 3.11 is required above.
# --------------------------------------------------------------------------- #
RUN ROCM_WHEEL_BASE="https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1" \
 && wget -q --show-progress \
        "${ROCM_WHEEL_BASE}/torch-2.9.1+rocm7.2.1.lw.gitff65f5bc-cp311-cp311-linux_x86_64.whl" \
        "${ROCM_WHEEL_BASE}/torchvision-0.24.0+rocm7.2.1.gitb919bd0c-cp311-cp311-linux_x86_64.whl" \
        "${ROCM_WHEEL_BASE}/torchaudio-2.9.0+rocm7.2.1.gite3c6ee2b-cp311-cp311-linux_x86_64.whl" \
        "${ROCM_WHEEL_BASE}/triton-3.5.1+rocm7.2.1.gita272dfa8-cp311-cp311-linux_x86_64.whl" \
 && pip install --no-cache-dir \
        torch-2.9.1+rocm7.2.1.lw.gitff65f5bc-cp311-cp311-linux_x86_64.whl \
        torchvision-0.24.0+rocm7.2.1.gitb919bd0c-cp311-cp311-linux_x86_64.whl \
        torchaudio-2.9.0+rocm7.2.1.gite3c6ee2b-cp311-cp311-linux_x86_64.whl \
        triton-3.5.1+rocm7.2.1.gita272dfa8-cp311-cp311-linux_x86_64.whl \
 # Clean up .whl files — no need to keep them in the image layer
 && rm -f *.whl

# --------------------------------------------------------------------------- #
# Project dependencies (everything from pyproject.toml except torch)
# Install these BEFORE copying source so Docker can cache this layer.
# --------------------------------------------------------------------------- #
RUN pip install --no-cache-dir \
        "gymnasium>=0.29.1" \
        "numpy>=1.26.4" \
        "wandb>=0.16.6" \
        "pyyaml>=6.0.1" \
        "matplotlib>=3.8.4" \
        "seaborn>=0.13.2" \
        "rliable" \
        "opencv-python-headless>=4.9.0.80" \
        "tqdm>=4.66.2" \
        "imageio[ffmpeg]>=2.34.0" \
        "minigrid>=2.3.1"

# --------------------------------------------------------------------------- #
# envpool — primary fast env backend (~30-50% faster than native procgen).
# Requires a separate install step; pip extras are needed on some distros.
# If envpool fails to build/install, fall back to procgen (see below).
# --------------------------------------------------------------------------- #
RUN pip install --no-cache-dir envpool>=1.1.0 \
    || echo "[WARNING] envpool install failed — falling back to procgen."

# --------------------------------------------------------------------------- #
# procgen — optional fallback if envpool is unavailable.
# Uncomment the line below if you need it (or if envpool failed above).
# --------------------------------------------------------------------------- #
# RUN pip install --no-cache-dir procgen==0.10.7 --no-build-isolation

# --------------------------------------------------------------------------- #
# Copy project source and install in editable mode
# --------------------------------------------------------------------------- #
WORKDIR /workspace

COPY . /workspace/

RUN pip install --no-cache-dir -e . \
    # Skip torch/torchvision/torchaudio if pip tries to re-resolve them from
    # pyproject.toml — they were installed with the ROCm index above.
    --no-deps 2>/dev/null || pip install --no-cache-dir -e .

# --------------------------------------------------------------------------- #
# Pre-flight sanity check (non-fatal — GPU may not be present at build time)
# --------------------------------------------------------------------------- #
RUN python -c "import torch; print('PyTorch version:', torch.__version__)" \
 && python -c "import gymnasium; print('Gymnasium version:', gymnasium.__version__)" \
 && echo "Build-time GPU check (may show False if /dev/kfd not mounted):" \
 && python -c "import torch; print('CUDA/ROCm available:', torch.cuda.is_available())" || true

# --------------------------------------------------------------------------- #
# Volumes — mount checkpoints and figures outside the container so they
# persist across container restarts. Define defaults here; override in compose.
# --------------------------------------------------------------------------- #
VOLUME ["/workspace/checkpoints", "/workspace/figures", "/workspace/wandb"]

# --------------------------------------------------------------------------- #
# Entrypoint / default command
# --------------------------------------------------------------------------- #
# Default: drop into an interactive shell so you can run scripts manually.
# Override at runtime, e.g.:
#   docker compose run thesis-rl bash scripts/run_sweep.sh
CMD ["bash"]
