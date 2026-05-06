#!/usr/bin/env bash
# Quick smoke test: verifies the entire pipeline works end-to-end in ~2-5 minutes.
#
# Runs PPO for 200 000 steps on coinrun (no W&B logging), then evaluates
# the resulting checkpoint.  Any crash here means something is broken before
# you burn hours on a full run.
#
# Usage:
#   bash scripts/quick_test.sh
#   bash scripts/quick_test.sh bigfish    # test with a different game

set -euo pipefail

ENV_ID=${1:-coinrun}
SEED=42
STEPS=200000      # small enough to finish in a few minutes
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

echo "========================================================"
echo "  Quick Pipeline Smoke Test"
echo "  env_id : ${ENV_ID}"
echo "  steps  : ${STEPS}"
echo "  device : auto"
echo "========================================================"

# ---- Step 1: Pre-flight ----
echo ""
echo "[1/3] Running pre-flight checks..."
python scripts/validate_setup.py

# ---- Step 2: Short training run ----
echo ""
echo "[2/3] Training for ${STEPS} steps (no W&B)..."
python src/train_ppo.py \
    --env-id "${ENV_ID}" \
    --seed "${SEED}" \
    --encoder impala \
    --config configs/ppo_procgen_baseline.yaml \
    --total-timesteps "${STEPS}" \
    --no-wandb

# ---- Step 3: Evaluate the latest checkpoint ----
echo ""
echo "[3/3] Evaluating latest checkpoint..."

CKPT_DIR="checkpoints/${ENV_ID}"
# Find the most recently created latest.pt
CKPT=$(find "${CKPT_DIR}" -name "latest.pt" -printf "%T@ %p\n" 2>/dev/null \
       | sort -n | tail -1 | awk '{print $2}')

if [ -z "${CKPT}" ]; then
    echo "ERROR: No checkpoint found under ${CKPT_DIR}"
    exit 1
fi

echo "  Checkpoint: ${CKPT}"
python src/evaluate.py \
    --checkpoint "${CKPT}" \
    --num-episodes 20 \
    --num-eval-envs 8

echo ""
echo "========================================================"
echo "  Smoke test PASSED.  Pipeline is functional."
echo "  Returns after ${STEPS} steps will be well below baseline —"
echo "  that is expected.  Run scripts/run_baseline.sh for a full run."
echo "========================================================"
