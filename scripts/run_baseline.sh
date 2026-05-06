#!/usr/bin/env bash
# Run a single PPO baseline: game × seed.
#
# Usage:
#   bash scripts/run_baseline.sh starpilot 1
#   bash scripts/run_baseline.sh coinrun 3 nature   # optional: encoder override

set -euo pipefail

ENV_ID=${1:?"Usage: $0 <env_id> <seed> [encoder]"}
SEED=${2:?"Usage: $0 <env_id> <seed> [encoder]"}
ENCODER=${3:-impala}

echo "========================================"
echo "  PPO Procgen Baseline"
echo "  env_id  : ${ENV_ID}"
echo "  seed    : ${SEED}"
echo "  encoder : ${ENCODER}"
echo "========================================"

# Ensure we're in the project root regardless of where the script is called from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

python src/train_ppo.py \
    --env-id   "${ENV_ID}" \
    --seed     "${SEED}" \
    --encoder  "${ENCODER}" \
    --config   configs/ppo_procgen_baseline.yaml

echo ""
echo "[run_baseline.sh] Done: ${ENV_ID} seed=${SEED}"
