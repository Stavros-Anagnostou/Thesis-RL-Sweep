#!/usr/bin/env bash
# Encoder ablation sweep: IMPALA vs Nature vs Small CNN
# on 4 representative games × 3 seeds.
#
# Usage: bash scripts/run_encoder_ablation.sh
#        bash scripts/run_encoder_ablation.sh --dry-run

set -euo pipefail

GAMES=(starpilot coinrun ninja bigfish)
SEEDS=(1 2 3)
ENCODERS=(impala nature small)
CONFIG="configs/ppo_procgen_baseline.yaml"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[dry-run] No commands will be executed."
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

TOTAL=$(( ${#GAMES[@]} * ${#SEEDS[@]} * ${#ENCODERS[@]} ))
CURRENT=0

echo "========================================================"
echo "  Encoder Ablation Sweep"
echo "  Encoders : ${ENCODERS[*]}"
echo "  Games    : ${GAMES[*]}"
echo "  Seeds    : ${SEEDS[*]}"
echo "  Total    : ${TOTAL} runs"
echo "========================================================"
echo ""

for ENCODER in "${ENCODERS[@]}"; do
    for GAME in "${GAMES[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            CURRENT=$(( CURRENT + 1 ))

            echo "  [${CURRENT}/${TOTAL}]  encoder=${ENCODER} | ${GAME} | seed=${SEED}"

            CMD="python src/train_ppo.py \
                --config ${CONFIG} \
                --env-id ${GAME} \
                --seed ${SEED} \
                --encoder ${ENCODER} \
                --wandb-tags encoder_ablation,enc_${ENCODER}"

            if $DRY_RUN; then
                echo "    $CMD"
            else
                eval "$CMD"
                echo "  ✓ Done"
            fi
        done
    done
done

echo ""
echo "========================================================"
echo "  Encoder ablation complete."
echo "========================================================"
