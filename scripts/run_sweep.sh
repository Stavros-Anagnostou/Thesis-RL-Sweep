#!/usr/bin/env bash
# Run PPO baselines across 4 representative games × 5 seeds, sequentially.
# Total: 20 runs × 25 M steps each.  Estimated wall-clock on RX 9070 XT: ~20–40 h.
#
# Usage:
#   bash scripts/run_sweep.sh
#   bash scripts/run_sweep.sh 2>/dev/null   # suppress stderr if desired

set -euo pipefail

GAMES=(starpilot bigfish coinrun ninja)
SEEDS=(1 2 3 4 5)
ENCODER="impala"
CONFIG="configs/ppo_procgen_baseline.yaml"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

TOTAL=$(( ${#GAMES[@]} * ${#SEEDS[@]} ))
CURRENT=0
FAILED=()

echo "========================================================"
echo "  PPO Procgen Sweep"
echo "  Games : ${GAMES[*]}"
echo "  Seeds : ${SEEDS[*]}"
echo "  Total : ${TOTAL} runs"
echo "========================================================"
echo ""

for GAME in "${GAMES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        CURRENT=$(( CURRENT + 1 ))
        echo "────────────────────────────────────────────────────────"
        echo "  Run ${CURRENT}/${TOTAL}  |  ${GAME}  seed=${SEED}"
        echo "────────────────────────────────────────────────────────"

        START_TS=$(date +%s)

        if python src/train_ppo.py \
               --env-id  "${GAME}" \
               --seed    "${SEED}" \
               --encoder "${ENCODER}" \
               --config  "${CONFIG}"; then

            END_TS=$(date +%s)
            ELAPSED=$(( END_TS - START_TS ))
            echo "  ✓ Completed in $(( ELAPSED / 3600 ))h $(( (ELAPSED % 3600) / 60 ))m"
        else
            echo "  ✗ FAILED: ${GAME} seed=${SEED}"
            FAILED+=("${GAME}_seed${SEED}")
        fi
        echo ""
    done
done

echo "========================================================"
echo "  Sweep complete.  ${CURRENT}/${TOTAL} runs attempted."
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "  FAILED runs:"
    for f in "${FAILED[@]}"; do
        echo "    - ${f}"
    done
else
    echo "  All runs succeeded."
fi
echo "========================================================"
