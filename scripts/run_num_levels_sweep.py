#!/usr/bin/env python3
"""
Convenience script to run the num_levels ablation sweep.

Trains PPO (baseline) with varying numbers of training levels on the 4
representative games, to produce the "training distribution size vs
generalization" curve for the thesis.

Usage:
    python scripts/run_num_levels_sweep.py
    python scripts/run_num_levels_sweep.py --games coinrun ninja --seeds 1 2
    python scripts/run_num_levels_sweep.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

NUM_LEVELS = [10, 50, 100, 200, 500, 1000]
DEFAULT_GAMES = ["starpilot", "coinrun", "ninja", "bigfish"]
DEFAULT_SEEDS = [1, 2, 3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="num_levels sweep")
    parser.add_argument("--games",       nargs="+", default=DEFAULT_GAMES)
    parser.add_argument("--seeds",       nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--num-levels",  nargs="+", type=int, default=NUM_LEVELS)
    parser.add_argument("--config",      type=str,  default="configs/ppo_procgen_baseline.yaml")
    parser.add_argument("--dry-run",     action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    runs = [
        (game, seed, nl)
        for game in args.games
        for seed in args.seeds
        for nl in args.num_levels
    ]
    total = len(runs)
    print(f"[num_levels_sweep] {total} runs  "
          f"({len(args.games)} games × {len(args.seeds)} seeds × {len(args.num_levels)} levels)")
    print(f"  num_levels: {args.num_levels}\n")

    durations: list[float] = []
    for i, (game, seed, nl) in enumerate(runs, 1):
        eta = ""
        if durations:
            avg = sum(durations) / len(durations)
            eta = f"  ETA ~{avg * (total - i) / 3600:.1f}h"

        cmd = [
            sys.executable, "src/train_ppo.py",
            "--config", args.config,
            "--env-id", game,
            "--seed", str(seed),
            "--num-levels", str(nl),
            "--wandb-tags", f"num_levels_sweep,nl_{nl}",
        ]

        print(f"  [{i:>3}/{total}]  {game} | seed={seed} | num_levels={nl}{eta}")

        if args.dry_run:
            print(f"    CMD: {' '.join(cmd)}")
            continue

        t = time.time()
        try:
            subprocess.run(cmd, check=True)
            dur = time.time() - t
            durations.append(dur)
            print(f"  ✓ {dur / 60:.1f} min\n")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ FAILED exit={e.returncode}\n")

    print(f"\n[num_levels_sweep] Complete.  {len(durations)}/{total} succeeded.")


if __name__ == "__main__":
    main()
