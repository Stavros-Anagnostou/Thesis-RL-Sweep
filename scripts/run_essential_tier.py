#!/usr/bin/env python3
"""
Essential Tier — Minimum viable thesis experiments.

Runs:
  PPO IMPALA baseline    16 games × 5 seeds =  80 runs
  UCB-DrAC               16 games × 5 seeds =  80 runs
  PLR                    16 games × 5 seeds =  80 runs
  PLR + DrAC             16 games × 5 seeds =  80 runs
  DQN baseline            4 games × 5 seeds =  20 runs
  ─────────────────────────────────────────────────────
  Total                                       340 runs

Usage:
    python scripts/run_essential_tier.py                  # sequential
    python scripts/run_essential_tier.py --parallel 2     # 2 concurrent jobs
    python scripts/run_essential_tier.py --parallel 3     # 3 concurrent jobs
    python scripts/run_essential_tier.py --dry-run        # preview without running
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


TIER_MATRIX = Path(__file__).resolve().parent.parent / "configs" / "tier_essential.yaml"

DESCRIPTION = """
══════════════════════════════════════════
  ESSENTIAL TIER
  340 runs
══════════════════════════════════════════

  PPO IMPALA baseline    16 games × 5 seeds =  80 runs
  UCB-DrAC               16 games × 5 seeds =  80 runs
  PLR                    16 games × 5 seeds =  80 runs
  PLR + DrAC             16 games × 5 seeds =  80 runs
  DQN baseline            4 games × 5 seeds =  20 runs
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the essential tier experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=DESCRIPTION,
    )
    parser.add_argument(
        "--parallel", type=int, default=1,
        help="Number of concurrent training jobs (default: 1)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would run without executing anything",
    )
    parser.add_argument(
        "--skip-completed", action="store_true", default=True,
        help="Skip runs that already have a completed W&B entry (default: True)",
    )
    parser.add_argument(
        "--filter", type=str, default=None,
        help="Only run experiments whose ID contains this substring",
    )
    args = parser.parse_args()

    if not TIER_MATRIX.exists():
        print(f"Error: tier matrix not found at {TIER_MATRIX}")
        print("Make sure configs/tier_essential.yaml exists.")
        sys.exit(1)

    print(DESCRIPTION)
    print(f"  parallel = {args.parallel}")
    print(f"  matrix   = {TIER_MATRIX}")
    print()

    # Build the command to run_experiment_matrix.py
    matrix_runner = Path(__file__).resolve().parent / "run_experiment_matrix.py"
    cmd = [
        sys.executable, str(matrix_runner),
        "--matrix", str(TIER_MATRIX),
        "--parallel", str(args.parallel),
    ]

    if args.dry_run:
        cmd.append("--dry-run")
    if args.skip_completed:
        cmd.append("--skip-completed")
    if args.filter:
        cmd.extend(["--filter", args.filter])

    try:
        result = subprocess.run(cmd, check=False)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n\nInterrupted. Re-run this script to resume — completed runs will be skipped.")
        sys.exit(130)


if __name__ == "__main__":
    main()
