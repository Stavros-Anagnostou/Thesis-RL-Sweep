#!/usr/bin/env python3
"""
Experiment matrix runner.

Reads configs/experiment_matrix.yaml, expands all experiments into individual
runs (game × seed × override combinations), then dispatches them sequentially
or in parallel.

Features:
  - Skip runs that already have a completed W&B entry (safe to re-run)
  - Estimate remaining time based on completed run durations
  - --parallel N for running N processes concurrently
  - --dry-run to print the full run list without executing anything
  - --filter EXPR to run only experiments whose ID matches a substring

Usage:
    python scripts/run_experiment_matrix.py
    python scripts/run_experiment_matrix.py --dry-run
    python scripts/run_experiment_matrix.py --filter drac --parallel 2
    python scripts/run_experiment_matrix.py --filter num_levels_sweep
    python scripts/run_experiment_matrix.py --experiments ppo_drac ppo_rad
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Run descriptor
# ---------------------------------------------------------------------------

class Run:
    """Represents a single trainable unit (one game, one seed, one config)."""

    def __init__(
        self,
        experiment_id: str,
        config_file: str,
        env_id: str,
        seed: int,
        overrides: dict[str, Any],
        script: str = "src/train_ppo.py",
    ) -> None:
        self.experiment_id = experiment_id
        self.config_file   = config_file
        self.env_id        = env_id
        self.seed          = seed
        self.overrides     = overrides
        self.script        = script

    @property
    def run_id(self) -> str:
        """Human-readable unique identifier for this run."""
        override_str = "_".join(f"{k}={v}" for k, v in sorted(self.overrides.items()))
        parts = [self.experiment_id, self.env_id, f"seed{self.seed}"]
        if override_str:
            parts.append(override_str)
        return "__".join(parts)

    def to_cmd(self) -> list[str]:
        """Build the command-line invocation."""
        cmd = [
            sys.executable, self.script,
            "--config", f"configs/{self.config_file}",
            "--env-id", self.env_id,
            "--seed", str(self.seed),
        ]
        for key, value in self.overrides.items():
            # Convert snake_case → --kebab-case
            cli_key = f"--{key.replace('_', '-')}"
            cmd += [cli_key, str(value)]
        return cmd

    def __repr__(self) -> str:
        return f"Run({self.run_id})"


# ---------------------------------------------------------------------------
# Experiment matrix expansion
# ---------------------------------------------------------------------------

def load_matrix(matrix_path: Path) -> dict[str, Any]:
    """Load and parse experiment_matrix.yaml."""
    with open(matrix_path) as f:
        return yaml.safe_load(f)


def expand_experiment(exp_id: str, spec: dict[str, Any]) -> list[Run]:
    """
    Expand one experiment spec into individual Run objects.

    Handles:
      - games × seeds cross-product
      - overrides: single dict applied to all runs
      - overrides_sweep: dict of lists — all combinations are run
    """
    config_file = spec["config"]
    games       = spec["games"]
    seeds       = spec["seeds"]
    overrides   = spec.get("overrides", {})
    sweep       = spec.get("overrides_sweep", {})

    # Determine the training script.
    script = "src/train_dqn.py" if "dqn" in config_file else "src/train_ppo.py"

    # Expand overrides_sweep into all combinations.
    if sweep:
        sweep_keys  = list(sweep.keys())
        sweep_vals  = list(sweep.values())
        sweep_combos = [dict(zip(sweep_keys, combo)) for combo in itertools.product(*sweep_vals)]
    else:
        sweep_combos = [{}]

    runs = []
    for game in games:
        for seed in seeds:
            for combo in sweep_combos:
                merged_overrides = {**overrides, **combo}
                runs.append(Run(
                    experiment_id=exp_id,
                    config_file=config_file,
                    env_id=game,
                    seed=seed,
                    overrides=merged_overrides,
                    script=script,
                ))
    return runs


def build_run_list(
    matrix: dict[str, Any],
    filter_str: str | None = None,
    experiment_ids: list[str] | None = None,
) -> list[Run]:
    """Build the full flat list of Run objects from the matrix."""
    experiments = matrix.get("experiments", {})
    all_runs: list[Run] = []

    for exp_id, spec in experiments.items():
        # Apply filters
        if experiment_ids and exp_id not in experiment_ids:
            continue
        if filter_str and filter_str.lower() not in exp_id.lower():
            continue
        all_runs.extend(expand_experiment(exp_id, spec))

    return all_runs


# ---------------------------------------------------------------------------
# W&B completed-run check
# ---------------------------------------------------------------------------

def get_completed_run_ids(project: str, entity: str | None = None) -> set[str]:
    """
    Query W&B for all completed runs in the project.
    Returns a set of run names that have state='finished'.

    Returns empty set if wandb is not available or not logged in.
    """
    try:
        import wandb
        api = wandb.Api()
        path = f"{entity}/{project}" if entity else project
        runs = api.runs(path, filters={"state": "finished"})
        return {r.name for r in runs}
    except Exception as e:
        print(f"[wandb] Could not query completed runs: {e}")
        return set()


# ---------------------------------------------------------------------------
# Sequential runner
# ---------------------------------------------------------------------------

def run_sequential(runs: list[Run], skip_completed: set[str], dry_run: bool) -> None:
    """Execute runs one at a time, printing progress and ETA."""
    total = len(runs)
    durations: list[float] = []

    for i, run in enumerate(runs, 1):
        # Check if already completed
        if run.run_id in skip_completed:
            print(f"  [{i:>4}/{total}] SKIP (completed)  {run.run_id}")
            continue

        elapsed_total = sum(durations)
        eta_str = ""
        if durations:
            avg_dur = elapsed_total / len(durations)
            remaining = total - i
            eta_sec  = avg_dur * remaining
            eta_str  = f"  ETA ~{_format_duration(eta_sec)}"

        progress_pct = i / total * 100
        print(
            f"\n  [{i:>4}/{total}]  {progress_pct:5.1f}%  │  "
            f"{run.experiment_id:<20}  {run.env_id:<12}  seed={run.seed}{eta_str}"
        )
        if run.overrides:
            print(f"  overrides: {run.overrides}")

        if dry_run:
            print(f"  CMD: {' '.join(run.to_cmd())}")
            continue

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / f"{run.run_id}.log"

        t_start = time.time()
        try:
            with open(log_path, "w") as log_file:
                result = subprocess.run(
                    run.to_cmd(), check=True,
                    stdout=log_file, stderr=subprocess.STDOUT,
                )
            duration = time.time() - t_start
            durations.append(duration)
            print(f"  ✓ Done in {_format_duration(duration)}")
        except subprocess.CalledProcessError as e:
            duration = time.time() - t_start
            print(f"  ✗ FAILED (exit code {e.returncode}) after {_format_duration(duration)}")
            print(f"    └─ Log: {log_path}")
            # Continue to next run rather than aborting the sweep.

    print(f"\n{'═' * 70}")
    print(f"  COMPLETE  │  {len(durations)}/{total} succeeded")
    if durations:
        print(f"  Total: {_format_duration(sum(durations))}  │  Mean per run: {_format_duration(sum(durations) / len(durations))}")
    print(f"  Logs: {LOG_DIR.resolve()}/")
    print(f"{'═' * 70}\n")


# ---------------------------------------------------------------------------
# Parallel runner
# ---------------------------------------------------------------------------

LOG_DIR = Path("logs")


def _run_one(run: Run) -> tuple[Run, bool, float]:
    """Execute a single training run via subprocess, logging output to file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{run.run_id}.log"

    t = time.time()
    try:
        with open(log_path, "w") as log_file:
            subprocess.run(
                run.to_cmd(),
                check=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        return run, True, time.time() - t
    except subprocess.CalledProcessError:
        return run, False, time.time() - t


def _format_duration(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h{m:02d}m"


def _estimate_remaining(durations: list[float], remaining: int, n_parallel: int) -> str:
    """Estimate wall-clock time for remaining runs."""
    if not durations:
        return "estimating..."
    avg = sum(durations) / len(durations)
    # With N workers, remaining runs take ceil(remaining/N) batches
    batches = -(-remaining // n_parallel)  # ceiling division
    eta_seconds = avg * batches
    return _format_duration(eta_seconds)


def run_parallel(runs: list[Run], skip_completed: set[str], n_parallel: int) -> None:
    """Run up to n_parallel experiments concurrently using subprocess."""
    import concurrent.futures

    pending = [r for r in runs if r.run_id not in skip_completed]
    total = len(pending)
    completed = 0
    failed = 0
    durations: list[float] = []

    skipped = len(runs) - total
    if skipped > 0:
        print(f"  Skipping {skipped} already-completed runs")

    print(f"\n{'═' * 70}")
    print(f"  RUNNING {total} EXPERIMENTS  |  {n_parallel} parallel workers")
    print(f"  Logs: {LOG_DIR.resolve()}/")
    print(f"{'═' * 70}\n")

    # ThreadPoolExecutor is sufficient here — the actual training runs in
    # separate processes via subprocess.run(). Using threads avoids the
    # pickling issues that ProcessPoolExecutor has with complex objects.
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_parallel) as executor:
        futures = {executor.submit(_run_one, r): r for r in pending}
        for future in concurrent.futures.as_completed(futures):
            run, ok, dur = future.result()
            completed += 1
            durations.append(dur)
            remaining = total - completed

            status = "✓" if ok else "✗ FAIL"
            if not ok:
                failed += 1

            eta = _estimate_remaining(durations, remaining, n_parallel)
            progress_pct = completed / total * 100

            print(
                f"  {status}  [{completed:>4}/{total}]  {progress_pct:5.1f}%  "
                f"│  {_format_duration(dur):>6}  │  {run.experiment_id:<20}  "
                f"{run.env_id:<12}  seed={run.seed}"
            )

            if remaining > 0 and completed % n_parallel == 0:
                print(f"  {'─' * 66}")
                print(f"  ETA: ~{eta}  |  {remaining} runs remaining")
                print(f"  {'─' * 66}")

            # If a run failed, print where to find the log
            if not ok:
                log_path = LOG_DIR / f"{run.run_id}.log"
                print(f"         └─ Log: {log_path}")

    print(f"\n{'═' * 70}")
    print(f"  COMPLETE  │  {completed - failed}/{total} succeeded  │  {failed} failed")
    total_time = sum(durations)
    wall_time = max(durations) if durations else 0  # approximate
    print(f"  Total compute: {_format_duration(total_time)}  │  Wall clock: ~{_format_duration(sum(durations) / max(n_parallel, 1))}")
    print(f"{'═' * 70}\n")

    if failed > 0:
        print(f"  Failed runs (check logs in {LOG_DIR}/):")
        # Re-scan to list failures
        for future in futures:
            run = futures[future]
            r, ok, _ = future.result()
            if not ok:
                print(f"    ✗ {run.run_id}")
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full experiment matrix")
    parser.add_argument(
        "--matrix",
        type=str,
        default="configs/experiment_matrix.yaml",
        help="Path to experiment_matrix.yaml",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Only run experiments whose ID contains this substring",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Explicit list of experiment IDs to run (overrides --filter)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel processes (default: 1 = sequential)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print run commands without executing them",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        default=True,
        help="Skip runs with a matching completed W&B entry (default: True)",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Disable skipping of completed runs",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="thesis-rl-baselines",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    matrix_path = Path(args.matrix)
    if not matrix_path.exists():
        print(f"ERROR: matrix file not found: {matrix_path}")
        sys.exit(1)

    matrix = load_matrix(matrix_path)
    runs = build_run_list(matrix, filter_str=args.filter, experiment_ids=args.experiments)

    if not runs:
        print("No runs matched the given filters.  Check --filter or --experiments.")
        sys.exit(0)

    print(f"[matrix] Loaded {len(runs)} runs from {matrix_path}")

    # Determine completed runs to skip
    skip_completed: set[str] = set()
    if args.skip_completed and not args.no_skip and not args.dry_run:
        print("[wandb] Checking for completed runs to skip...")
        skip_completed = get_completed_run_ids(args.wandb_project, args.wandb_entity)
        print(f"[wandb] Found {len(skip_completed)} completed runs.")

    # Print summary
    pending = [r for r in runs if r.run_id not in skip_completed]
    print(f"  Total:    {len(runs)}")
    print(f"  Skip:     {len(runs) - len(pending)}")
    print(f"  To run:   {len(pending)}")

    if args.dry_run:
        print("\n[dry-run] Commands that would be executed:\n")
        for i, run in enumerate(pending, 1):
            print(f"  [{i:>4}] {' '.join(run.to_cmd())}")
        return

    if args.parallel > 1:
        run_parallel(runs, skip_completed, args.parallel)
    else:
        run_sequential(runs, skip_completed, dry_run=False)


if __name__ == "__main__":
    main()
