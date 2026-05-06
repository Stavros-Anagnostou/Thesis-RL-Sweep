"""
Statistical analysis using rliable.

Pulls experimental results from W&B, normalises scores, and computes:
  - IQM (Interquartile Mean) with 95% stratified bootstrap CIs
  - Performance profiles (score distribution CDFs)
  - Probability of improvement between method pairs

Outputs:
  - analysis/results.json  — all computed statistics
  - figures/rliable_iqm.pdf
  - figures/rliable_profiles.pdf
  - figures/rliable_prob_improvement.pdf

Usage:
    python analysis/rliable_analysis.py
    python analysis/rliable_analysis.py --project thesis-rl-baselines --output-dir figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

try:
    from rliable import library as rly
    from rliable import metrics as rl_metrics
    from rliable import plot_utils as rl_plot
    RLIABLE_AVAILABLE = True
except ImportError:
    RLIABLE_AVAILABLE = False
    print("[rliable] WARNING: rliable not installed — IQM/CI computations will be skipped.")
    print("  Install: pip install rliable")


# ---------------------------------------------------------------------------
# W&B data fetching
# ---------------------------------------------------------------------------

METHODS = [
    "none",         # PPO baseline (augmentation_method=none)
    "rad",
    "drac",
    "ucb_drac",
    "plr",
    "plr_drac",
    "dqn",          # pulled from dqn runs (tag="dqn")
]

METHOD_DISPLAY_NAMES = {
    "none":     "PPO (Baseline)",
    "rad":      "PPO + RAD",
    "drac":     "PPO + DrAC",
    "ucb_drac": "PPO + UCB-DrAC",
    "plr":      "PPO + PLR",
    "plr_drac": "PPO + PLR + DrAC",
    "dqn":      "DQN (Baseline)",
}

ALL_GAMES = [
    "starpilot", "bigfish", "coinrun", "ninja",
    "miner", "dodgeball", "heist", "climber",
    "maze", "jumper", "leaper", "chaser",
    "fruitbot", "plunder", "bossfight", "caveflyer",
]

# Random policy baselines (approximate, from published Procgen papers)
# Used for normalisation: normalised_score = (score - random) / (max - random)
RANDOM_BASELINES: dict[str, float] = {
    "starpilot": 2.0, "bigfish": 1.0, "coinrun": 5.0, "ninja": 3.0,
    "miner": 1.5, "dodgeball": 1.0, "heist": 2.0, "climber": 1.0,
    "maze": 4.0, "jumper": 1.0, "leaper": 1.5, "chaser": 0.5,
    "fruitbot": -1.5, "plunder": 3.0, "bossfight": 0.5, "caveflyer": 2.5,
}


def fetch_final_returns(
    project: str,
    entity: str | None,
    methods: list[str],
    games: list[str],
) -> dict[str, dict[str, list[float]]]:
    """
    Pull final eval/test_return for each (method, game) combination.

    Returns: {method: {game: [seed1_return, seed2_return, ...]}}
    """
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    print(f"[wandb] Fetching runs from {path}...")

    runs = api.runs(path, filters={"state": "finished"})

    # Build index: (method_tag, game, seed) → final test return
    data: dict[str, dict[str, list[float]]] = {m: {g: [] for g in games} for m in methods}

    for run in runs:
        tags = run.tags or []
        cfg  = run.config or {}

        game = cfg.get("env_id")
        if game not in games:
            continue

        # Determine method from tags
        method = None
        for tag in tags:
            if tag in methods:
                method = tag
                break
        if method is None:
            continue

        # Get final eval/test_return (last logged value)
        history = run.history(keys=["eval/test_return"], pandas=True)
        if history.empty or "eval/test_return" not in history.columns:
            continue
        final_return = history["eval/test_return"].dropna().iloc[-1] if len(history) > 0 else None
        if final_return is None:
            continue

        data[method][game].append(float(final_return))

    return data


def normalise_scores(
    data: dict[str, dict[str, list[float]]],
    games: list[str],
) -> dict[str, dict[str, list[float]]]:
    """
    Normalise scores to [0, 1] per game.

    normalised = (score - random) / (max_across_methods - random)
    """
    # Find max return per game across all methods
    max_per_game: dict[str, float] = {}
    for game in games:
        all_scores = []
        for method_data in data.values():
            all_scores.extend(method_data.get(game, []))
        max_per_game[game] = max(all_scores) if all_scores else 1.0

    normalised: dict[str, dict[str, list[float]]] = {}
    for method, game_data in data.items():
        normalised[method] = {}
        for game in games:
            scores = game_data.get(game, [])
            rand_base = RANDOM_BASELINES.get(game, 0.0)
            denom = max_per_game[game] - rand_base
            if denom < 1e-8:
                denom = 1.0
            normalised[method][game] = [
                (s - rand_base) / denom for s in scores
            ]
    return normalised


# ---------------------------------------------------------------------------
# rliable computation
# ---------------------------------------------------------------------------

def compute_rliable_stats(
    normalised: dict[str, dict[str, list[float]]],
    games: list[str],
    n_bootstrap: int = 50_000,
) -> dict[str, Any]:
    """
    Compute IQM + 95% CI for each method using rliable.

    Returns a dict of statistics suitable for JSON serialisation.
    """
    if not RLIABLE_AVAILABLE:
        return {"error": "rliable not installed"}

    # rliable expects scores as numpy arrays of shape (n_runs, n_games)
    stats: dict[str, Any] = {}

    for method, game_data in normalised.items():
        # Build score matrix: (n_seeds, n_games) — pad with NaN if seeds differ
        max_seeds = max((len(game_data.get(g, [])) for g in games), default=0)
        if max_seeds == 0:
            continue
        score_matrix = np.full((max_seeds, len(games)), np.nan)
        for j, game in enumerate(games):
            scores = game_data.get(game, [])
            for i, s in enumerate(scores):
                score_matrix[i, j] = s

        # Drop rows (seeds) that are all NaN
        valid_rows = ~np.all(np.isnan(score_matrix), axis=1)
        score_matrix = score_matrix[valid_rows]

        if score_matrix.shape[0] == 0:
            continue

        # Replace NaN with 0 for rliable (missing game = 0 normalised score)
        score_matrix = np.nan_to_num(score_matrix, nan=0.0)

        # IQM with bootstrap CI
        iqm, iqm_ci = rly.get_interval_estimates(
            {"method": score_matrix},
            rl_metrics.aggregate_iqm,
            reps=n_bootstrap,
        )
        stats[method] = {
            "iqm":     float(iqm["method"]),
            "iqm_ci":  [float(iqm_ci["method"][0]), float(iqm_ci["method"][1])],
            "n_seeds": int(score_matrix.shape[0]),
            "n_games": int(score_matrix.shape[1]),
        }

    return stats


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_iqm_bar(
    stats: dict[str, Any],
    output_dir: Path,
    display_names: dict[str, str],
) -> None:
    """Horizontal bar chart of IQM ± CI for each method."""
    sns.set_theme(style="whitegrid", font_scale=1.2)
    methods = [m for m in METHODS if m in stats and "iqm" in stats[m]]
    if not methods:
        print("[plot_iqm_bar] No data to plot.")
        return

    labels  = [display_names.get(m, m) for m in methods]
    iqms    = [stats[m]["iqm"] for m in methods]
    ci_low  = [stats[m]["iqm"] - stats[m]["iqm_ci"][0] for m in methods]
    ci_high = [stats[m]["iqm_ci"][1] - stats[m]["iqm"] for m in methods]

    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(methods))
    colors = sns.color_palette("tab10", len(methods))

    ax.barh(y, iqms, xerr=[ci_low, ci_high], color=colors, capsize=4, alpha=0.85, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("IQM Normalised Score (95% CI)", fontsize=12)
    ax.set_title("Aggregate Generalisation Performance (IQM across 16 games)", fontsize=13)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    out = output_dir / "rliable_iqm.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")


def plot_performance_profiles(
    normalised: dict[str, dict[str, list[float]]],
    games: list[str],
    output_dir: Path,
    display_names: dict[str, str],
) -> None:
    """
    Performance profiles: for each method, fraction of (game, seed) pairs
    that achieve normalised score ≥ τ, plotted as a function of τ.
    """
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(8, 5))

    tau_range = np.linspace(0, 1.5, 200)
    palette = sns.color_palette("tab10", len(METHODS))

    for color, method in zip(palette, METHODS):
        if method not in normalised:
            continue
        all_scores = []
        for game in games:
            all_scores.extend(normalised[method].get(game, []))
        if not all_scores:
            continue
        scores = np.array(all_scores)
        fractions = np.array([(scores >= tau).mean() for tau in tau_range])
        ax.plot(tau_range, fractions, label=display_names.get(method, method),
                color=color, linewidth=2)

    ax.set_xlabel("Normalised Score Threshold τ", fontsize=12)
    ax.set_ylabel("Fraction of Runs with Score ≥ τ", fontsize=12)
    ax.set_title("Performance Profiles", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.5)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    out = output_dir / "rliable_profiles.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")


def plot_prob_improvement(
    normalised: dict[str, dict[str, list[float]]],
    games: list[str],
    output_dir: Path,
    display_names: dict[str, str],
    baseline_method: str = "none",
) -> None:
    """
    P(method > baseline) across games, shown as a bar chart.
    """
    sns.set_theme(style="whitegrid", font_scale=1.1)
    comparison_methods = [m for m in METHODS if m != baseline_method and m in normalised]
    if not comparison_methods:
        return

    probs = []
    labels = []
    for method in comparison_methods:
        method_scores   = []
        baseline_scores = []
        for game in games:
            ms = normalised.get(method,   {}).get(game, [])
            bs = normalised.get(baseline_method, {}).get(game, [])
            method_scores.extend(ms)
            baseline_scores.extend(bs)

        if not method_scores or not baseline_scores:
            continue

        # Bootstrap estimate of P(method > baseline)
        n = min(len(method_scores), len(baseline_scores))
        m_arr = np.array(method_scores[:n])
        b_arr = np.array(baseline_scores[:n])
        p = float((m_arr > b_arr).mean())
        probs.append(p)
        labels.append(display_names.get(method, method))

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = sns.color_palette("tab10", len(probs))
    x = np.arange(len(labels))
    ax.bar(x, probs, color=colors, alpha=0.85, width=0.6)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="P=0.5 (no improvement)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(f"P(method > {display_names.get(baseline_method, baseline_method)})", fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title("Probability of Improvement over PPO Baseline", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = output_dir / "rliable_prob_improvement.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="rliable statistical analysis")
    parser.add_argument("--project",     type=str, default="thesis-rl-baselines")
    parser.add_argument("--entity",      type=str, default=None)
    parser.add_argument("--games",       nargs="+", default=ALL_GAMES)
    parser.add_argument("--methods",     nargs="+", default=METHODS)
    parser.add_argument("--output-dir",  type=str, default="figures")
    parser.add_argument("--n-bootstrap", type=int, default=50_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_data   = fetch_final_returns(args.project, args.entity, args.methods, args.games)
    normalised = normalise_scores(raw_data, args.games)

    stats = compute_rliable_stats(normalised, args.games, n_bootstrap=args.n_bootstrap)

    # Save JSON results
    results_path = Path("analysis") / "results.json"
    with open(results_path, "w") as f:
        json.dump({"iqm_stats": stats, "raw_counts": {
            m: {g: len(v) for g, v in gd.items()} for m, gd in raw_data.items()
        }}, f, indent=2)
    print(f"[results] Saved to {results_path}")

    # Print IQM table
    print(f"\n{'='*55}")
    print(f"  {'Method':<20}  {'IQM':>8}  {'95% CI':>18}  {'N seeds':>8}")
    print(f"{'─'*55}")
    for method in args.methods:
        if method in stats and "iqm" in stats[method]:
            s = stats[method]
            print(f"  {METHOD_DISPLAY_NAMES.get(method, method):<20}  "
                  f"{s['iqm']:>8.3f}  "
                  f"[{s['iqm_ci'][0]:.3f}, {s['iqm_ci'][1]:.3f}]  "
                  f"{s['n_seeds']:>8}")
    print(f"{'='*55}\n")

    plot_iqm_bar(stats, output_dir, METHOD_DISPLAY_NAMES)
    plot_performance_profiles(normalised, args.games, output_dir, METHOD_DISPLAY_NAMES)
    plot_prob_improvement(normalised, args.games, output_dir, METHOD_DISPLAY_NAMES)
    print("\n[rliable_analysis] Done.")


if __name__ == "__main__":
    main()
