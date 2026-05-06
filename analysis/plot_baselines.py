"""
Pull training curves from W&B and generate baseline plots.

Produces per-game PDF figures showing:
  1. Train return + test return vs timesteps (mean ± std across seeds)
  2. Generalization gap over training

Usage:
    python analysis/plot_baselines.py
    python analysis/plot_baselines.py --project thesis-rl-baselines --output-dir figures/
    python analysis/plot_baselines.py --games coinrun ninja --seeds 1 2 3
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless WSL2
import matplotlib.pyplot as plt
import seaborn as sns
import wandb


# ---------------------------------------------------------------------------
# W&B data fetching
# ---------------------------------------------------------------------------

def fetch_runs(
    project: str,
    entity: str | None,
    games: list[str],
    seeds: list[int],
) -> dict[str, list[pd.DataFrame]]:
    """
    Pull run histories from W&B for each game.

    Returns a dict mapping game_id → list of DataFrames (one per seed).
    Each DataFrame has columns: step, train_return, test_return, gen_gap.
    """
    api = wandb.Api()
    project_path = f"{entity}/{project}" if entity else project

    print(f"[wandb] Fetching runs from: {project_path}")
    print(f"        Games: {games}")
    print(f"        Seeds: {seeds}")

    runs = api.runs(project_path)

    # Index runs by (game, seed)
    run_map: dict[tuple[str, int], Any] = {}
    for run in runs:
        cfg = run.config
        game = cfg.get("env_id")
        seed = cfg.get("seed")
        if game in games and seed in seeds:
            # Keep the most recent run for this (game, seed) pair.
            key = (game, seed)
            if key not in run_map or run.created_at > run_map[key].created_at:
                run_map[key] = run

    print(f"[wandb] Found {len(run_map)} matching runs.")

    result: dict[str, list[pd.DataFrame]] = {g: [] for g in games}

    for (game, seed), run in sorted(run_map.items()):
        print(f"  Pulling: {game} seed={seed}  run={run.id}  ({run.state})")
        history = run.history(
            keys=[
                "train/global_step",
                "eval/train_return",
                "eval/test_return",
                "eval/generalization_gap",
            ],
            x_axis="train/global_step",
            pandas=True,
        )

        if history.empty:
            print(f"    ⚠ Empty history for {game} seed={seed}, skipping.")
            continue

        df = history.rename(columns={
            "train/global_step":       "step",
            "eval/train_return":       "train_return",
            "eval/test_return":        "test_return",
            "eval/generalization_gap": "gen_gap",
        })
        df = df.dropna(subset=["step", "train_return", "test_return"])
        df = df.sort_values("step").reset_index(drop=True)
        result[game].append(df)

    return result


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_returns_for_game(
    ax_ret: plt.Axes,
    ax_gap: plt.Axes,
    game: str,
    dataframes: list[pd.DataFrame],
    palette: dict[str, str],
) -> None:
    """
    Plot train/test return curves and generalization gap for a single game.
    Uses aligned interpolation on a common step grid to plot mean ± std bands.
    """
    if not dataframes:
        ax_ret.set_title(f"{game}\n(no data)")
        return

    # Determine a common step grid.
    max_step = max(df["step"].max() for df in dataframes)
    step_grid = np.linspace(0, max_step, num=200)

    def interpolate_series(dfs: list[pd.DataFrame], col: str) -> np.ndarray:
        """Interpolate each run onto step_grid, return array (n_seeds, n_steps)."""
        arrays = []
        for df in dfs:
            y = np.interp(step_grid, df["step"].values, df[col].values,
                          left=np.nan, right=df[col].values[-1])
            arrays.append(y)
        return np.stack(arrays, axis=0)  # (n_seeds, n_steps)

    train_arr = interpolate_series(dataframes, "train_return")
    test_arr  = interpolate_series(dataframes, "test_return")
    gap_arr   = interpolate_series(dataframes, "gen_gap")

    def _plot_band(ax, arr, color, label):
        mean = np.nanmean(arr, axis=0)
        std  = np.nanstd(arr, axis=0)
        ax.plot(step_grid / 1e6, mean, color=color, label=label, linewidth=2)
        ax.fill_between(
            step_grid / 1e6,
            mean - std,
            mean + std,
            color=color,
            alpha=0.2,
        )

    _plot_band(ax_ret, train_arr, palette["train"], "Train levels")
    _plot_band(ax_ret, test_arr,  palette["test"],  "Test levels")
    _plot_band(ax_gap, gap_arr,   palette["gap"],   "Gen. gap")

    ax_ret.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax_gap.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    ax_ret.set_title(f"{game.capitalize()}", fontsize=14, fontweight="bold")
    ax_ret.set_ylabel("Episode Return", fontsize=11)
    ax_ret.legend(fontsize=9, loc="upper left")
    ax_ret.grid(True, alpha=0.3)

    ax_gap.set_xlabel("Environment Steps (M)", fontsize=11)
    ax_gap.set_ylabel("Gen. Gap (train − test)", fontsize=11)
    ax_gap.grid(True, alpha=0.3)

    n = len(dataframes)
    ax_ret.set_title(f"{game.capitalize()}  (n={n} seeds)", fontsize=13, fontweight="bold")


def make_plots(
    run_data: dict[str, list[pd.DataFrame]],
    games: list[str],
    output_dir: Path,
) -> None:
    """Generate one PDF per game plus a summary grid PDF."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    palette = {
        "train": sns.color_palette("tab10")[0],  # blue
        "test":  sns.color_palette("tab10")[1],  # orange
        "gap":   sns.color_palette("tab10")[2],  # green
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Per-game PDFs ---
    for game in games:
        dataframes = run_data.get(game, [])
        fig, (ax_ret, ax_gap) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        _plot_returns_for_game(ax_ret, ax_gap, game, dataframes, palette)
        fig.suptitle(f"PPO Baseline — {game.capitalize()}", fontsize=15, fontweight="bold")
        fig.tight_layout()
        out = output_dir / f"baseline_{game}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] Saved: {out}")

    # --- Summary grid (all games, 2 rows per game) ---
    n_games = len(games)
    fig, axes = plt.subplots(
        nrows=n_games,
        ncols=2,
        figsize=(14, 4 * n_games),
        sharex=False,
    )
    if n_games == 1:
        axes = np.array([axes])  # ensure 2-D indexing

    for row, game in enumerate(games):
        dataframes = run_data.get(game, [])
        _plot_returns_for_game(axes[row, 0], axes[row, 1], game, dataframes, palette)

    for ax in axes[:, 1]:
        ax.set_title("")  # remove duplicate titles from gap axes

    fig.suptitle("PPO Procgen Baselines — Easy Mode, 200 Training Levels", fontsize=14)
    fig.tight_layout()
    out = output_dir / "baselines_summary.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Summary grid saved: {out}")

    # --- Final performance bar chart ---
    fig, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    game_labels, train_means, train_stds, test_means, test_stds = [], [], [], [], []

    for game in games:
        dfs = run_data.get(game, [])
        if not dfs:
            continue
        final_trains = [df["train_return"].dropna().iloc[-1] for df in dfs if not df.empty]
        final_tests  = [df["test_return"].dropna().iloc[-1]  for df in dfs if not df.empty]
        if not final_trains:
            continue
        game_labels.append(game.capitalize())
        train_means.append(np.mean(final_trains))
        train_stds.append(np.std(final_trains))
        test_means.append(np.mean(final_tests))
        test_stds.append(np.std(final_tests))

    x = np.arange(len(game_labels))
    w = 0.35
    axes2[0].bar(x - w/2, train_means, w, yerr=train_stds, label="Train", capsize=4,
                 color=palette["train"], alpha=0.85)
    axes2[0].bar(x + w/2, test_means, w, yerr=test_stds, label="Test", capsize=4,
                 color=palette["test"], alpha=0.85)
    axes2[0].set_xticks(x)
    axes2[0].set_xticklabels(game_labels, rotation=15)
    axes2[0].set_ylabel("Mean Return (final)")
    axes2[0].set_title("Final Performance")
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.3, axis="y")

    gen_gaps = [t - v for t, v in zip(train_means, test_means)]
    gen_stds  = [np.sqrt(ts**2 + vs**2) for ts, vs in zip(train_stds, test_stds)]
    axes2[1].bar(x, gen_gaps, yerr=gen_stds, capsize=4, color=palette["gap"], alpha=0.85)
    axes2[1].set_xticks(x)
    axes2[1].set_xticklabels(game_labels, rotation=15)
    axes2[1].set_ylabel("Generalization Gap")
    axes2[1].set_title("Generalization Gap (train − test)")
    axes2[1].grid(True, alpha=0.3, axis="y")

    fig.suptitle("PPO Baseline Summary", fontsize=13)
    fig.tight_layout()
    out = output_dir / "baselines_bar_chart.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Bar chart saved: {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PPO Procgen baselines from W&B")
    parser.add_argument("--project",    type=str, default="thesis-rl-baselines")
    parser.add_argument("--entity",     type=str, default=None, help="W&B entity (username/team)")
    parser.add_argument("--games",      nargs="+", default=["starpilot", "bigfish", "coinrun", "ninja"])
    parser.add_argument("--seeds",      nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--output-dir", type=str, default="figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_data = fetch_runs(
        project=args.project,
        entity=args.entity,
        games=args.games,
        seeds=args.seeds,
    )
    make_plots(run_data, args.games, Path(args.output_dir))
    print("\n[plot_baselines] All figures saved.")


if __name__ == "__main__":
    main()
