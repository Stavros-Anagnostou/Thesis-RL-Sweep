"""
Generalization plots:
  1. Overfitting curves (train vs test return over training) per game per method
  2. Generalization gap over time
  3. Final generalization gap bar chart, grouped by method
  4. Heatmap: methods × games, cell = final test return
  5. num_levels sweep: test return vs num_levels (log x-axis)

Usage:
    python analysis/plot_generalization.py
    python analysis/plot_generalization.py --methods none drac plr --output-dir figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import wandb

from analysis.rliable_analysis import (
    METHODS,
    METHOD_DISPLAY_NAMES,
    ALL_GAMES,
)


# ---------------------------------------------------------------------------
# W&B data helpers
# ---------------------------------------------------------------------------

def fetch_training_curves(
    project: str,
    entity: str | None,
    methods: list[str],
    games: list[str],
    extra_tags: list[str] | None = None,
) -> dict[str, dict[str, list[pd.DataFrame]]]:
    """
    Pull train/test return curves per (method, game).
    Returns: {method: {game: [df_seed1, df_seed2, ...]}}
    Each DataFrame has columns: step, train_return, test_return, gen_gap.
    """
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    all_runs = api.runs(path, filters={"state": "finished"})

    result: dict[str, dict[str, list[pd.DataFrame]]] = {
        m: {g: [] for g in games} for m in methods
    }

    for run in all_runs:
        tags = run.tags or []
        cfg  = run.config or {}
        game = cfg.get("env_id")
        if game not in games:
            continue

        # Match method
        method = None
        for tag in tags:
            if tag in methods:
                method = tag
                break
        if method is None:
            continue

        if extra_tags and not all(t in tags for t in extra_tags):
            continue

        history = run.history(
            keys=["train/global_step", "eval/train_return", "eval/test_return", "eval/generalization_gap"],
            x_axis="train/global_step",
            pandas=True,
        )
        if history.empty:
            continue
        df = history.rename(columns={
            "train/global_step":       "step",
            "eval/train_return":       "train_return",
            "eval/test_return":        "test_return",
            "eval/generalization_gap": "gen_gap",
        }).dropna(subset=["step", "train_return", "test_return"]).sort_values("step")

        result[method][game].append(df)

    return result


def _interpolate_seeds(
    dfs: list[pd.DataFrame],
    col: str,
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate seeds onto a common step grid. Returns (grid, mean, std)."""
    if not dfs:
        return np.array([]), np.array([]), np.array([])
    max_step = max(df["step"].max() for df in dfs)
    grid = np.linspace(0, max_step, n_points)
    arrays = []
    for df in dfs:
        y = np.interp(grid, df["step"].values, df[col].values,
                      left=np.nan, right=df[col].values[-1])
        arrays.append(y)
    mat = np.stack(arrays, axis=0)
    mean = np.nanmean(mat, axis=0)
    std  = np.nanstd(mat, axis=0)
    return grid, mean, std


# ---------------------------------------------------------------------------
# Plot 1 & 2: Overfitting curves per game
# ---------------------------------------------------------------------------

def plot_overfitting_curves(
    curves: dict[str, dict[str, list[pd.DataFrame]]],
    games: list[str],
    methods: list[str],
    output_dir: Path,
) -> None:
    """One figure per game: train + test return, all methods, shaded std."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    palette = dict(zip(methods, sns.color_palette("tab10", len(methods))))

    for game in games:
        has_data = any(curves[m][game] for m in methods if m in curves)
        if not has_data:
            continue

        fig, (ax_ret, ax_gap) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

        for method in methods:
            dfs = curves.get(method, {}).get(game, [])
            if not dfs:
                continue
            color = palette[method]
            label = METHOD_DISPLAY_NAMES.get(method, method)

            grid, train_mean, train_std = _interpolate_seeds(dfs, "train_return")
            _,    test_mean,  test_std  = _interpolate_seeds(dfs, "test_return")
            _,    gap_mean,   gap_std   = _interpolate_seeds(dfs, "gen_gap")

            x = grid / 1e6
            ax_ret.plot(x, train_mean, color=color, linewidth=2, linestyle="--",
                        label=f"{label} (train)")
            ax_ret.fill_between(x, train_mean - train_std, train_mean + train_std,
                                color=color, alpha=0.1)
            ax_ret.plot(x, test_mean, color=color, linewidth=2, label=f"{label} (test)")
            ax_ret.fill_between(x, test_mean - test_std, test_mean + test_std,
                                color=color, alpha=0.15)

            ax_gap.plot(x, gap_mean, color=color, linewidth=2, label=label)
            ax_gap.fill_between(x, gap_mean - gap_std, gap_mean + gap_std,
                                color=color, alpha=0.15)

        ax_ret.set_title(f"{game.capitalize()} — Train vs Test Return", fontsize=13)
        ax_ret.set_ylabel("Episode Return", fontsize=11)
        ax_ret.legend(fontsize=8, ncol=2, loc="upper left")
        ax_ret.grid(True, alpha=0.3)

        ax_gap.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        ax_gap.set_xlabel("Environment Steps (M)", fontsize=11)
        ax_gap.set_ylabel("Gen. Gap (train − test)", fontsize=11)
        ax_gap.legend(fontsize=8, loc="upper right")
        ax_gap.grid(True, alpha=0.3)

        fig.tight_layout()
        out = output_dir / f"overfitting_{game}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] {out}")


# ---------------------------------------------------------------------------
# Plot 3: Final generalization gap bar chart
# ---------------------------------------------------------------------------

def plot_gap_bar_chart(
    curves: dict[str, dict[str, list[pd.DataFrame]]],
    games: list[str],
    methods: list[str],
    output_dir: Path,
) -> None:
    """Grouped bar chart: games on x-axis, bars grouped by method."""
    sns.set_theme(style="whitegrid", font_scale=1.0)
    palette = sns.color_palette("tab10", len(methods))

    n_games = len(games)
    n_methods = len(methods)
    x = np.arange(n_games)
    w = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(max(14, n_games * 1.0), 5))

    for j, (method, color) in enumerate(zip(methods, palette)):
        gaps = []
        errs = []
        for game in games:
            dfs = curves.get(method, {}).get(game, [])
            if dfs:
                final_gaps = [df["gen_gap"].dropna().iloc[-1] for df in dfs if not df.empty]
                gaps.append(np.mean(final_gaps))
                errs.append(np.std(final_gaps))
            else:
                gaps.append(np.nan)
                errs.append(0.0)

        offset = (j - n_methods / 2 + 0.5) * w
        label = METHOD_DISPLAY_NAMES.get(method, method)
        ax.bar(x + offset, gaps, w, yerr=errs, label=label, color=color, alpha=0.82, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([g.capitalize() for g in games], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Final Generalisation Gap (train − test)", fontsize=11)
    ax.set_title("Generalisation Gap by Game and Method", fontsize=13)
    ax.legend(fontsize=8, ncol=3)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = output_dir / "gen_gap_bar_chart.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")


# ---------------------------------------------------------------------------
# Plot 4: Methods × Games heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(
    curves: dict[str, dict[str, list[pd.DataFrame]]],
    games: list[str],
    methods: list[str],
    output_dir: Path,
) -> None:
    """Heatmap: methods (rows) × games (columns), cell = mean final test return."""
    sns.set_theme(style="white", font_scale=0.9)

    matrix = np.full((len(methods), len(games)), np.nan)
    for i, method in enumerate(methods):
        for j, game in enumerate(games):
            dfs = curves.get(method, {}).get(game, [])
            if dfs:
                vals = [df["test_return"].dropna().iloc[-1] for df in dfs if not df.empty]
                if vals:
                    matrix[i, j] = np.mean(vals)

    fig, ax = plt.subplots(figsize=(max(14, len(games) * 0.9), len(methods) * 0.7 + 1.5))
    row_labels = [METHOD_DISPLAY_NAMES.get(m, m) for m in methods]
    col_labels = [g.capitalize() for g in games]

    sns.heatmap(
        matrix,
        ax=ax,
        xticklabels=col_labels,
        yticklabels=row_labels,
        annot=True, fmt=".1f",
        cmap="YlOrRd",
        linewidths=0.5,
        cbar_kws={"label": "Mean Final Test Return"},
    )
    ax.set_title("Final Test Return: Methods × Games", fontsize=13)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    out = output_dir / "test_return_heatmap.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")


# ---------------------------------------------------------------------------
# Plot 5: num_levels sweep
# ---------------------------------------------------------------------------

def plot_num_levels_sweep(
    project: str,
    entity: str | None,
    games: list[str],
    output_dir: Path,
) -> None:
    """Test return vs number of training levels (log x-axis)."""
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path, filters={"state": "finished", "tags": "num_levels_sweep"})

    # Collect: {game: {num_levels: [final_test_return]}}
    data: dict[str, dict[int, list[float]]] = {g: {} for g in games}
    for run in runs:
        cfg  = run.config or {}
        game = cfg.get("env_id")
        nl   = cfg.get("num_levels")
        if game not in games or nl is None:
            continue
        history = run.history(keys=["eval/test_return"], pandas=True)
        if history.empty:
            continue
        final = history["eval/test_return"].dropna().iloc[-1] if len(history) > 0 else None
        if final is None:
            continue
        if nl not in data[game]:
            data[game][nl] = []
        data[game][nl].append(float(final))

    sns.set_theme(style="whitegrid", font_scale=1.1)
    palette = dict(zip(games, sns.color_palette("tab10", len(games))))

    fig, ax = plt.subplots(figsize=(8, 5))
    for game in games:
        game_data = data.get(game, {})
        if not game_data:
            continue
        nl_sorted = sorted(game_data.keys())
        means = [np.mean(game_data[nl]) for nl in nl_sorted]
        stds  = [np.std(game_data[nl])  for nl in nl_sorted]
        ax.plot(nl_sorted, means, "o-", color=palette[game], label=game.capitalize(), linewidth=2)
        ax.fill_between(
            nl_sorted,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            color=palette[game], alpha=0.2,
        )

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Number of Training Levels", fontsize=12)
    ax.set_ylabel("Mean Test Return", fontsize=12)
    ax.set_title("Generalisation vs Training Distribution Size", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = output_dir / "num_levels_sweep.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot generalization curves")
    parser.add_argument("--project",    type=str, default="thesis-rl-baselines")
    parser.add_argument("--entity",     type=str, default=None)
    parser.add_argument("--methods",    nargs="+", default=METHODS)
    parser.add_argument("--games",      nargs="+", default=ALL_GAMES)
    parser.add_argument("--output-dir", type=str, default="figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    curves = fetch_training_curves(args.project, args.entity, args.methods, args.games)

    plot_overfitting_curves(curves, args.games, args.methods, output_dir)
    plot_gap_bar_chart(curves, args.games, args.methods, output_dir)
    plot_heatmap(curves, args.games, args.methods, output_dir)
    plot_num_levels_sweep(args.project, args.entity, args.games, output_dir)

    print("\n[plot_generalization] Done.")


if __name__ == "__main__":
    main()
