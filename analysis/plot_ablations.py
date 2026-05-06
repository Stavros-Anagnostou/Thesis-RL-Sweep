"""
Ablation study plots:
  1. Augmentation type ablation: heatmap (aug_type × game) → test return
  2. Encoder ablation: grouped bar chart (games, bars per encoder)
  3. DrAC component ablation: bar chart (actor_only vs critic_only vs full)

Usage:
    python analysis/plot_ablations.py
    python analysis/plot_ablations.py --output-dir figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

from analysis.rliable_analysis import ALL_GAMES


AUG_TYPES       = ["crop", "color_jitter", "random_conv", "grayscale", "cutout", "flip", "rotate"]
ENCODERS        = ["impala", "nature", "small"]
DRAC_MODES      = ["full", "actor_only", "critic_only"]
REP_GAMES       = ["starpilot", "bigfish", "coinrun", "ninja"]

AUG_DISPLAY     = {
    "crop": "Crop", "color_jitter": "Color Jitter", "random_conv": "Random Conv",
    "grayscale": "Grayscale", "cutout": "Cutout", "flip": "Flip", "rotate": "Rotate",
}
ENCODER_DISPLAY = {"impala": "IMPALA-CNN", "nature": "Nature-CNN", "small": "SmallCNN"}
MODE_DISPLAY    = {"full": "Full DrAC", "actor_only": "Actor only", "critic_only": "Critic only"}


# ---------------------------------------------------------------------------
# Shared W&B fetch
# ---------------------------------------------------------------------------

def fetch_final_test_returns_by_tag(
    project: str,
    entity: str | None,
    tag_key: str,
    tag_values: list[str],
    games: list[str],
    filter_tags: list[str] | None = None,
) -> dict[str, dict[str, list[float]]]:
    """
    Returns: {tag_value: {game: [seed1, seed2, ...]}}
    Filters runs where tag_key=tag_value (from run config).
    """
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    all_runs = api.runs(path, filters={"state": "finished"})

    result: dict[str, dict[str, list[float]]] = {
        v: {g: [] for g in games} for v in tag_values
    }

    for run in all_runs:
        cfg  = run.config or {}
        tags = run.tags or []
        game = cfg.get("env_id")
        if game not in games:
            continue

        val = cfg.get(tag_key)
        if val not in tag_values:
            continue

        if filter_tags and not all(t in tags for t in filter_tags):
            continue

        history = run.history(keys=["eval/test_return"], pandas=True)
        if history.empty:
            continue
        final = history["eval/test_return"].dropna().iloc[-1] if len(history) > 0 else None
        if final is None:
            continue

        result[str(val)][game].append(float(final))

    return result


# ---------------------------------------------------------------------------
# 1. Augmentation type heatmap
# ---------------------------------------------------------------------------

def plot_aug_type_heatmap(
    project: str,
    entity: str | None,
    games: list[str],
    output_dir: Path,
) -> None:
    """Heatmap: augmentation type (rows) × game (cols), cell = mean test return."""
    data = fetch_final_test_returns_by_tag(
        project=project, entity=entity,
        tag_key="aug_type", tag_values=AUG_TYPES,
        games=games,
        filter_tags=["drac"],  # Only DrAC runs — isolates augmentation effect
    )

    matrix = np.full((len(AUG_TYPES), len(games)), np.nan)
    for i, aug in enumerate(AUG_TYPES):
        for j, game in enumerate(games):
            vals = data.get(aug, {}).get(game, [])
            if vals:
                matrix[i, j] = np.mean(vals)

    sns.set_theme(style="white", font_scale=0.9)
    fig, ax = plt.subplots(figsize=(max(12, len(games) * 0.8), len(AUG_TYPES) * 0.75 + 1.5))

    row_labels = [AUG_DISPLAY.get(a, a) for a in AUG_TYPES]
    col_labels = [g.capitalize() for g in games]

    sns.heatmap(
        matrix, ax=ax,
        xticklabels=col_labels, yticklabels=row_labels,
        annot=True, fmt=".1f", cmap="YlGn", linewidths=0.4,
        cbar_kws={"label": "Mean Final Test Return"},
    )
    ax.set_title("DrAC: Test Return by Augmentation Type and Game", fontsize=13)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    out = output_dir / "ablation_aug_type_heatmap.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")


# ---------------------------------------------------------------------------
# 2. Encoder ablation grouped bar chart
# ---------------------------------------------------------------------------

def plot_encoder_ablation(
    project: str,
    entity: str | None,
    games: list[str],
    output_dir: Path,
) -> None:
    """Grouped bar chart: encoder type per game."""
    data = fetch_final_test_returns_by_tag(
        project=project, entity=entity,
        tag_key="encoder", tag_values=ENCODERS,
        games=games,
        filter_tags=["none"],  # Only baseline (no augmentation) runs
    )

    sns.set_theme(style="whitegrid", font_scale=1.0)
    palette = sns.color_palette("Set2", len(ENCODERS))

    n_games   = len(games)
    n_enc     = len(ENCODERS)
    x         = np.arange(n_games)
    w         = 0.75 / n_enc

    fig, ax = plt.subplots(figsize=(max(10, n_games * 1.0), 5))
    for j, (enc, color) in enumerate(zip(ENCODERS, palette)):
        means, errs = [], []
        for game in games:
            vals = data.get(enc, {}).get(game, [])
            means.append(np.mean(vals) if vals else np.nan)
            errs.append(np.std(vals)  if vals else 0.0)
        offset = (j - n_enc / 2 + 0.5) * w
        ax.bar(x + offset, means, w, yerr=errs, label=ENCODER_DISPLAY.get(enc, enc),
               color=color, alpha=0.85, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([g.capitalize() for g in games], rotation=25, ha="right")
    ax.set_ylabel("Mean Final Test Return", fontsize=11)
    ax.set_title("Encoder Ablation: Test Return by Architecture", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = output_dir / "ablation_encoder.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")


# ---------------------------------------------------------------------------
# 3. DrAC component ablation
# ---------------------------------------------------------------------------

def plot_drac_components(
    project: str,
    entity: str | None,
    games: list[str],
    output_dir: Path,
) -> None:
    """Bar chart comparing actor_only, critic_only, and full DrAC."""
    data = fetch_final_test_returns_by_tag(
        project=project, entity=entity,
        tag_key="drac_mode", tag_values=DRAC_MODES,
        games=games,
        filter_tags=["drac"],
    )

    sns.set_theme(style="whitegrid", font_scale=1.1)
    palette = sns.color_palette("Set1", len(DRAC_MODES))

    n_games   = len(games)
    n_modes   = len(DRAC_MODES)
    x         = np.arange(n_games)
    w         = 0.75 / n_modes

    fig, ax = plt.subplots(figsize=(max(8, n_games * 0.9), 5))
    for j, (mode, color) in enumerate(zip(DRAC_MODES, palette)):
        means, errs = [], []
        for game in games:
            vals = data.get(mode, {}).get(game, [])
            means.append(np.mean(vals) if vals else np.nan)
            errs.append(np.std(vals)  if vals else 0.0)
        offset = (j - n_modes / 2 + 0.5) * w
        ax.bar(x + offset, means, w, yerr=errs, label=MODE_DISPLAY.get(mode, mode),
               color=color, alpha=0.82, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([g.capitalize() for g in games], rotation=20, ha="right")
    ax.set_ylabel("Mean Final Test Return", fontsize=11)
    ax.set_title("DrAC Component Ablation: Actor vs Critic Regularisation", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = output_dir / "ablation_drac_components.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ablation studies")
    parser.add_argument("--project",    type=str, default="thesis-rl-baselines")
    parser.add_argument("--entity",     type=str, default=None)
    parser.add_argument("--games",      nargs="+", default=REP_GAMES)
    parser.add_argument("--all-games",  action="store_true", help="Use all 16 games")
    parser.add_argument("--output-dir", type=str, default="figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    games = ALL_GAMES if args.all_games else args.games
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_aug_type_heatmap(args.project, args.entity, games, output_dir)
    plot_encoder_ablation(args.project, args.entity, games, output_dir)
    plot_drac_components(args.project, args.entity, games, output_dir)

    print("\n[plot_ablations] Done.")


if __name__ == "__main__":
    main()
