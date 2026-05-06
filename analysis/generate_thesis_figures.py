"""
Master figure generation script for the MSc thesis.

Calls all analysis sub-scripts and produces a numbered, consistently-styled
set of publication-quality PDFs in figures/.

Figure inventory:
  fig01_overfitting_*.pdf        — per-game train vs test curves (one per game)
  fig02_gen_gap_bar.pdf          — final generalisation gap by game & method
  fig03_test_return_heatmap.pdf  — methods × games test return heatmap
  fig04_num_levels_sweep.pdf     — training distribution size ablation
  fig05_rliable_iqm.pdf          — IQM aggregate performance
  fig06_rliable_profiles.pdf     — performance profiles
  fig07_rliable_prob_improvement.pdf — P(method > baseline)
  fig08_ablation_aug_type.pdf    — augmentation type heatmap
  fig09_ablation_encoder.pdf     — encoder ablation bar chart
  fig10_ablation_drac_components.pdf — DrAC component ablation

Usage:
    python analysis/generate_thesis_figures.py
    python analysis/generate_thesis_figures.py --project thesis-rl-baselines --output-dir figures/
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# ---------------------------------------------------------------------------
# Global style settings for camera-ready thesis figures
# ---------------------------------------------------------------------------

def apply_thesis_style() -> None:
    """
    Apply consistent Seaborn + Matplotlib styling across all figures.

    - whitegrid background
    - Colorblind-safe palette (IBM's 8-color palette)
    - Font size suitable for thesis body text (10pt figures, 8pt labels)
    - No top/right spines
    - 300 DPI for PDF/PNG export
    """
    import seaborn as sns

    sns.set_theme(
        style="whitegrid",
        palette="colorblind",
        font_scale=1.15,
        rc={
            "figure.dpi":         300,
            "savefig.dpi":        300,
            "savefig.format":     "pdf",
            "savefig.bbox":       "tight",
            "axes.spines.top":    False,
            "axes.spines.right":  False,
            "axes.titlesize":     13,
            "axes.labelsize":     11,
            "xtick.labelsize":    9,
            "ytick.labelsize":    9,
            "legend.fontsize":    9,
            "lines.linewidth":    2.0,
            "patch.linewidth":    0.5,
            "font.family":        "sans-serif",
            # Use DejaVu if available; falls back gracefully.
            "font.sans-serif":    ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
        },
    )
    print("[style] Thesis matplotlib/seaborn style applied.")


# ---------------------------------------------------------------------------
# Run a sub-script as a subprocess (captures stdout/stderr)
# ---------------------------------------------------------------------------

def run_script(
    script_path: str,
    extra_args: list[str],
    label: str,
) -> bool:
    """Run a Python analysis script, return True on success."""
    cmd = [sys.executable, script_path] + extra_args
    print(f"\n[{label}] Running: {' '.join(cmd)}")
    t = time.time()
    try:
        subprocess.run(cmd, check=True)
        print(f"[{label}] ✓ Completed in {time.time() - t:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[{label}] ✗ FAILED (exit code {e.returncode})")
        return False


# ---------------------------------------------------------------------------
# Rename outputs with figure numbers
# ---------------------------------------------------------------------------

FIGURE_MAP: dict[str, str] = {
    # overfitting curves: not renamed individually — already named per game
    "gen_gap_bar_chart.pdf":               "fig02_gen_gap_bar.pdf",
    "test_return_heatmap.pdf":             "fig03_test_return_heatmap.pdf",
    "num_levels_sweep.pdf":                "fig04_num_levels_sweep.pdf",
    "rliable_iqm.pdf":                     "fig05_rliable_iqm.pdf",
    "rliable_profiles.pdf":                "fig06_rliable_profiles.pdf",
    "rliable_prob_improvement.pdf":        "fig07_rliable_prob_improvement.pdf",
    "ablation_aug_type_heatmap.pdf":       "fig08_ablation_aug_type.pdf",
    "ablation_encoder.pdf":                "fig09_ablation_encoder.pdf",
    "ablation_drac_components.pdf":        "fig10_ablation_drac_components.pdf",
}


def rename_with_numbers(output_dir: Path) -> None:
    """Rename generated figures to thesis-numbered filenames."""
    for src_name, dst_name in FIGURE_MAP.items():
        src = output_dir / src_name
        dst = output_dir / dst_name
        if src.exists():
            shutil.move(str(src), str(dst))
            print(f"  {src_name} → {dst_name}")

    # Rename overfitting curves: overfitting_<game>.pdf → fig01_overfitting_<game>.pdf
    for p in sorted(output_dir.glob("overfitting_*.pdf")):
        new_name = "fig01_" + p.name
        shutil.move(str(p), str(output_dir / new_name))
        print(f"  {p.name} → {new_name}")


# ---------------------------------------------------------------------------
# Summary page: table of all figures
# ---------------------------------------------------------------------------

def generate_figure_index(output_dir: Path) -> None:
    """Write a simple text index of all generated figures."""
    index_path = output_dir / "FIGURE_INDEX.txt"
    figures = sorted(output_dir.glob("fig*.pdf"))
    with open(index_path, "w") as f:
        f.write("Thesis Figure Index\n")
        f.write("=" * 50 + "\n\n")
        for fig in figures:
            f.write(f"  {fig.name}\n")
        f.write(f"\nTotal: {len(figures)} figures\n")
    print(f"\n[index] {index_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate all thesis figures")
    parser.add_argument("--project",     type=str, default="thesis-rl-baselines")
    parser.add_argument("--entity",      type=str, default=None)
    parser.add_argument("--output-dir",  type=str, default="figures")
    parser.add_argument("--methods",     nargs="+", default=None,
                        help="Subset of methods to include (default: all)")
    parser.add_argument("--games",       nargs="+", default=None,
                        help="Subset of games to include (default: all 16)")
    parser.add_argument("--rep-games",   action="store_true",
                        help="Use representative 4-game subset for ablation plots")
    parser.add_argument("--skip-rliable", action="store_true",
                        help="Skip rliable computations (faster, no IQM)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    apply_thesis_style()

    # Build shared CLI args for sub-scripts
    wandb_args = ["--project", args.project]
    if args.entity:
        wandb_args += ["--entity", args.entity]
    wandb_args += ["--output-dir", str(output_dir)]

    method_args = (["--methods"] + args.methods) if args.methods else []
    game_args   = (["--games"] + args.games) if args.games else []

    print(f"\n{'='*60}")
    print(f"  Generating thesis figures")
    print(f"  W&B project : {args.project}")
    print(f"  Output dir  : {output_dir}")
    print(f"{'='*60}")

    results: dict[str, bool] = {}

    # --- 1. Main generalization plots ---
    results["generalization"] = run_script(
        "analysis/plot_generalization.py",
        wandb_args + method_args + game_args,
        "plot_generalization",
    )

    # --- 2. rliable statistical analysis ---
    if not args.skip_rliable:
        results["rliable"] = run_script(
            "analysis/rliable_analysis.py",
            wandb_args + method_args + game_args,
            "rliable",
        )
    else:
        print("\n[rliable] Skipped (--skip-rliable).")

    # --- 3. Ablation plots ---
    ablation_game_args = ["--rep-games"] if args.rep_games else (game_args or [])
    results["ablations"] = run_script(
        "analysis/plot_ablations.py",
        wandb_args + ablation_game_args,
        "plot_ablations",
    )

    # --- 4. Baseline training curves (re-use existing plot_baselines.py) ---
    results["baselines"] = run_script(
        "analysis/plot_baselines.py",
        wandb_args,
        "plot_baselines",
    )

    # --- Rename with thesis figure numbers ---
    print("\n[rename] Applying thesis figure numbering...")
    rename_with_numbers(output_dir)

    # --- Generate index ---
    generate_figure_index(output_dir)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  Figure generation summary")
    for script, ok in results.items():
        print(f"  {'✓' if ok else '✗'}  {script}")
    total_figs = len(list(output_dir.glob("fig*.pdf")))
    print(f"\n  {total_figs} figures written to {output_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
