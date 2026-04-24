#!/usr/bin/env python3
"""
Generate demo-quality plots from a completed (or in-progress) GRPO run.

Usage
-----
    # from the run output directory
    python scripts/plot_grpo_run.py checkpoints/grpo/<run_name>/metrics.jsonl

    # auto-discover the latest run
    python scripts/plot_grpo_run.py --latest

    # custom output directory
    python scripts/plot_grpo_run.py metrics.jsonl --out-dir plots/my_run

Output
------
Six PNG files saved next to the JSONL (or --out-dir if given):

  01_training_objective.png   – combined_score vs iteration (PRIMARY demo plot)
  02_reward_components.png    – 4-panel breakdown: correct / PRM / SymPy / format
  03_training_dynamics.png    – GRPO loss + batch reward + batch accuracy
  04_reward_vs_eval.png       – training reward vs eval score on same axis
  05_component_area.png       – stacked-area chart of the 4 weighted components
  06_summary_card.png         – single-panel card: all key metrics in one view

All figures use a clean dark-on-white academic style.  They are saved at
300 dpi so they look sharp in slides and posters.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # headless — no display needed on training servers
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np


# ── Style ────────────────────────────────────────────────────────────────────

PALETTE = {
    "combined":  "#2563EB",   # blue  — training objective
    "correct":   "#16A34A",   # green — correctness
    "prm":       "#DC2626",   # red   — PRM step quality
    "sympy":     "#D97706",   # amber — SymPy verification
    "fmt":       "#7C3AED",   # violet — format
    "reward":    "#0891B2",   # cyan  — mean batch reward
    "loss":      "#64748B",   # slate — loss
    "batch_acc": "#059669",   # emerald — batch accuracy
}

plt.rcParams.update({
    "figure.dpi":          150,
    "savefig.dpi":         300,
    "font.family":         "DejaVu Sans",
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.grid":           True,
    "grid.alpha":          0.3,
    "grid.linestyle":      "--",
    "axes.labelsize":      11,
    "axes.titlesize":      13,
    "legend.fontsize":     9,
    "xtick.labelsize":     9,
    "ytick.labelsize":     9,
})


# ── Data loading ─────────────────────────────────────────────────────────────

def _load(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _field(rows: List[Dict], key: str) -> Tuple[List[int], List[float]]:
    """Return (iterations, values) for rows that have a non-empty key."""
    iters, vals = [], []
    for r in rows:
        v = r.get(key)
        if v is not None and v != "" and not (isinstance(v, float) and np.isnan(v)):
            try:
                iters.append(int(r["iteration"]))
                vals.append(float(v))
            except (TypeError, ValueError):
                pass
    return iters, vals


# ── Individual plots ─────────────────────────────────────────────────────────

def plot_training_objective(rows: List[Dict], out: Path) -> None:
    """Plot 01: combined_score — the single most important demo plot."""
    xi, xv = _field(rows, "combined_score")
    if not xi:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xi, xv, color=PALETTE["combined"], linewidth=2.5,
            marker="o", markersize=5, label="Training-objective score")
    ax.fill_between(xi, xv, alpha=0.12, color=PALETTE["combined"])

    # annotate first and last eval points
    ax.annotate(f"{xv[0]:.3f}", (xi[0], xv[0]), textcoords="offset points",
                xytext=(8, 6), fontsize=8, color=PALETTE["combined"])
    ax.annotate(f"{xv[-1]:.3f}", (xi[-1], xv[-1]), textcoords="offset points",
                xytext=(8, 6), fontsize=8, color=PALETTE["combined"])

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Score  (0 – 1)")
    ax.set_title(
        "GRPO Training — Combined Reward Score\n"
        "0.60 × correct + 0.15 × PRM + 0.15 × SymPy + 0.10 × format",
        fontsize=12,
    )
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out.name}")


def plot_reward_components(rows: List[Dict], out: Path) -> None:
    """Plot 02: four-panel breakdown of each reward component."""
    specs = [
        ("correct_rate",   "correct",  "Correctness (gt_match)",           "60 %"),
        ("prm_mean",       "prm",      "PRM Step Quality",                  "15 %"),
        ("sympy_mean",     "sympy",    "SymPy Verification",                "15 %"),
        ("format_mean",    "fmt",      "Format Compliance",                 "10 %"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=False)
    axes = axes.flatten()

    for ax, (key, pal, title, weight) in zip(axes, specs):
        xi, xv = _field(rows, key)
        if not xi:
            ax.set_visible(False)
            continue
        ax.plot(xi, xv, color=PALETTE[pal], linewidth=2,
                marker="o", markersize=4)
        ax.fill_between(xi, xv, alpha=0.12, color=PALETTE[pal])
        ax.set_title(f"{title}  (weight {weight})", fontsize=11)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))

        if xv:
            delta = xv[-1] - xv[0]
            sign = "+" if delta >= 0 else ""
            ax.set_title(
                f"{title}  (weight {weight})  Δ={sign}{delta:+.1%}",
                fontsize=10,
            )

    fig.suptitle("Reward Component Breakdown over Training", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


def plot_training_dynamics(rows: List[Dict], out: Path) -> None:
    """Plot 03: loss, mean_reward, batch_accuracy over all iterations."""
    li, lv = _field(rows, "loss")
    ri, rv = _field(rows, "mean_reward")
    bi, bv = _field(rows, "batch_accuracy")

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    if lv:
        axes[0].plot(li, lv, color=PALETTE["loss"], linewidth=1.8)
        axes[0].fill_between(li, lv, alpha=0.1, color=PALETTE["loss"])
        axes[0].set_ylabel("GRPO Loss")
        axes[0].set_title("Training Loss", fontsize=11)
        axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)

    if rv:
        axes[1].plot(ri, rv, color=PALETTE["reward"], linewidth=1.8)
        axes[1].fill_between(ri, rv, alpha=0.1, color=PALETTE["reward"])
        axes[1].set_ylabel("Reward")
        axes[1].set_ylim(0, 1.05)
        axes[1].set_title("Mean Batch Reward", fontsize=11)
        axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))

    if bv:
        axes[2].plot(bi, bv, color=PALETTE["batch_acc"], linewidth=1.8)
        axes[2].fill_between(bi, bv, alpha=0.1, color=PALETTE["batch_acc"])
        axes[2].set_ylabel("Accuracy")
        axes[2].set_ylim(0, 1.05)
        axes[2].set_title("Batch Accuracy (training rollouts)", fontsize=11)
        axes[2].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))

    for ax in axes:
        ax.set_xlabel("Iteration")

    fig.suptitle("GRPO Training Dynamics", fontsize=13)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out.name}")


def plot_reward_vs_eval(rows: List[Dict], out: Path) -> None:
    """Plot 04: mean_reward (all iters) + combined_score (eval iters) overlaid."""
    ri, rv = _field(rows, "mean_reward")
    ei, ev = _field(rows, "combined_score")

    fig, ax = plt.subplots(figsize=(10, 5))

    if rv:
        ax.plot(ri, rv, color=PALETTE["reward"], linewidth=1.4, alpha=0.7,
                label="Batch reward (training)")
        ax.fill_between(ri, rv, alpha=0.06, color=PALETTE["reward"])

    if ev:
        ax.plot(ei, ev, color=PALETTE["combined"], linewidth=2.5,
                marker="D", markersize=6, label="Eval score (held-out GSM8K)")
        for x, y in zip(ei, ev):
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=7,
                        color=PALETTE["combined"])

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Score  (0 – 1)")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_title("Training Reward vs Held-Out Eval Score", fontsize=12)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out.name}")


def plot_component_area(rows: List[Dict], out: Path) -> None:
    """Plot 05: stacked-area of the four WEIGHTED components summing to combined_score."""
    ei, ev_combined = _field(rows, "combined_score")
    if not ei:
        return

    # Build per-component weighted series aligned to eval iterations
    iter_set = set(ei)
    aligned: Dict[str, List[float]] = {k: [] for k in ("correct", "prm", "sympy", "fmt")}
    weights = {"correct": 0.60, "prm": 0.15, "sympy": 0.15, "fmt": 0.10}
    keys    = {"correct": "correct_rate", "prm": "prm_mean",
               "sympy": "sympy_mean",     "fmt": "format_mean"}

    # Build lookup per iteration
    it_map: Dict[int, Dict] = {r["iteration"]: r for r in rows if r["iteration"] in iter_set}
    iters_sorted = sorted(iter_set)

    for it in iters_sorted:
        row = it_map.get(it, {})
        for comp, field in keys.items():
            v = row.get(field)
            if v is not None and v != "":
                aligned[comp].append(float(v) * weights[comp])
            else:
                aligned[comp].append(0.0)

    x = np.array(iters_sorted)
    arr = np.array([aligned["correct"], aligned["prm"],
                    aligned["sympy"],   aligned["fmt"]])

    fig, ax = plt.subplots(figsize=(10, 5))
    labels  = ["Correct (×0.60)", "PRM (×0.15)", "SymPy (×0.15)", "Format (×0.10)"]
    colors  = [PALETTE[k] for k in ("correct", "prm", "sympy", "fmt")]
    ax.stackplot(x, arr, labels=labels, colors=colors, alpha=0.75)

    ax.plot(x, ev_combined, color="black", linewidth=1.5,
            linestyle="--", label="Combined score", zorder=5)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Weighted contribution to score")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
    ax.set_title("Contribution of Each Reward Component (Stacked)", fontsize=12)
    ax.legend(loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved {out.name}")


def plot_summary_card(rows: List[Dict], run_name: str, out: Path) -> None:
    """Plot 06: all key metrics on a single clean card — ideal for poster / slide."""
    ei, ev = _field(rows, "combined_score")
    _, crv  = _field(rows, "correct_rate")
    _, prmv = _field(rows, "prm_mean")
    _, syv  = _field(rows, "sympy_mean")
    _, fmv  = _field(rows, "format_mean")
    _, lv   = _field(rows, "loss")
    _, rv   = _field(rows, "mean_reward")
    li      = _field(rows, "loss")[0]
    ri      = _field(rows, "mean_reward")[0]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    def _panel(ax, iters, vals, color, title, pct=True):
        if not iters:
            ax.set_visible(False)
            return
        ax.plot(iters, vals, color=color, linewidth=2, marker="o", markersize=4)
        ax.fill_between(iters, vals, alpha=0.12, color=color)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Iteration", fontsize=9)
        if pct:
            ax.set_ylim(0, 1.05)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
        if vals:
            ax.annotate(f"{vals[-1]:.3f}", (iters[-1], vals[-1]),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=8, color=color)

    _panel(axes[0], ei,  ev,   PALETTE["combined"],  "Training-Objective Score")
    _panel(axes[1], ei,  crv,  PALETTE["correct"],   "Correctness Rate")
    _panel(axes[2], ei,  prmv, PALETTE["prm"],       "PRM Step Quality")
    _panel(axes[3], ei,  syv,  PALETTE["sympy"],     "SymPy Verification")
    _panel(axes[4], ei,  fmv,  PALETTE["fmt"],       "Format Compliance")
    _panel(axes[5], li,  lv,   PALETTE["loss"],      "GRPO Loss", pct=False)

    fig.suptitle(f"GRPO Training Summary — {run_name}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def find_latest_metrics() -> Optional[Path]:
    """Find the most recently modified metrics.jsonl under checkpoints/grpo/."""
    ckpt = Path("checkpoints/grpo")
    if not ckpt.exists():
        return None
    candidates = sorted(
        ckpt.rglob("metrics.jsonl"),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def generate_plots(metrics_path: Path, out_dir: Optional[Path] = None) -> Path:
    """Generate all six plots and return the output directory."""
    rows = _load(metrics_path)
    if not rows:
        print(f"[plot] No data in {metrics_path}", file=sys.stderr)
        return metrics_path.parent

    out_dir = out_dir or metrics_path.parent / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Derive run name from the directory name two levels up
    run_name = metrics_path.parent.name

    print(f"[plot] Generating plots for run '{run_name}'  ({len(rows)} iterations)")
    print(f"[plot] Output → {out_dir}")

    plot_training_objective(rows, out_dir / "01_training_objective.png")
    plot_reward_components(rows,  out_dir / "02_reward_components.png")
    plot_training_dynamics(rows,  out_dir / "03_training_dynamics.png")
    plot_reward_vs_eval(rows,     out_dir / "04_reward_vs_eval.png")
    plot_component_area(rows,     out_dir / "05_component_area.png")
    plot_summary_card(rows, run_name, out_dir / "06_summary_card.png")

    print(f"[plot] Done — {len(list(out_dir.glob('*.png')))} PNGs in {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate demo plots from a GRPO metrics.jsonl file."
    )
    parser.add_argument(
        "metrics_jsonl", nargs="?", type=Path, default=None,
        help="Path to metrics.jsonl produced by run_grpo_training.py",
    )
    parser.add_argument(
        "--latest", action="store_true",
        help="Auto-discover the most recent metrics.jsonl under checkpoints/grpo/",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Directory to write PNG files (default: <metrics_dir>/plots/)",
    )
    args = parser.parse_args()

    if args.latest:
        path = find_latest_metrics()
        if path is None:
            print("No metrics.jsonl found under checkpoints/grpo/", file=sys.stderr)
            sys.exit(1)
        print(f"[plot] Auto-selected {path}")
    elif args.metrics_jsonl:
        path = args.metrics_jsonl
    else:
        parser.print_help()
        sys.exit(1)

    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    generate_plots(path, args.out_dir)


if __name__ == "__main__":
    main()
