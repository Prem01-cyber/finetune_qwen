"""
Visualize recursive training behavior (expert phases + replay buffer health).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_iteration_data(training_dir: Path) -> List[Dict]:
    metrics: List[Dict] = []
    for iter_dir in sorted(training_dir.glob("iteration_*")):
        metrics_file = iter_dir / "metrics.json"
        if metrics_file.exists():
            metrics.append(json.loads(metrics_file.read_text(encoding="utf-8")))
    return metrics


def _extract_series(metrics: List[Dict], path: Tuple[str, ...], default: float = 0.0) -> List[float]:
    values: List[float] = []
    for row in metrics:
        cur: object = row
        for key in path:
            if not isinstance(cur, dict):
                cur = default
                break
            cur = cur.get(key, default)
        values.append(float(cur) if isinstance(cur, (int, float)) else float(default))
    return values


def plot_expert_phase_reward(metrics: List[Dict], out_dir: Path) -> None:
    iterations = [int(m.get("iteration", idx + 1)) for idx, m in enumerate(metrics)]
    combined = _extract_series(metrics, ("curriculum", "avg_combined_reward"))
    pre_expert = _extract_series(metrics, ("curriculum", "avg_pre_expert_reward"))
    modifier = _extract_series(metrics, ("curriculum", "avg_expert_modifier"))

    plt.figure(figsize=(12, 6))
    plt.plot(iterations, pre_expert, marker="o", label="pre_expert_reward")
    plt.plot(iterations, combined, marker="o", label="post_expert_reward")
    plt.bar(iterations, modifier, alpha=0.25, label="expert_modifier")
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.title("Expert Phase Reward Shaping Impact")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "expert_phase_reward_impact.png", dpi=200)
    plt.close()


def plot_replay_health(metrics: List[Dict], out_dir: Path) -> None:
    iterations = [int(m.get("iteration", idx + 1)) for idx, m in enumerate(metrics)]
    health = _extract_series(metrics, ("replay_buffer", "buffer_health"))
    avg_quality = _extract_series(metrics, ("replay_buffer", "avg_quality"))
    staleness = _extract_series(metrics, ("replay_buffer", "staleness"))
    entropy = _extract_series(metrics, ("replay_buffer", "topic_entropy"))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()

    axes[0].plot(iterations, health, marker="o")
    axes[0].set_title("Buffer Health")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(alpha=0.3)

    axes[1].plot(iterations, avg_quality, marker="o")
    axes[1].set_title("Average Buffer Quality")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(alpha=0.3)

    axes[2].plot(iterations, staleness, marker="o")
    axes[2].set_title("Buffer Staleness")
    axes[2].grid(alpha=0.3)

    axes[3].plot(iterations, entropy, marker="o")
    axes[3].set_title("Topic Entropy")
    axes[3].grid(alpha=0.3)

    for ax in axes:
        ax.set_xlabel("Iteration")

    plt.suptitle("Replay Buffer Health Dashboard")
    plt.tight_layout()
    plt.savefig(out_dir / "replay_buffer_health_dashboard.png", dpi=200)
    plt.close()


def plot_fresh_vs_replay(metrics: List[Dict], out_dir: Path) -> None:
    iterations = [int(m.get("iteration", idx + 1)) for idx, m in enumerate(metrics)]
    fresh = _extract_series(metrics, ("curriculum", "fresh_mean_reward"))
    replay = _extract_series(metrics, ("curriculum", "replay_mean_reward"))
    ratio = _extract_series(metrics, ("replay_ratio",))

    plt.figure(figsize=(12, 6))
    plt.plot(iterations, fresh, marker="o", label="fresh_mean_reward")
    plt.plot(iterations, replay, marker="o", label="replay_mean_reward")
    plt.plot(iterations, ratio, marker="x", linestyle="--", label="replay_ratio")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title("Fresh vs Replay Trajectory Quality")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fresh_vs_replay_quality.png", dpi=200)
    plt.close()


def plot_generation_cohorts(metrics: List[Dict], out_dir: Path) -> None:
    iterations = [int(m.get("iteration", idx + 1)) for idx, m in enumerate(metrics)]
    reward = _extract_series(metrics, ("curriculum", "avg_combined_reward"))
    replay_added = _extract_series(metrics, ("curriculum", "replay_added_count"))
    replay_success = _extract_series(metrics, ("replay_buffer", "replay_success_rate"))

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(iterations, reward, color="tab:blue", marker="o", label="avg_combined_reward")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Reward", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.bar(iterations, replay_added, alpha=0.25, color="tab:green", label="replay_admissions")
    ax2.plot(iterations, replay_success, color="tab:red", marker="x", label="replay_success_rate")
    ax2.set_ylabel("Replay Dynamics", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    plt.title("Generational Cohort Progress (Admissions + Success)")
    fig.tight_layout()
    plt.savefig(out_dir / "generation_cohort_progress.png", dpi=200)
    plt.close()


def write_summary(metrics: List[Dict], out_dir: Path) -> None:
    if not metrics:
        return
    first = metrics[0]
    last = metrics[-1]
    summary = {
        "iterations": len(metrics),
        "initial_reward": first.get("curriculum", {}).get("avg_combined_reward", 0.0),
        "final_reward": last.get("curriculum", {}).get("avg_combined_reward", 0.0),
        "initial_buffer_health": first.get("replay_buffer", {}).get("buffer_health", 0.0),
        "final_buffer_health": last.get("replay_buffer", {}).get("buffer_health", 0.0),
        "max_replay_ratio": max(float(m.get("replay_ratio", 0.0)) for m in metrics),
    }
    (out_dir / "recursive_learning_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize recursive replay learning outputs")
    parser.add_argument("--training-dir", required=True, type=str)
    parser.add_argument("--output-dir", type=str, default="")
    args = parser.parse_args()

    training_dir = Path(args.training_dir)
    output_dir = Path(args.output_dir) if args.output_dir else training_dir / "recursive_visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_iteration_data(training_dir)
    if not metrics:
        raise SystemExit(f"No iteration metrics found in {training_dir}")

    plot_expert_phase_reward(metrics, output_dir)
    plot_replay_health(metrics, output_dir)
    plot_fresh_vs_replay(metrics, output_dir)
    plot_generation_cohorts(metrics, output_dir)
    write_summary(metrics, output_dir)

    print(f"Saved recursive-learning visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
