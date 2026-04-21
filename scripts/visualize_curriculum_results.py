"""
Visualize curriculum-driven PPO training results.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - optional dependency
    go = None


def load_iteration_data(training_dir: Path) -> Tuple[List[Dict], List[Dict]]:
    metrics = []
    trajectories = []
    for iter_dir in sorted(training_dir.glob("iteration_*")):
        metrics_file = iter_dir / "metrics.json"
        traj_file = iter_dir / "trajectories.jsonl"
        if metrics_file.exists():
            metrics.append(json.loads(metrics_file.read_text(encoding="utf-8")))
        if traj_file.exists():
            with traj_file.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        trajectories.append(json.loads(line))
    return metrics, trajectories


def plot_topic_success_over_time(metrics: List[Dict], out_dir: Path) -> None:
    iterations = [m["iteration"] for m in metrics]
    topic_series: Dict[str, List[float]] = defaultdict(list)

    for m in metrics:
        per_topic = m.get("curriculum", {}).get("per_topic_success", {})
        for topic in per_topic:
            topic_series[topic].append(per_topic[topic])
        missing = set(topic_series.keys()) - set(per_topic.keys())
        for topic in missing:
            topic_series[topic].append(np.nan)

    plt.figure(figsize=(12, 6))
    for topic, values in sorted(topic_series.items()):
        plt.plot(iterations, values, marker="o", linewidth=1.5, label=topic)
    plt.axhspan(0.4, 0.7, alpha=0.15)
    plt.xlabel("Iteration")
    plt.ylabel("Success Rate")
    plt.title("Topic Success Rate Over Time")
    plt.ylim(0.0, 1.0)
    plt.legend(loc="best", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "topic_success_over_time.png", dpi=200)
    plt.close()


def plot_difficulty_progression(trajectories: List[Dict], out_dir: Path) -> None:
    grouped_target: Dict[str, List[float]] = defaultdict(list)
    grouped_actual: Dict[str, List[float]] = defaultdict(list)
    for row in trajectories:
        meta = row.get("metadata", {})
        topic = meta.get("target_topic", "unknown")
        grouped_target[topic].append(float(meta.get("target_difficulty", 0.0)))
        grouped_actual[topic].append(float(meta.get("estimated_difficulty", 0.0)))

    topics = sorted(grouped_target.keys())
    x = np.arange(len(topics))
    target_vals = [np.mean(grouped_target[t]) if grouped_target[t] else 0.0 for t in topics]
    actual_vals = [np.mean(grouped_actual[t]) if grouped_actual[t] else 0.0 for t in topics]

    plt.figure(figsize=(12, 6))
    plt.bar(x - 0.2, target_vals, width=0.4, label="target")
    plt.bar(x + 0.2, actual_vals, width=0.4, label="measured")
    plt.xticks(x, topics, rotation=35, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Difficulty")
    plt.title("Target vs Measured Difficulty by Topic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "difficulty_progression.png", dpi=200)
    plt.close()


def plot_topic_distribution(trajectories: List[Dict], out_dir: Path) -> None:
    topic_counts = Counter()
    for row in trajectories:
        topic_counts[row.get("metadata", {}).get("target_topic", "unknown")] += 1
    labels = list(topic_counts.keys())
    values = [topic_counts[label] for label in labels]
    if not values:
        return
    plt.figure(figsize=(8, 8))
    plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.title("Topic Distribution Across Trajectories")
    plt.tight_layout()
    plt.savefig(out_dir / "topic_distribution.png", dpi=200)
    plt.close()


def plot_reward_breakdown(metrics: List[Dict], out_dir: Path) -> None:
    iterations = [m["iteration"] for m in metrics]
    q = [m.get("curriculum", {}).get("avg_question_reward", 0.0) for m in metrics]
    s = [m.get("curriculum", {}).get("avg_solution_reward", 0.0) for m in metrics]
    total = [m.get("curriculum", {}).get("avg_combined_reward", 0.0) for m in metrics]

    plt.figure(figsize=(12, 6))
    plt.stackplot(iterations, q, s, labels=["question_reward", "solution_reward"], alpha=0.6)
    plt.plot(iterations, total, color="black", linewidth=2, label="combined")
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.title("Reward Breakdown Over Time")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "reward_breakdown.png", dpi=200)
    plt.close()


def plot_curriculum_heatmap(metrics: List[Dict], out_dir: Path) -> None:
    topic_names = sorted(
        {
            topic
            for metric in metrics
            for topic in metric.get("curriculum", {}).get("per_topic_success", {}).keys()
        }
    )
    if not topic_names:
        return
    matrix = []
    for metric in metrics:
        per_topic = metric.get("curriculum", {}).get("per_topic_success", {})
        matrix.append([per_topic.get(topic, np.nan) for topic in topic_names])
    data = np.array(matrix, dtype=np.float64).T

    plt.figure(figsize=(12, 8))
    plt.imshow(data, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Success Rate")
    plt.yticks(range(len(topic_names)), topic_names)
    plt.xticks(range(len(metrics)), [m["iteration"] for m in metrics])
    plt.title("Curriculum Success Heatmap")
    plt.xlabel("Iteration")
    plt.ylabel("Topic")
    plt.tight_layout()
    plt.savefig(out_dir / "curriculum_heatmap.png", dpi=200)
    plt.close()


def plot_question_quality_evolution(metrics: List[Dict], out_dir: Path) -> None:
    iterations = [m["iteration"] for m in metrics]
    topic_match = [m.get("curriculum", {}).get("avg_topic_match", 0.0) for m in metrics]
    diff_match = [m.get("curriculum", {}).get("avg_difficulty_match", 0.0) for m in metrics]
    clarity = [m.get("curriculum", {}).get("avg_clarity", 0.0) for m in metrics]
    novelty = [m.get("curriculum", {}).get("avg_novelty", 0.0) for m in metrics]

    plt.figure(figsize=(12, 6))
    plt.stackplot(iterations, topic_match, diff_match, clarity, novelty, labels=["topic", "difficulty", "clarity", "novelty"], alpha=0.65)
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    plt.title("Question Quality Component Evolution")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "question_quality_evolution.png", dpi=200)
    plt.close()


def plot_topic_transitions(trajectories: List[Dict], out_dir: Path) -> None:
    ordered = sorted(
        trajectories,
        key=lambda row: (
            row.get("metadata", {}).get("curriculum_iteration", 0),
            row.get("trajectory_id", 0),
        ),
    )
    transitions = Counter()
    prev_topic = None
    for row in ordered:
        topic = row.get("metadata", {}).get("target_topic", "unknown")
        if prev_topic is not None and prev_topic != topic:
            transitions[(prev_topic, topic)] += 1
        prev_topic = topic

    if go is None or not transitions:
        return

    nodes = sorted(set([src for src, _ in transitions.keys()] + [dst for _, dst in transitions.keys()]))
    idx = {node: i for i, node in enumerate(nodes)}
    source = [idx[src] for src, _ in transitions.keys()]
    target = [idx[dst] for _, dst in transitions.keys()]
    value = [count for count in transitions.values()]

    fig = go.Figure(
        data=[
            go.Sankey(
                node={"label": nodes, "pad": 12, "thickness": 18},
                link={"source": source, "target": target, "value": value},
            )
        ]
    )
    fig.update_layout(title="Topic Transition Flow")
    fig.write_html(str(out_dir / "topic_transition_sankey.html"))


def plot_success_distribution(metrics: List[Dict], out_dir: Path) -> None:
    if not metrics:
        return
    sample_metrics = [
        metrics[0],
        metrics[len(metrics) // 3],
        metrics[(2 * len(metrics)) // 3],
        metrics[-1],
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    bins = [0.0, 0.3, 0.4, 0.7, 0.8, 1.0]
    labels = ["0-0.3", "0.3-0.4", "0.4-0.7", "0.7-0.8", "0.8-1.0"]
    for ax, metric in zip(axes, sample_metrics):
        values = list(metric.get("curriculum", {}).get("per_topic_success", {}).values())
        counts, _ = np.histogram(values, bins=bins)
        ax.bar(range(len(counts)), counts)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(labels, rotation=25)
        ax.set_title(f"Iteration {metric['iteration']}")
        ax.set_ylabel("Topics")
    plt.suptitle("Topic Success Distribution Over Training")
    plt.tight_layout()
    plt.savefig(out_dir / "success_distribution.png", dpi=200)
    plt.close()


def write_summary(metrics: List[Dict], trajectories: List[Dict], out_dir: Path) -> None:
    if not metrics:
        return
    first = metrics[0]
    last = metrics[-1]
    summary = {
        "iterations": len(metrics),
        "num_trajectories": len(trajectories),
        "initial_combined_reward": first.get("curriculum", {}).get("avg_combined_reward", 0.0),
        "final_combined_reward": last.get("curriculum", {}).get("avg_combined_reward", 0.0),
        "initial_topics_in_sweet_spot": first.get("curriculum", {}).get("topics_in_sweet_spot", 0),
        "final_topics_in_sweet_spot": last.get("curriculum", {}).get("topics_in_sweet_spot", 0),
    }
    (out_dir / "curriculum_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Visualize curriculum training outputs")
    parser.add_argument("--training-dir", required=True, type=str)
    parser.add_argument("--output-dir", type=str, default="")
    args = parser.parse_args()

    training_dir = Path(args.training_dir)
    output_dir = Path(args.output_dir) if args.output_dir else training_dir / "plots_curriculum"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics, trajectories = load_iteration_data(training_dir)
    if not metrics:
        raise RuntimeError(f"No iteration metrics found under {training_dir}")

    plot_topic_success_over_time(metrics, output_dir)
    plot_difficulty_progression(trajectories, output_dir)
    plot_topic_distribution(trajectories, output_dir)
    plot_reward_breakdown(metrics, output_dir)
    plot_curriculum_heatmap(metrics, output_dir)
    plot_question_quality_evolution(metrics, output_dir)
    plot_topic_transitions(trajectories, output_dir)
    plot_success_distribution(metrics, output_dir)
    write_summary(metrics, trajectories, output_dir)

    print(f"Saved curriculum visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
