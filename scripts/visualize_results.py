"""
Visualization and Analysis for PPO Training

Creates plots and analysis for hackathon submission:
1. Reward curves over iterations
2. Question difficulty progression
3. Solution quality improvement
4. GSM8K accuracy tracking
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")


def load_iteration_data(output_dir: Path) -> dict:
    """
    Load all iteration data for analysis.

    Returns:
        Dict mapping iteration → {metrics, trajectories}
    """
    data = defaultdict(dict)

    for iter_dir in sorted(output_dir.glob("iteration_*")):
        iter_num = int(iter_dir.name.split("_")[1])

        metrics_path = iter_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                data[iter_num]["metrics"] = json.load(f)

        traj_path = iter_dir / "trajectories.jsonl"
        if traj_path.exists():
            trajectories = []
            with open(traj_path) as f:
                for line in f:
                    trajectories.append(json.loads(line))
            data[iter_num]["trajectories"] = trajectories

    return data


def plot_reward_curves(data: dict, save_path: Path):
    """
    Plot reward progression over iterations.

    Shows:
    - Mean reward per iteration with ±1 std band
    - Question vs solution reward components
    - Per-iteration reward distribution (violin)
    - GSM8K accuracy curve
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    iterations = sorted(data.keys())

    mean_rewards, std_rewards = [], []
    question_rewards, solution_rewards = [], []

    for iter_num in iterations:
        trajs = data[iter_num]["trajectories"]

        rewards = [t["total_reward"] for t in trajs]
        mean_rewards.append(np.mean(rewards))
        std_rewards.append(np.std(rewards))

        q_rewards = [
            t["reward_breakdown"]["question_metrics"]["overall_score"]
            for t in trajs
        ]
        s_rewards = [
            t["reward_breakdown"]["solution_metrics"]["overall_score"]
            for t in trajs
        ]

        question_rewards.append(np.mean(q_rewards))
        solution_rewards.append(np.mean(s_rewards))

    # Plot 1: Mean reward with std
    ax = axes[0, 0]
    ax.plot(iterations, mean_rewards, "b-", linewidth=2, label="Mean Reward")
    ax.fill_between(
        iterations,
        np.array(mean_rewards) - np.array(std_rewards),
        np.array(mean_rewards) + np.array(std_rewards),
        alpha=0.3,
        label="±1 std",
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Combined Reward")
    ax.set_title("Reward Progression")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Question vs Solution rewards
    ax = axes[0, 1]
    ax.plot(iterations, question_rewards, "g-", linewidth=2, label="Question Quality")
    ax.plot(iterations, solution_rewards, "r-", linewidth=2, label="Solution Quality")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Reward Component")
    ax.set_title("Question vs Solution Quality")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Reward distribution (violin plot)
    ax = axes[1, 0]
    reward_data = []
    for iter_num in iterations:
        trajs = data[iter_num]["trajectories"]
        for t in trajs:
            reward_data.append({"Iteration": iter_num, "Reward": t["total_reward"]})

    df = pd.DataFrame(reward_data)
    sns.violinplot(data=df, x="Iteration", y="Reward", ax=ax)
    ax.set_title("Reward Distribution per Iteration")
    ax.grid(True, alpha=0.3)

    # Plot 4: GSM8K accuracy
    ax = axes[1, 1]
    accuracies = []
    for iter_num in iterations:
        metrics = data[iter_num]["metrics"]
        if "eval" in metrics and "accuracy" in metrics["eval"]:
            accuracies.append(metrics["eval"]["accuracy"] * 100)
        else:
            accuracies.append(None)

    valid_iters = [i for i, a in zip(iterations, accuracies) if a is not None]
    valid_accs = [a for a in accuracies if a is not None]

    if valid_accs:
        ax.plot(valid_iters, valid_accs, "mo-", linewidth=2, markersize=8)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("GSM8K Test Accuracy")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Reward curves saved to {save_path}")


def plot_question_difficulty(data: dict, save_path: Path):
    """
    Plot question difficulty progression.

    Measures average step count per question and its distribution.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    iterations = sorted(data.keys())

    mean_steps = []
    step_distributions: dict = defaultdict(list)

    for iter_num in iterations:
        trajs = data[iter_num]["trajectories"]

        steps = []
        for t in trajs:
            breakdown = t["reward_breakdown"]
            if "solution_metrics" in breakdown:
                step_count = breakdown["solution_metrics"].get("steps_total", 0)
                steps.append(step_count)
                step_distributions[iter_num].append(step_count)

        mean_steps.append(np.mean(steps) if steps else 0)

    # Plot 1: Mean step count
    ax = axes[0]
    ax.plot(iterations, mean_steps, "b-", linewidth=2, marker="o")
    ax.axhline(y=2, color="r", linestyle="--", alpha=0.5, label="Min target (2)")
    ax.axhline(y=5, color="r", linestyle="--", alpha=0.5, label="Max target (5)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average Step Count")
    ax.set_title("Question Difficulty Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Step count distribution
    ax = axes[1]
    step_data = []
    for iter_num in iterations:
        for step_count in step_distributions[iter_num]:
            step_data.append(
                {"Iteration": iter_num, "Steps": min(step_count, 10)}
            )

    df = pd.DataFrame(step_data)
    sns.boxplot(data=df, x="Iteration", y="Steps", ax=ax)
    ax.set_title("Step Count Distribution")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Difficulty plot saved to {save_path}")


def generate_summary_report(data: dict, save_path: Path):
    """
    Generate text summary for hackathon submission.
    """
    iterations = sorted(data.keys())

    report = []
    report.append("=" * 80)
    report.append("PPO SELF-IMPROVEMENT TRAINING SUMMARY")
    report.append("=" * 80)
    report.append("")

    first_iter = iterations[0]
    last_iter = iterations[-1]

    first_reward = np.mean(
        [t["total_reward"] for t in data[first_iter]["trajectories"]]
    )
    last_reward = np.mean(
        [t["total_reward"] for t in data[last_iter]["trajectories"]]
    )

    report.append(f"Training Iterations: {len(iterations)}")
    report.append(f"Initial Mean Reward: {first_reward:.3f}")
    report.append(f"Final Mean Reward:   {last_reward:.3f}")
    report.append(
        f"Reward Improvement: {last_reward - first_reward:+.3f} "
        f"({(last_reward / first_reward - 1) * 100:+.1f}%)"
    )
    report.append("")

    first_acc = data[first_iter]["metrics"].get("eval", {}).get("accuracy")
    last_acc = data[last_iter]["metrics"].get("eval", {}).get("accuracy")

    if first_acc is not None and last_acc is not None:
        report.append(f"Initial GSM8K Accuracy: {first_acc * 100:.1f}%")
        report.append(f"Final GSM8K Accuracy:   {last_acc * 100:.1f}%")
        report.append(f"Accuracy Improvement:   {(last_acc - first_acc) * 100:+.1f}%")
        report.append("")

    report.append("SAMPLE GENERATED QUESTIONS (Final Iteration):")
    report.append("-" * 80)

    final_trajs = data[last_iter]["trajectories"]
    sorted_trajs = sorted(
        final_trajs, key=lambda x: x["total_reward"], reverse=True
    )

    for i, traj in enumerate(sorted_trajs[:5], 1):
        report.append(f"\n{i}. Reward: {traj['total_reward']:.3f}")
        report.append(f"   Question: {traj['generated_question'][:200]}...")
        report.append(f"   Solution: {traj['generated_solution'][:200]}...")

    report.append("\n" + "=" * 80)

    with open(save_path, "w") as f:
        f.write("\n".join(report))

    print(f"Summary report saved to {save_path}")
    print("\n".join(report))


def main():
    parser = argparse.ArgumentParser(description="Visualize PPO training results")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory with iteration results",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Where to save plots (default: output_dir/plots)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    save_dir = Path(args.save_dir) if args.save_dir else output_dir / "plots"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {output_dir}...")
    data = load_iteration_data(output_dir)
    print(f"Found {len(data)} iterations")

    plot_reward_curves(data, save_dir / "reward_curves.png")
    plot_question_difficulty(data, save_dir / "difficulty_progression.png")
    generate_summary_report(data, save_dir / "summary.txt")

    print(f"\nAll visualizations saved to {save_dir}")


if __name__ == "__main__":
    main()
