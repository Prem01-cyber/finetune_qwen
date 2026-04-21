"""
Visualize Consensus Training Results

Creates plots showing:
1. Consensus rate progression over iterations
2. Answer diversity over iterations
3. SymPy vs Consensus scores correlation
4. Reward progression (combined, SymPy-only, consensus-only)
5. Side-by-side examples (iteration 1 vs iteration 10)
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_iteration_data(checkpoint_dir: Path) -> Dict[int, Dict[str, Any]]:
    """
    Load data from all iterations.
    
    Args:
        checkpoint_dir: Path to checkpoint directory (e.g., checkpoints/ppo_training_consensus)
    
    Returns:
        Dict mapping iteration number to metrics and trajectories
    """
    data = {}
    
    # Find all iteration directories
    iteration_dirs = sorted(checkpoint_dir.glob("iteration_*"))
    
    for iter_dir in iteration_dirs:
        # Extract iteration number from directory name
        iter_num = int(iter_dir.name.split("_")[1])
        
        # Load metrics
        metrics_file = iter_dir / "metrics.json"
        if not metrics_file.exists():
            continue
        
        with open(metrics_file) as f:
            metrics = json.load(f)
        
        # Load trajectories
        trajectories_file = iter_dir / "trajectories.jsonl"
        trajectories = []
        
        if trajectories_file.exists():
            with open(trajectories_file) as f:
                for line in f:
                    if line.strip():
                        trajectories.append(json.loads(line))
        
        data[iter_num] = {
            "metrics": metrics,
            "trajectories": trajectories,
        }
    
    return data


def plot_consensus_progression(data: Dict[int, Dict[str, Any]], output_path: Path):
    """
    Create 4-panel plot showing consensus metrics over time.
    
    Panels:
    1. Consensus rate over iterations
    2. Answer diversity over iterations
    3. SymPy vs Consensus scores scatter
    4. Reward progression
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Consensus Verification Training Progression", fontsize=16, fontweight='bold')
    
    # Extract data for plotting
    iterations = sorted(data.keys())
    
    consensus_rates = []
    answer_diversities = []
    mean_rewards = []
    consensus_scores = []
    sympy_scores = []
    
    for iter_num in iterations:
        metrics = data[iter_num]["metrics"]
        
        # Consensus metrics
        if "consensus" in metrics:
            consensus_rates.append(metrics["consensus"]["consensus_rate"])
            answer_diversities.append(metrics["consensus"]["mean_answer_diversity"])
            consensus_scores.append(metrics["consensus"]["mean_consensus_score"])
            sympy_scores.append(metrics["consensus"]["mean_sympy_score"])
        
        # Buffer metrics
        if "buffer" in metrics:
            mean_rewards.append(metrics["buffer"]["mean_episode_reward"])
    
    # Panel 1: Consensus rate over iterations
    ax1 = axes[0, 0]
    if consensus_rates:
        ax1.plot(iterations, [r * 100 for r in consensus_rates], 
                marker='o', linewidth=2, markersize=6, color='#2ecc71')
        ax1.fill_between(iterations, 0, [r * 100 for r in consensus_rates], 
                         alpha=0.3, color='#2ecc71')
        ax1.axhline(y=80, color='r', linestyle='--', alpha=0.5, label='Target: 80%')
    ax1.set_xlabel("Iteration", fontsize=11)
    ax1.set_ylabel("Consensus Rate (%)", fontsize=11)
    ax1.set_title("1. Consensus Rate Progression", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Panel 2: Answer diversity over iterations
    ax2 = axes[0, 1]
    if answer_diversities:
        ax2.plot(iterations, answer_diversities, 
                marker='s', linewidth=2, markersize=6, color='#e74c3c')
        ax2.fill_between(iterations, 1, answer_diversities, 
                         alpha=0.3, color='#e74c3c')
        ax2.axhline(y=1.2, color='g', linestyle='--', alpha=0.5, label='Target: <1.2')
    ax2.set_xlabel("Iteration", fontsize=11)
    ax2.set_ylabel("Mean Unique Answers per Question", fontsize=11)
    ax2.set_title("2. Answer Diversity (Lower = Better Agreement)", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Panel 3: SymPy vs Consensus scores scatter
    ax3 = axes[1, 0]
    if sympy_scores and consensus_scores:
        # Scatter plot with color by iteration
        scatter = ax3.scatter(sympy_scores, consensus_scores, 
                             c=iterations, cmap='viridis', 
                             s=100, alpha=0.6, edgecolors='black')
        
        # Add diagonal line (perfect correlation)
        ax3.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect correlation')
        
        plt.colorbar(scatter, ax=ax3, label='Iteration')
        
        # Compute correlation
        if len(sympy_scores) > 1:
            corr = np.corrcoef(sympy_scores, consensus_scores)[0, 1]
            ax3.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                    transform=ax3.transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax3.set_xlabel("SymPy Score (Arithmetic)", fontsize=11)
    ax3.set_ylabel("Consensus Score (Semantics)", fontsize=11)
    ax3.set_title("3. SymPy vs Consensus Correlation", fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    
    # Panel 4: Reward progression
    ax4 = axes[1, 1]
    if mean_rewards:
        ax4.plot(iterations, mean_rewards, 
                marker='o', linewidth=2, markersize=6, color='#3498db',
                label='Combined Reward')
        
        if consensus_scores:
            # Scale consensus and sympy scores by their weights (0.4 each)
            weighted_consensus = [0.4 * s for s in consensus_scores]
            weighted_sympy = [0.4 * s for s in sympy_scores]
            
            ax4.plot(iterations, weighted_consensus, 
                    marker='^', linewidth=1.5, markersize=5, alpha=0.7,
                    color='#9b59b6', linestyle='--', label='Consensus (0.4×)')
            ax4.plot(iterations, weighted_sympy, 
                    marker='v', linewidth=1.5, markersize=5, alpha=0.7,
                    color='#e67e22', linestyle='--', label='SymPy (0.4×)')
    
    ax4.set_xlabel("Iteration", fontsize=11)
    ax4.set_ylabel("Reward", fontsize=11)
    ax4.set_title("4. Reward Progression Over Training", fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved consensus progression plot to {output_path}")
    plt.close()


def create_side_by_side_examples(
    data: Dict[int, Dict[str, Any]], 
    output_path: Path,
    early_iter: int = 1,
    late_iter: int = None,
):
    """
    Create visual comparison of early vs late iteration examples.
    
    Shows:
    - Early iteration: vague question, no consensus
    - Late iteration: clear question, strong consensus
    """
    if late_iter is None:
        late_iter = max(data.keys())
    
    if early_iter not in data or late_iter not in data:
        print(f"Warning: Iterations {early_iter} or {late_iter} not found")
        return
    
    # Find good examples (one with low consensus, one with high)
    early_trajs = data[early_iter]["trajectories"]
    late_trajs = data[late_iter]["trajectories"]
    
    # Find trajectory with low consensus in early iteration
    early_example = None
    for traj in early_trajs:
        consensus = traj.get("verification", {}).get("consensus", {})
        if not consensus.get("has_majority", False):
            early_example = traj
            break
    
    if early_example is None and early_trajs:
        early_example = early_trajs[0]
    
    # Find trajectory with high consensus in late iteration
    late_example = None
    for traj in late_trajs:
        consensus = traj.get("verification", {}).get("consensus", {})
        if (consensus.get("has_majority", False) and 
            consensus.get("consensus_strength", 0) > 0.8):
            late_example = traj
            break
    
    if late_example is None and late_trajs:
        late_example = late_trajs[0]
    
    if not early_example or not late_example:
        print("Warning: Could not find suitable examples")
        return
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    fig.suptitle("Training Progression: Early vs Late Iterations", 
                fontsize=16, fontweight='bold')
    
    for idx, (example, iteration, ax) in enumerate([
        (early_example, early_iter, axes[0]),
        (late_example, late_iter, axes[1])
    ]):
        ax.axis('off')
        
        # Extract data
        question = example.get("generated_question", "")[:150]
        primary_sol = example.get("primary_solution", "")[:200]
        alt_sols = example.get("alternative_solutions", [])
        consensus = example.get("verification", {}).get("consensus", {})
        reward = example.get("total_reward", 0)
        
        # Format text
        text_lines = [
            f"ITERATION {iteration}",
            "=" * 50,
            "",
            f"Question:",
            f"{question}...",
            "",
            f"Primary Solution:",
            f"{primary_sol}...",
            "",
            f"Alternative Solutions:",
            f"  Sol 2: {alt_sols[0][:100] if len(alt_sols) > 0 else 'N/A'}...",
            f"  Sol 3: {alt_sols[1][:100] if len(alt_sols) > 1 else 'N/A'}...",
            "",
            f"Consensus:",
            f"  Has Majority: {consensus.get('has_majority', False)}",
            f"  Strength: {consensus.get('consensus_strength', 0):.2f}",
            f"  Primary Matches: {consensus.get('primary_matches_majority', False)}",
            f"  Answer Diversity: {consensus.get('answer_diversity', 0)}",
            "",
            f"Total Reward: {reward:.3f}",
        ]
        
        # Color based on consensus
        has_consensus = consensus.get("has_majority", False)
        bg_color = '#d5f4e6' if has_consensus else '#fadbd8'
        
        # Add colored background
        ax.add_patch(mpatches.Rectangle((0, 0), 1, 1, 
                                       facecolor=bg_color, 
                                       transform=ax.transAxes, zorder=0))
        
        # Add text
        text = "\n".join(text_lines)
        ax.text(0.05, 0.95, text, 
               transform=ax.transAxes,
               fontsize=9,
               verticalalignment='top',
               fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='white', 
                        alpha=0.8, edgecolor='black', linewidth=2))
        
        # Add title
        title = f"{'Low' if not has_consensus else 'High'} Consensus"
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved side-by-side examples to {output_path}")
    plt.close()


def print_summary_statistics(data: Dict[int, Dict[str, Any]]):
    """Print summary statistics for all iterations."""
    print("\n" + "=" * 80)
    print("CONSENSUS TRAINING SUMMARY")
    print("=" * 80)
    
    for iter_num in sorted(data.keys()):
        metrics = data[iter_num]["metrics"]
        
        print(f"\nIteration {iter_num}:")
        print("-" * 40)
        
        if "buffer" in metrics:
            print(f"  Mean Reward: {metrics['buffer']['mean_episode_reward']:.3f}")
        
        if "consensus" in metrics:
            cons = metrics["consensus"]
            print(f"  Consensus Rate: {cons['consensus_rate']*100:.1f}%")
            print(f"  Answer Diversity: {cons['mean_answer_diversity']:.2f}")
            print(f"  Mean Consensus Score: {cons['mean_consensus_score']:.3f}")
            print(f"  Mean SymPy Score: {cons['mean_sympy_score']:.3f}")
        
        if "eval" in metrics and metrics["eval"]:
            print(f"  GSM8K Accuracy: {metrics['eval'].get('accuracy', 0)*100:.1f}%")
    
    # Print improvement
    if len(data) >= 2:
        first_iter = min(data.keys())
        last_iter = max(data.keys())
        
        first_reward = data[first_iter]["metrics"]["buffer"]["mean_episode_reward"]
        last_reward = data[last_iter]["metrics"]["buffer"]["mean_episode_reward"]
        
        print("\n" + "=" * 80)
        print(f"IMPROVEMENT: {first_reward:.3f} → {last_reward:.3f} "
              f"(+{last_reward - first_reward:.3f})")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize consensus training results"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/ppo_training_consensus",
        help="Path to checkpoint directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/ppo_training_consensus/plots",
        help="Directory to save plots",
    )
    
    args = parser.parse_args()
    
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {checkpoint_dir}...")
    data = load_iteration_data(checkpoint_dir)
    
    if not data:
        print("Error: No iteration data found!")
        return
    
    print(f"Loaded {len(data)} iterations")
    
    # Print summary statistics
    print_summary_statistics(data)
    
    # Create plots
    print("\nGenerating plots...")
    
    plot_consensus_progression(
        data, 
        output_dir / "consensus_progression.png"
    )
    
    create_side_by_side_examples(
        data,
        output_dir / "example_solutions.png",
        early_iter=1,
        late_iter=max(data.keys()),
    )
    
    print(f"\n✓ All plots saved to {output_dir}")


if __name__ == "__main__":
    main()
