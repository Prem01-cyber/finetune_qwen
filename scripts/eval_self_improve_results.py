#!/usr/bin/env python3
"""
Evaluate and visualize self-improvement training results across iterations.

This script:
1. Loads evaluation results from all iterations
2. Compares performance metrics across iterations
3. Generates visualizations and reports

Usage:
    python scripts/eval_self_improve_results.py \
        --training-dir checkpoints/rl_iterations \
        --output reports/self_improve_analysis.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_iteration_summaries(training_dir: Path) -> list[dict[str, Any]]:
    """Load summary data from all iterations."""
    summaries = []
    
    # Find all iteration directories
    iter_dirs = sorted([d for d in training_dir.iterdir() if d.is_dir() and d.name.startswith("iteration_")])
    
    for iter_dir in iter_dirs:
        summary_file = iter_dir / "summary.json"
        if summary_file.exists():
            with summary_file.open() as f:
                summaries.append(json.load(f))
    
    return summaries


def load_training_summary(training_dir: Path) -> dict[str, Any] | None:
    """Load overall training summary."""
    summary_file = training_dir / "training_summary.json"
    if summary_file.exists():
        with summary_file.open() as f:
            return json.load(f)
    return None


def generate_markdown_report(
    summaries: list[dict[str, Any]],
    training_summary: dict[str, Any] | None,
    output_path: Path,
) -> None:
    """Generate a markdown report with analysis."""
    
    lines = []
    lines.append("# Self-Improvement RL Training Analysis")
    lines.append("")
    lines.append(f"**Training Directory**: {training_summary.get('output_dir', 'N/A') if training_summary else 'N/A'}")
    lines.append(f"**Initial Model**: {training_summary.get('initial_adapter', 'N/A') if training_summary else 'N/A'}")
    lines.append(f"**Iterations Completed**: {len(summaries)}")
    lines.append("")
    
    # Configuration
    if training_summary and "config" in training_summary:
        lines.append("## Training Configuration")
        lines.append("")
        config = training_summary["config"]
        lines.append(f"- Samples per iteration: {config.get('samples_per_iteration')}")
        lines.append(f"- Preference pairs per iteration: {config.get('pairs_per_iteration')}")
        lines.append(f"- Learning rate: {config.get('learning_rate')}")
        lines.append(f"- Beta (KL penalty): {config.get('beta')}")
        lines.append(f"- Temperature: {config.get('temperature')}")
        lines.append("")
    
    # Metrics table
    lines.append("## Performance Across Iterations")
    lines.append("")
    lines.append("| Iteration | Accuracy | Format Rate | Improvement |")
    lines.append("|-----------|----------|-------------|-------------|")
    
    prev_acc = None
    for summary in summaries:
        metrics = summary.get("eval_metrics", {})
        acc = metrics.get("accuracy", 0.0)
        fmt = metrics.get("format_rate", 0.0)
        
        improvement = ""
        if prev_acc is not None:
            diff = acc - prev_acc
            if diff > 0:
                improvement = f"+{diff:.2%}"
            elif diff < 0:
                improvement = f"{diff:.2%}"
            else:
                improvement = "—"
        else:
            improvement = "—"
        
        lines.append(f"| {summary['iteration']} | {acc:.2%} | {fmt:.2%} | {improvement} |")
        prev_acc = acc
    
    lines.append("")
    
    # Best model
    if summaries:
        best_iter = max(summaries, key=lambda s: s.get("eval_metrics", {}).get("accuracy", 0.0))
        lines.append("## Best Model")
        lines.append("")
        lines.append(f"- **Iteration**: {best_iter['iteration']}")
        lines.append(f"- **Accuracy**: {best_iter['eval_metrics']['accuracy']:.2%}")
        lines.append(f"- **Format Rate**: {best_iter['eval_metrics']['format_rate']:.2%}")
        lines.append(f"- **Model Path**: `{best_iter['output_adapter']}`")
        lines.append("")
    
    # Final model
    if summaries:
        final = summaries[-1]
        lines.append("## Final Model")
        lines.append("")
        lines.append(f"- **Iteration**: {final['iteration']}")
        lines.append(f"- **Accuracy**: {final['eval_metrics']['accuracy']:.2%}")
        lines.append(f"- **Format Rate**: {final['eval_metrics']['format_rate']:.2%}")
        lines.append(f"- **Model Path**: `{final['output_adapter']}`")
        lines.append("")
    
    # Analysis
    lines.append("## Analysis")
    lines.append("")
    
    if len(summaries) >= 2:
        initial_acc = summaries[0]["eval_metrics"]["accuracy"]
        final_acc = summaries[-1]["eval_metrics"]["accuracy"]
        improvement = final_acc - initial_acc
        
        lines.append(f"### Overall Improvement")
        lines.append("")
        lines.append(f"- Initial Accuracy: {initial_acc:.2%}")
        lines.append(f"- Final Accuracy: {final_acc:.2%}")
        lines.append(f"- Absolute Improvement: {improvement:+.2%}")
        lines.append(f"- Relative Improvement: {(improvement/initial_acc)*100:+.1f}%")
        lines.append("")
        
        # Trajectory analysis
        accuracies = [s["eval_metrics"]["accuracy"] for s in summaries]
        
        if len(accuracies) >= 3:
            lines.append("### Training Trajectory")
            lines.append("")
            
            # Check for consistent improvement
            improvements = [accuracies[i+1] - accuracies[i] for i in range(len(accuracies)-1)]
            positive_improvements = sum(1 for imp in improvements if imp > 0)
            
            if positive_improvements == len(improvements):
                lines.append("- ✅ Consistent improvement across all iterations")
            elif positive_improvements >= len(improvements) * 0.7:
                lines.append("- ⚠️  Mostly improving with some fluctuations")
            else:
                lines.append("- ❌ Unstable training with frequent performance drops")
            
            lines.append("")
            
            # Peak performance
            peak_acc = max(accuracies)
            peak_iter = accuracies.index(peak_acc) + 1
            
            lines.append(f"- Peak performance: {peak_acc:.2%} at iteration {peak_iter}")
            
            if peak_iter < len(summaries):
                lines.append(f"- Warning: Peak was not at final iteration (possible overfitting)")
            
            lines.append("")
    
    # Recommendations
    lines.append("## Recommendations")
    lines.append("")
    
    if summaries:
        final_acc = summaries[-1]["eval_metrics"]["accuracy"]
        
        if final_acc < 0.50:
            lines.append("- ❌ **Low accuracy**: Consider increasing training data quality or model capacity")
        elif final_acc < 0.70:
            lines.append("- ⚠️  **Moderate accuracy**: Model shows promise but needs more iterations or tuning")
        else:
            lines.append("- ✅ **Good accuracy**: Model is performing well")
        
        lines.append("")
        
        # Check for overfitting
        if len(summaries) >= 3:
            accuracies = [s["eval_metrics"]["accuracy"] for s in summaries]
            if max(accuracies[:-1]) > accuracies[-1] + 0.02:
                lines.append("- ⚠️  Possible overfitting detected (best model not at end)")
                lines.append("- Recommendation: Use best checkpoint instead of final")
        
        lines.append("")
    
    # ASCII plot (simple text visualization)
    lines.append("## Accuracy Progression")
    lines.append("")
    lines.append("```")
    
    if summaries:
        accuracies = [s["eval_metrics"]["accuracy"] for s in summaries]
        min_acc = min(accuracies)
        max_acc = max(accuracies)
        range_acc = max_acc - min_acc if max_acc > min_acc else 0.1
        
        # Create simple ASCII plot
        height = 10
        width = len(summaries)
        
        for row in range(height, -1, -1):
            threshold = min_acc + (row / height) * range_acc
            line_parts = []
            
            for i, acc in enumerate(accuracies):
                if acc >= threshold:
                    line_parts.append("█")
                else:
                    line_parts.append(" ")
            
            # Add axis label
            label = f"{threshold:.1%}".rjust(6)
            lines.append(f"{label} | {''.join(line_parts)}")
        
        # X-axis
        lines.append("       " + "-" * width)
        x_labels = "       " + "".join(str(i+1) for i in range(width))
        lines.append(x_labels)
    
    lines.append("```")
    lines.append("")
    
    # Detailed iteration data
    lines.append("## Detailed Iteration Data")
    lines.append("")
    
    for summary in summaries:
        iter_num = summary["iteration"]
        metrics = summary.get("eval_metrics", {})
        
        lines.append(f"### Iteration {iter_num}")
        lines.append("")
        lines.append(f"- Accuracy: {metrics.get('accuracy', 0.0):.4f}")
        lines.append(f"- Format Rate: {metrics.get('format_rate', 0.0):.4f}")
        lines.append(f"- Samples Generated: {summary.get('num_samples', 'N/A')}")
        lines.append(f"- Preference Pairs: {summary.get('num_pairs', 'N/A')}")
        lines.append(f"- Model: `{summary.get('output_adapter', 'N/A')}`")
        lines.append("")
    
    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"Report written to: {output_path}")


def generate_json_summary(
    summaries: list[dict[str, Any]],
    training_summary: dict[str, Any] | None,
    output_path: Path,
) -> None:
    """Generate JSON summary for programmatic access."""
    
    accuracies = [s["eval_metrics"]["accuracy"] for s in summaries]
    format_rates = [s["eval_metrics"]["format_rate"] for s in summaries]
    
    data = {
        "training_dir": str(training_summary.get("output_dir")) if training_summary else None,
        "iterations": len(summaries),
        "metrics": {
            "accuracy": {
                "initial": accuracies[0] if accuracies else None,
                "final": accuracies[-1] if accuracies else None,
                "best": max(accuracies) if accuracies else None,
                "progression": accuracies,
            },
            "format_rate": {
                "initial": format_rates[0] if format_rates else None,
                "final": format_rates[-1] if format_rates else None,
                "best": max(format_rates) if format_rates else None,
                "progression": format_rates,
            },
        },
        "best_iteration": summaries[accuracies.index(max(accuracies))]["iteration"] if accuracies else None,
        "iterations_data": summaries,
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    print(f"JSON summary written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze self-improvement training results"
    )
    parser.add_argument(
        "--training-dir",
        type=str,
        required=True,
        help="Directory containing iteration subdirectories"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/self_improve_analysis.md",
        help="Output markdown report path"
    )
    parser.add_argument(
        "--json-output",
        type=str,
        help="Optional JSON output path"
    )
    
    args = parser.parse_args()
    
    training_dir = Path(args.training_dir)
    if not training_dir.exists():
        raise SystemExit(f"Training directory not found: {training_dir}")
    
    # Load data
    print(f"Loading results from {training_dir}...")
    summaries = load_iteration_summaries(training_dir)
    
    if not summaries:
        raise SystemExit("No iteration summaries found in training directory")
    
    print(f"  Found {len(summaries)} iterations")
    
    training_summary = load_training_summary(training_dir)
    
    # Generate markdown report
    output_path = Path(args.output)
    print(f"\nGenerating markdown report...")
    generate_markdown_report(summaries, training_summary, output_path)
    
    # Generate JSON summary if requested
    if args.json_output:
        json_path = Path(args.json_output)
        print(f"\nGenerating JSON summary...")
        generate_json_summary(summaries, training_summary, json_path)
    
    print("\n" + "=" * 60)
    print("Analysis Complete")
    print("=" * 60)
    
    # Print quick summary to console
    if summaries:
        accuracies = [s["eval_metrics"]["accuracy"] for s in summaries]
        print(f"\nQuick Summary:")
        print(f"  Iterations: {len(summaries)}")
        print(f"  Initial Accuracy: {accuracies[0]:.2%}")
        print(f"  Final Accuracy: {accuracies[-1]:.2%}")
        print(f"  Best Accuracy: {max(accuracies):.2%}")
        print(f"  Improvement: {accuracies[-1] - accuracies[0]:+.2%}")


if __name__ == "__main__":
    main()
