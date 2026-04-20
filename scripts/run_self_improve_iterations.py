#!/usr/bin/env python3
"""
Run multiple iterations of self-improvement RL loop.

This script orchestrates the full RL training loop:
1. Self-play: Generate questions and solutions
2. Reward calculation: Score trajectories
3. DPO training: Update model based on preferences
4. Evaluation: Check performance on GSM8K
5. Repeat until convergence or max iterations

Usage:
    python scripts/run_self_improve_iterations.py \
        --initial-adapter checkpoints/dual_task_v1 \
        --output-dir checkpoints/rl_iterations \
        --max-iterations 5 \
        --samples-per-iteration 1000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_command(cmd: list[str], description: str) -> None:
    """Run a subprocess command with error handling."""
    print(f"\n{'=' * 60}")
    print(f"{description}")
    print(f"{'=' * 60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


def evaluate_model(
    adapter_path: Path,
    eval_data: Path,
    max_samples: int,
) -> dict[str, float]:
    """
    Evaluate model on GSM8K test set.
    
    Returns metrics dictionary.
    """
    print(f"\nEvaluating model: {adapter_path}")
    
    output_json = adapter_path / "eval_results.json"
    
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "eval_dual_task.py"),
        "--adapter", str(adapter_path),
        "--solution-data", str(eval_data),
        "--skip-question-eval",
        "--max-solution-samples", str(max_samples),
        "--output-json", str(output_json),
    ]
    
    run_command(cmd, f"Evaluating iteration model")
    
    # Load results
    if output_json.exists():
        with output_json.open() as f:
            results = json.load(f)
            solution_summary = results.get("solution_generation", {}).get("summary", {})
            return {
                "accuracy": solution_summary.get("exact_match_accuracy", 0.0),
                "format_rate": solution_summary.get("format_compliance_rate", 0.0),
            }
    
    return {"accuracy": 0.0, "format_rate": 0.0}


def run_iteration(
    iteration: int,
    input_adapter: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[Path, dict[str, float]]:
    """
    Run one iteration of self-improvement.
    
    Returns (new_adapter_path, eval_metrics)
    """
    print(f"\n{'#' * 60}")
    print(f"# ITERATION {iteration}")
    print(f"{'#' * 60}")
    
    iter_dir = output_dir / f"iteration_{iteration:03d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    
    # Paths
    trajectories_file = iter_dir / "trajectories.jsonl"
    new_adapter = iter_dir / "model"
    
    # Step 1: Self-play generation
    cmd_selfplay = [
        sys.executable,
        str(ROOT / "scripts" / "self_play_generate.py"),
        "--adapter", str(input_adapter),
        "--output", str(trajectories_file),
        "--num-samples", str(args.samples_per_iteration),
        "--iteration", str(iteration),
        "--temperature", str(args.temperature),
    ]
    
    if args.reference_questions:
        cmd_selfplay.extend(["--reference-questions", str(args.reference_questions)])
    
    run_command(cmd_selfplay, f"Self-play generation (iteration {iteration})")
    
    # Step 2: DPO training
    cmd_dpo = [
        sys.executable,
        str(ROOT / "scripts" / "rl_self_improve.py"),
        "--base-adapter", str(input_adapter),
        "--trajectories", str(trajectories_file),
        "--output", str(new_adapter),
        "--num-pairs", str(args.pairs_per_iteration),
        "--learning-rate", str(args.learning_rate),
        "--beta", str(args.beta),
        "--epochs", str(args.dpo_epochs),
    ]
    
    run_command(cmd_dpo, f"DPO training (iteration {iteration})")
    
    # Step 3: Evaluate
    if args.eval_data:
        eval_metrics = evaluate_model(
            adapter_path=new_adapter,
            eval_data=Path(args.eval_data),
            max_samples=args.eval_samples,
        )
    else:
        eval_metrics = {"accuracy": 0.0, "format_rate": 0.0}
    
    # Save iteration summary
    summary = {
        "iteration": iteration,
        "input_adapter": str(input_adapter),
        "output_adapter": str(new_adapter),
        "trajectories": str(trajectories_file),
        "num_samples": args.samples_per_iteration,
        "num_pairs": args.pairs_per_iteration,
        "eval_metrics": eval_metrics,
    }
    
    summary_file = iter_dir / "summary.json"
    with summary_file.open("w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"Iteration {iteration} Summary")
    print(f"{'=' * 60}")
    print(f"  Accuracy: {eval_metrics['accuracy']:.2%}")
    print(f"  Format Rate: {eval_metrics['format_rate']:.2%}")
    print(f"  Model saved to: {new_adapter}")
    
    return new_adapter, eval_metrics


def check_convergence(
    history: list[dict[str, float]],
    patience: int,
    min_improvement: float,
) -> tuple[bool, str]:
    """
    Check if training should stop.
    
    Returns (should_stop, reason)
    """
    if len(history) < 2:
        return False, ""
    
    current = history[-1]
    best = max(history[:-1], key=lambda x: x["accuracy"])
    
    # Check for degradation
    if current["accuracy"] < best["accuracy"] - 0.02:
        return True, f"Performance degraded: {current['accuracy']:.2%} < {best['accuracy']:.2%}"
    
    # Check for stagnation
    if len(history) >= patience:
        recent = history[-patience:]
        improvements = [recent[i+1]["accuracy"] - recent[i]["accuracy"] 
                       for i in range(len(recent) - 1)]
        
        if all(imp < min_improvement for imp in improvements):
            return True, f"No significant improvement in last {patience} iterations"
    
    return False, ""


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple iterations of RL self-improvement"
    )
    
    # Paths
    parser.add_argument(
        "--initial-adapter",
        type=str,
        required=True,
        help="Initial dual-task model to improve"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for all iterations"
    )
    parser.add_argument(
        "--reference-questions",
        type=str,
        help="Path to reference questions for novelty scoring"
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        help="Path to evaluation data (e.g., GSM8K test split)"
    )
    
    # Iteration settings
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum number of iterations"
    )
    parser.add_argument(
        "--samples-per-iteration",
        type=int,
        default=1000,
        help="Number of self-play samples per iteration"
    )
    parser.add_argument(
        "--pairs-per-iteration",
        type=int,
        default=500,
        help="Number of preference pairs for DPO per iteration"
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=100,
        help="Number of samples for evaluation"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature for self-play"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-7,
        help="DPO learning rate"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO KL penalty strength"
    )
    parser.add_argument(
        "--dpo-epochs",
        type=float,
        default=1.0,
        help="DPO training epochs per iteration"
    )
    
    # Convergence criteria
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Number of iterations without improvement before stopping"
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.005,
        help="Minimum accuracy improvement to continue (0.5%)"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    initial_adapter = Path(args.initial_adapter)
    if not initial_adapter.exists():
        raise SystemExit(f"Initial adapter not found: {initial_adapter}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run iterations
    current_adapter = initial_adapter
    eval_history = []
    
    print(f"\n{'#' * 60}")
    print(f"# RL Self-Improvement Loop")
    print(f"{'#' * 60}")
    print(f"Initial adapter: {initial_adapter}")
    print(f"Output directory: {output_dir}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Samples per iteration: {args.samples_per_iteration}")
    
    for iteration in range(1, args.max_iterations + 1):
        try:
            new_adapter, eval_metrics = run_iteration(
                iteration=iteration,
                input_adapter=current_adapter,
                output_dir=output_dir,
                args=args,
            )
            
            eval_history.append(eval_metrics)
            current_adapter = new_adapter
            
            # Check convergence
            should_stop, reason = check_convergence(
                history=eval_history,
                patience=args.patience,
                min_improvement=args.min_improvement,
            )
            
            if should_stop:
                print(f"\n{'=' * 60}")
                print(f"Early stopping triggered: {reason}")
                print(f"{'=' * 60}")
                break
        
        except Exception as e:
            print(f"\n{'!' * 60}")
            print(f"Error in iteration {iteration}: {e}")
            print(f"{'!' * 60}")
            break
    
    # Final summary
    print(f"\n{'#' * 60}")
    print(f"# Training Complete")
    print(f"{'#' * 60}")
    print(f"\nCompleted {len(eval_history)} iterations")
    
    if eval_history:
        print(f"\nAccuracy progression:")
        for i, metrics in enumerate(eval_history, 1):
            print(f"  Iteration {i}: {metrics['accuracy']:.2%}")
        
        best_idx = max(range(len(eval_history)), key=lambda i: eval_history[i]["accuracy"])
        print(f"\nBest model: Iteration {best_idx + 1} ({eval_history[best_idx]['accuracy']:.2%})")
        print(f"Final model: Iteration {len(eval_history)} ({eval_history[-1]['accuracy']:.2%})")
        
        # Save overall summary
        summary = {
            "initial_adapter": str(initial_adapter),
            "output_dir": str(output_dir),
            "iterations_completed": len(eval_history),
            "max_iterations": args.max_iterations,
            "eval_history": eval_history,
            "best_iteration": best_idx + 1,
            "best_accuracy": eval_history[best_idx]["accuracy"],
            "final_accuracy": eval_history[-1]["accuracy"],
            "config": {
                "samples_per_iteration": args.samples_per_iteration,
                "pairs_per_iteration": args.pairs_per_iteration,
                "learning_rate": args.learning_rate,
                "beta": args.beta,
                "temperature": args.temperature,
            },
        }
        
        summary_file = output_dir / "training_summary.json"
        with summary_file.open("w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nTraining summary saved to: {summary_file}")
        print(f"Best model available at: {output_dir / f'iteration_{best_idx+1:03d}' / 'model'}")


if __name__ == "__main__":
    main()
