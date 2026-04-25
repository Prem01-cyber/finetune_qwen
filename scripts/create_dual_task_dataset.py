#!/usr/bin/env python3
"""
Create dual-task training dataset by mixing question-generation and solution-generation examples.

This script:
1. Loads existing solution data (GSM8K format)
2. Loads question-generation data (synthetic)
3. Adds task prefixes to distinguish tasks
4. Mixes datasets according to specified ratio
5. Shuffles and splits into train/validation

Usage:
    python scripts/create_dual_task_dataset.py \
        --solution-data data/sft/gsm8k_sft.jsonl \
        --question-data data/sft/question_generation.jsonl \
        --output-train data/sft/dual_task_train.jsonl \
        --output-val data/sft/dual_task_val.jsonl \
        --mix-ratio 0.8 \
        --val-split 0.1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config.prompts import SOLVE_TASK_PREFIX, GENERATE_TASK_PREFIX


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL file into list of records."""
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def add_solve_prefix(record: dict[str, Any]) -> dict[str, Any]:
    """
    Add 'Solve Problem' task prefix to user message.
    
    This signals the model to generate a step-by-step solution.
    """
    modified = record.copy()
    modified["messages"] = []
    
    for msg in record["messages"]:
        new_msg = msg.copy()
        if msg["role"] == "user":
            # Add task prefix to user content
            content = msg["content"]
            if not content.startswith(SOLVE_TASK_PREFIX):
                new_msg["content"] = SOLVE_TASK_PREFIX + content
        modified["messages"].append(new_msg)
    
    # Update text field if present
    if "text" in modified:
        # Find and update user section
        text = modified["text"]
        if "<|user|>" in text:
            parts = text.split("<|user|>")
            if len(parts) > 1:
                user_part = parts[1]
                if not user_part.strip().startswith(SOLVE_TASK_PREFIX):
                    parts[1] = f"\n{SOLVE_TASK_PREFIX}" + user_part
                    modified["text"] = "<|user|>".join(parts)
    
    # Mark as solve task
    modified["task_type"] = "solve"
    
    return modified


def verify_question_prefix(record: dict[str, Any]) -> dict[str, Any]:
    """
    Verify question generation record has proper prefix.
    
    Should already have it from generation script, but double-check.
    """
    modified = record.copy()
    modified["messages"] = []
    
    for msg in record["messages"]:
        new_msg = msg.copy()
        if msg["role"] == "user":
            content = msg["content"]
            if not content.startswith(GENERATE_TASK_PREFIX):
                new_msg["content"] = GENERATE_TASK_PREFIX + content
        modified["messages"].append(new_msg)
    
    # Update text field if present
    if "text" in modified:
        text = modified["text"]
        if "<|user|>" in text:
            parts = text.split("<|user|>")
            if len(parts) > 1:
                user_part = parts[1]
                if not user_part.strip().startswith(GENERATE_TASK_PREFIX):
                    parts[1] = f"\n{GENERATE_TASK_PREFIX}" + user_part
                    modified["text"] = "<|user|>".join(parts)
    
    # Mark as question generation task
    modified["task_type"] = "generate"
    
    return modified


def sample_with_ratio(
    solution_records: list[dict[str, Any]],
    question_records: list[dict[str, Any]],
    mix_ratio: float,
    target_total: int | None = None,
) -> list[dict[str, Any]]:
    """
    Sample and mix datasets according to specified ratio.
    
    Args:
        solution_records: Solution examples
        question_records: Question generation examples
        mix_ratio: Fraction of solutions in final dataset (0.8 = 80% solutions, 20% questions)
        target_total: Target total examples (None = use all available data)
    
    Returns:
        Mixed dataset
    """
    n_solutions = len(solution_records)
    n_questions = len(question_records)
    
    if target_total is None:
        # Use all available data
        target_total = n_solutions + n_questions
    
    # Calculate target counts
    n_sol_target = int(target_total * mix_ratio)
    n_q_target = target_total - n_sol_target
    
    # Check availability
    if n_sol_target > n_solutions:
        print(f"Warning: Requested {n_sol_target} solutions but only {n_solutions} available.")
        n_sol_target = n_solutions
    
    if n_q_target > n_questions:
        print(f"Warning: Requested {n_q_target} questions but only {n_questions} available.")
        n_q_target = n_questions
    
    # Sample
    selected_solutions = random.sample(solution_records, n_sol_target)
    selected_questions = random.sample(question_records, n_q_target)
    
    print(f"Sampled {n_sol_target} solutions and {n_q_target} questions")
    print(f"Actual ratio: {n_sol_target/(n_sol_target+n_q_target):.2%} solutions, "
          f"{n_q_target/(n_sol_target+n_q_target):.2%} questions")
    
    return selected_solutions + selected_questions


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    """Write records to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create dual-task training dataset from solution and question-generation examples."
    )
    parser.add_argument(
        "--solution-data",
        type=Path,
        required=True,
        help="Path to solution training data (GSM8K format)",
    )
    parser.add_argument(
        "--question-data",
        type=Path,
        required=True,
        help="Path to question-generation training data",
    )
    parser.add_argument(
        "--output-train",
        type=Path,
        required=True,
        help="Output path for training split",
    )
    parser.add_argument(
        "--output-val",
        type=Path,
        required=True,
        help="Output path for validation split",
    )
    parser.add_argument(
        "--mix-ratio",
        type=float,
        default=0.8,
        help="Fraction of solutions in mixed dataset (default: 0.8 = 80%% solutions)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation (default: 0.1 = 10%%)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--max-total",
        type=int,
        default=None,
        help="Maximum total examples to include (None = use all available)",
    )
    args = parser.parse_args()
    
    # Validate inputs
    if not args.solution_data.exists():
        raise SystemExit(f"Error: Solution data not found at {args.solution_data}")
    if not args.question_data.exists():
        raise SystemExit(f"Error: Question data not found at {args.question_data}")
    
    if not (0 < args.mix_ratio < 1):
        raise SystemExit("Error: --mix-ratio must be between 0 and 1")
    if not (0 < args.val_split < 1):
        raise SystemExit("Error: --val-split must be between 0 and 1")
    
    # Set random seed
    random.seed(args.seed)
    
    print("=" * 60)
    print("Dual-Task Dataset Creation")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    print(f"   Solution data: {args.solution_data}")
    solution_records = load_jsonl(args.solution_data)
    print(f"   Loaded {len(solution_records)} solution examples")
    
    print(f"   Question data: {args.question_data}")
    question_records = load_jsonl(args.question_data)
    print(f"   Loaded {len(question_records)} question-generation examples")
    
    # Add task prefixes
    print("\n2. Adding task prefixes...")
    print("   Adding 'Solve Problem' prefix to solution examples...")
    solution_records = [add_solve_prefix(r) for r in solution_records]
    
    print("   Verifying 'Generate Question' prefix on question examples...")
    question_records = [verify_question_prefix(r) for r in question_records]
    
    # Mix datasets
    print(f"\n3. Mixing datasets (ratio: {args.mix_ratio:.0%} solutions, {1-args.mix_ratio:.0%} questions)...")
    mixed_records = sample_with_ratio(
        solution_records=solution_records,
        question_records=question_records,
        mix_ratio=args.mix_ratio,
        target_total=args.max_total,
    )
    
    # Shuffle
    print(f"\n4. Shuffling {len(mixed_records)} total examples...")
    random.shuffle(mixed_records)
    
    # Split train/val
    n_val = int(len(mixed_records) * args.val_split)
    n_train = len(mixed_records) - n_val
    
    train_records = mixed_records[:n_train]
    val_records = mixed_records[n_train:]
    
    print(f"\n5. Splitting data:")
    print(f"   Training: {len(train_records)} examples ({len(train_records)/len(mixed_records):.1%})")
    print(f"   Validation: {len(val_records)} examples ({len(val_records)/len(mixed_records):.1%})")
    
    # Verify split composition
    train_solve = sum(1 for r in train_records if r.get("task_type") == "solve")
    train_gen = sum(1 for r in train_records if r.get("task_type") == "generate")
    val_solve = sum(1 for r in val_records if r.get("task_type") == "solve")
    val_gen = sum(1 for r in val_records if r.get("task_type") == "generate")
    
    print(f"\n   Train composition:")
    print(f"     Solve: {train_solve} ({train_solve/len(train_records):.1%})")
    print(f"     Generate: {train_gen} ({train_gen/len(train_records):.1%})")
    
    print(f"   Val composition:")
    print(f"     Solve: {val_solve} ({val_solve/len(val_records):.1%})")
    print(f"     Generate: {val_gen} ({val_gen/len(val_records):.1%})")
    
    # Write outputs
    print(f"\n6. Writing output files...")
    print(f"   Training data: {args.output_train}")
    write_jsonl(train_records, args.output_train)
    
    print(f"   Validation data: {args.output_val}")
    write_jsonl(val_records, args.output_val)
    
    print("\n" + "=" * 60)
    print("Dual-task dataset creation complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  Train: {args.output_train} ({len(train_records)} examples)")
    print(f"  Val:   {args.output_val} ({len(val_records)} examples)")
    print(f"\nNext step: Train dual-task model using these files")


if __name__ == "__main__":
    main()
