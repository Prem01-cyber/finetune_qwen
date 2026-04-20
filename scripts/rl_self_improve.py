#!/usr/bin/env python3
"""
RL self-improvement using DPO (Direct Preference Optimization).

This script:
1. Loads self-play trajectories with reward scores
2. Creates preference pairs (best vs. worst completions)
3. Trains model with DPO to prefer high-reward outputs
4. Saves improved model checkpoint

Usage:
    python scripts/rl_self_improve.py \
        --base-adapter checkpoints/dual_task_v1 \
        --trajectories data/rl/self_play_iteration_001.jsonl \
        --output checkpoints/iteration_001 \
        --num-pairs 500
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if "HF_HUB_DISABLE_XET" not in os.environ:
    os.environ["HF_HUB_DISABLE_XET"] = "1"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

# Task prefixes
SOLVE_TASK_PREFIX = "### Task: Solve Problem\n"
GENERATE_TASK_PREFIX = "### Task: Generate Question\n"

# System prompts
SOLVER_SYSTEM_PROMPT = (
    "You are a step-by-step math solver. "
    "Solve the given problem one step at a time. "
    "Each step must be on its own line, starting with 'Step N:'. "
    "End with a line starting with 'Final Answer:'. "
    "Write every mathematical expression in Python/SymPy syntax "
    "so it can be verified programmatically."
)

GENERATOR_SYSTEM_PROMPT = (
    "You are a math problem generator. "
    "Generate grade-school level math word problems that require 2-5 steps to solve. "
    "Problems should involve realistic scenarios and use simple arithmetic, fractions, "
    "percentages, or basic algebra. "
    "Output ONLY the problem statement, no solutions or steps."
)


def load_trajectories(trajectory_path: Path) -> list[dict[str, Any]]:
    """Load self-play trajectories from JSONL."""
    trajectories = []
    with trajectory_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                trajectories.append(json.loads(line))
    return trajectories


def group_by_prompt(trajectories: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group trajectories by generation prompt for pairing."""
    groups = defaultdict(list)
    for traj in trajectories:
        prompt = traj.get("generation_prompt", "")
        groups[prompt].append(traj)
    return dict(groups)


def create_preference_pairs(
    trajectories: list[dict[str, Any]],
    num_pairs: int,
    min_score_diff: float = 0.1,
) -> list[dict[str, Any]]:
    """
    Create preference pairs from trajectories.
    
    For each prompt group, pairs the best and worst completions.
    
    Args:
        trajectories: List of trajectory records with rewards
        num_pairs: Target number of preference pairs
        min_score_diff: Minimum score difference to create a pair
    
    Returns:
        List of preference pair records for DPO training
    """
    print(f"\nCreating preference pairs from {len(trajectories)} trajectories...")
    
    # Group by generation prompt
    groups = group_by_prompt(trajectories)
    print(f"  Found {len(groups)} unique prompts")
    
    pairs = []
    
    for prompt, group in groups.items():
        if len(group) < 2:
            # Need at least 2 completions to create a pair
            continue
        
        # Sort by combined score
        sorted_group = sorted(group, key=lambda x: x.get("combined_score", 0.0), reverse=True)
        
        # Take best and worst
        best = sorted_group[0]
        worst = sorted_group[-1]
        
        score_diff = best["combined_score"] - worst["combined_score"]
        if score_diff < min_score_diff:
            # Scores too similar, skip this pair
            continue
        
        # Create preference pair
        # For question generation task
        chosen_question = best["generated_question"]
        rejected_question = worst["generated_question"]
        
        # For solution task (use the question as prompt, solution as completion)
        question_for_solve = best["generated_question"]
        chosen_solution = best["generated_solution"]
        rejected_solution = worst["generated_solution"]
        
        # Create two types of pairs: one for question generation, one for solution
        
        # Question generation pair
        pairs.append({
            "prompt": [
                {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
                {"role": "user", "content": f"{GENERATE_TASK_PREFIX}{prompt}"},
            ],
            "chosen": [
                {"role": "assistant", "content": chosen_question}
            ],
            "rejected": [
                {"role": "assistant", "content": rejected_question}
            ],
            "chosen_score": best["rewards"]["question"]["overall_score"],
            "rejected_score": worst["rewards"]["question"]["overall_score"],
        })
        
        # Solution generation pair
        user_content = (
            f"{SOLVE_TASK_PREFIX}"
            "Solve the following problem. Show your reasoning as numbered steps, "
            "then give the final numeric answer on the last line.\n\n"
            f"Problem:\n{question_for_solve}"
        )
        pairs.append({
            "prompt": [
                {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "chosen": [
                {"role": "assistant", "content": chosen_solution}
            ],
            "rejected": [
                {"role": "assistant", "content": rejected_solution}
            ],
            "chosen_score": best["rewards"]["solution"]["overall_score"],
            "rejected_score": worst["rewards"]["solution"]["overall_score"],
        })
        
        if len(pairs) >= num_pairs:
            break
    
    print(f"  Created {len(pairs)} preference pairs")
    print(f"  Avg score difference: {sum(p['chosen_score'] - p['rejected_score'] for p in pairs) / len(pairs):.3f}")
    
    return pairs[:num_pairs]


def prepare_dpo_dataset(pairs: list[dict[str, Any]], tokenizer: Any) -> Dataset:
    """
    Prepare dataset for DPO training.
    
    DPO requires specific format with 'prompt', 'chosen', and 'rejected' fields.
    """
    
    def format_messages(messages: list[dict]) -> str:
        """Format messages using chat template."""
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    
    # Prepare data for Dataset
    data = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
    }
    
    for pair in pairs:
        # Format prompt (system + user)
        prompt_text = format_messages(pair["prompt"])
        
        # Format chosen and rejected completions
        chosen_text = format_messages(pair["prompt"] + pair["chosen"])
        rejected_text = format_messages(pair["prompt"] + pair["rejected"])
        
        data["prompt"].append(prompt_text)
        data["chosen"].append(chosen_text)
        data["rejected"].append(rejected_text)
    
    return Dataset.from_dict(data)


def load_model_for_dpo(
    adapter_path: Path,
    base_model: str,
    compute_dtype_str: str,
) -> tuple[Any, Any, Any]:
    """
    Load model for DPO training.
    
    Returns (model, ref_model, tokenizer)
    """
    print(f"\nLoading model from {adapter_path} for DPO training...")
    
    compute_dtype = getattr(torch, compute_dtype_str)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # Check metadata
    meta_path = adapter_path / "pipeline_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        base_model = meta.get("base_model", base_model)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load base model
    print("  Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load trained adapter
    print("  Loading adapter...")
    model = PeftModel.from_pretrained(base, str(adapter_path))
    
    # Merge and unload for DPO training
    print("  Merging adapter with base model...")
    model = model.merge_and_unload()
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    
    # Create reference model (copy of current model for KL penalty)
    # In DPO, ref_model is used to compute KL divergence
    # For memory efficiency, we'll use the same model as reference
    ref_model = model
    
    print("Model prepared for DPO training.")
    
    return model, ref_model, tokenizer


def train_dpo(
    model: Any,
    ref_model: Any,
    tokenizer: Any,
    train_dataset: Dataset,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Train model with DPO."""
    print(f"\nStarting DPO training...")
    print(f"  Output directory: {output_dir}")
    print(f"  Training examples: {len(train_dataset)}")
    
    # DPO configuration
    dpo_config = DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        beta=args.beta,  # KL penalty strength
        max_length=args.max_seq_length,
        max_prompt_length=args.max_prompt_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=args.bf16 and torch.cuda.is_available(),
        fp16=args.fp16 and torch.cuda.is_available() and not args.bf16,
        gradient_checkpointing=True,
        report_to="none",
    )
    
    # Create DPO trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    
    # Train
    print("\nTraining...")
    trainer.train()
    
    # Save final model
    print("\nSaving model...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    # Save metadata
    with (output_dir / "pipeline_meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "pipeline_type": "dpo_rl",
                "base_adapter": str(args.base_adapter),
                "trajectories": str(args.trajectories),
                "num_pairs": len(train_dataset),
                "beta": args.beta,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
            },
            f,
            indent=2,
        )
    
    print(f"\nDPO training complete! Model saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="RL self-improvement with DPO")
    
    # Input/Output
    parser.add_argument(
        "--base-adapter",
        type=str,
        required=True,
        help="Path to base dual-task adapter to improve"
    )
    parser.add_argument(
        "--trajectories",
        type=str,
        required=True,
        help="Path to self-play trajectories JSONL"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for improved model"
    )
    
    # Data preparation
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=500,
        help="Number of preference pairs to create"
    )
    parser.add_argument(
        "--min-score-diff",
        type=float,
        default=0.1,
        help="Minimum score difference for preference pairs"
    )
    
    # Model
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-Math-7B-Instruct"
    )
    parser.add_argument(
        "--bnb-compute-dtype",
        type=str,
        default="bfloat16"
    )
    
    # DPO training hyperparameters
    parser.add_argument(
        "--epochs",
        type=float,
        default=1.0,
        help="Training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=8
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-7,
        help="Learning rate (lower than SFT)"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO KL penalty strength"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=1024
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True
    )
    parser.add_argument(
        "--no-bf16",
        dest="bf16",
        action="store_false"
    )
    parser.add_argument(
        "--fp16",
        action="store_true"
    )
    
    args = parser.parse_args()
    
    # Load trajectories
    trajectory_path = Path(args.trajectories)
    if not trajectory_path.exists():
        raise SystemExit(f"Trajectories file not found: {trajectory_path}")
    
    trajectories = load_trajectories(trajectory_path)
    print(f"Loaded {len(trajectories)} trajectories")
    
    # Create preference pairs
    pairs = create_preference_pairs(
        trajectories=trajectories,
        num_pairs=args.num_pairs,
        min_score_diff=args.min_score_diff,
    )
    
    if len(pairs) == 0:
        raise SystemExit("Error: No valid preference pairs created. Check trajectory quality.")
    
    # Load model
    adapter_path = Path(args.base_adapter)
    model, ref_model, tokenizer = load_model_for_dpo(
        adapter_path=adapter_path,
        base_model=args.base_model,
        compute_dtype_str=args.bnb_compute_dtype,
    )
    
    # Prepare DPO dataset
    print("\nPreparing DPO dataset...")
    train_dataset = prepare_dpo_dataset(pairs, tokenizer)
    print(f"  Dataset size: {len(train_dataset)}")
    
    # Train with DPO
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_dpo(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        output_dir=output_dir,
        args=args,
    )


if __name__ == "__main__":
    main()
