#!/usr/bin/env python3
"""
Self-play generation pipeline for RL training.

This script implements the core self-play loop:
1. Generate questions using the dual-task model
2. Solve generated questions using the same model
3. Verify solutions with SymPy
4. Calculate rewards for both question and solution quality
5. Store trajectories for DPO training

Usage:
    python scripts/self_play_generate.py \
        --adapter checkpoints/dual_task_v1 \
        --output data/rl/self_play_iteration_001.jsonl \
        --num-samples 1000 \
        --reference-questions data/sft/gsm8k_sft.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

if "HF_HUB_DISABLE_XET" not in os.environ:
    os.environ["HF_HUB_DISABLE_XET"] = "1"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.rl.reward_calculator import RewardCalculator, load_reference_questions

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

# Question generation prompt templates
QUESTION_PROMPTS = [
    "Create a word problem about money and fractions requiring {steps} steps.",
    "Generate a problem about time and distance with {steps} calculation steps.",
    "Create a problem about buying multiple items with different quantities requiring {steps} steps.",
    "Generate a problem about percentages and discounts needing {steps} steps.",
    "Create a word problem about ratios and sharing requiring {steps} steps.",
    "Generate a problem about collecting items over time with {steps} steps.",
    "Create a problem about calculating areas or perimeters with {steps} steps.",
    "Generate a problem requiring {steps} multi-step arithmetic operations.",
    "Create a problem about age comparisons requiring {steps} steps.",
    "Generate a problem about scaling recipe quantities with {steps} steps.",
    "Create a problem about test scores and averages needing {steps} steps.",
    "Generate a problem about distributing items among groups with {steps} steps.",
]


def load_model_and_tokenizer(adapter_path: Path, base_model: str, compute_dtype_str: str):
    """Load the dual-task model for self-play."""
    print(f"Loading model from {adapter_path}...")
    
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
        print(f"  Pipeline type: {meta.get('pipeline_type')}")

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()

    print("Model loaded successfully.")
    return model, tokenizer


def generate_question(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """Generate a question using the model."""
    user_content = f"{GENERATE_TASK_PREFIX}{prompt.strip()}"
    messages = [
        {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def generate_solution(
    model: Any,
    tokenizer: Any,
    question: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """Generate a solution for a given question."""
    user_content = (
        f"{SOLVE_TASK_PREFIX}"
        "Solve the following problem. Show your reasoning as numbered steps, "
        "then give the final numeric answer on the last line.\n\n"
        f"Problem:\n{question.strip()}"
    )
    messages = [
        {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def create_trajectory_record(
    iteration: int,
    sample_idx: int,
    generation_prompt: str,
    generated_question: str,
    generated_solution: str,
    reward_result: Any,
) -> dict[str, Any]:
    """Create a trajectory record for storage."""
    return {
        "iteration": iteration,
        "sample_idx": sample_idx,
        "generation_prompt": generation_prompt,
        "generated_question": generated_question,
        "generated_solution": generated_solution,
        "rewards": reward_result.to_dict(),
        "combined_score": reward_result.combined_score,
    }


def run_self_play(
    model: Any,
    tokenizer: Any,
    reward_calculator: RewardCalculator,
    num_samples: int,
    iteration: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    output_path: Path,
) -> dict[str, Any]:
    """
    Run self-play generation loop.
    
    Returns summary statistics.
    """
    print(f"\n{'=' * 60}")
    print(f"Self-Play Generation - Iteration {iteration}")
    print(f"{'=' * 60}")
    print(f"Generating {num_samples} question-solution pairs...")
    
    trajectories = []
    stats = {
        "total_samples": num_samples,
        "avg_combined_score": 0.0,
        "avg_question_score": 0.0,
        "avg_solution_score": 0.0,
        "solvable_questions": 0,
        "valid_solutions": 0,
    }
    
    # Open output file for streaming writes
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for i in range(num_samples):
            if (i + 1) % 10 == 0:
                print(f"Progress: {i+1}/{num_samples}...", end="\r", flush=True)
            
            # Sample a random prompt template and step count
            prompt_template = random.choice(QUESTION_PROMPTS)
            target_steps = random.randint(2, 5)
            generation_prompt = prompt_template.format(steps=target_steps)
            
            # Generate question
            try:
                question = generate_question(
                    model, tokenizer, generation_prompt,
                    max_new_tokens, temperature, top_p
                )
            except Exception as e:
                print(f"\nError generating question {i}: {e}")
                continue
            
            # Generate solution for the question
            try:
                solution = generate_solution(
                    model, tokenizer, question,
                    max_new_tokens, temperature, top_p
                )
            except Exception as e:
                print(f"\nError generating solution {i}: {e}")
                continue
            
            # Calculate rewards
            try:
                reward_result = reward_calculator.calculate_reward(
                    generated_question=question,
                    generated_solution=solution,
                )
            except Exception as e:
                print(f"\nError calculating reward {i}: {e}")
                continue
            
            # Create trajectory record
            trajectory = create_trajectory_record(
                iteration=iteration,
                sample_idx=i,
                generation_prompt=generation_prompt,
                generated_question=question,
                generated_solution=solution,
                reward_result=reward_result,
            )
            
            # Write to file immediately (streaming)
            f.write(json.dumps(trajectory, ensure_ascii=False) + "\n")
            f.flush()
            
            # Update statistics
            stats["avg_combined_score"] += reward_result.combined_score
            stats["avg_question_score"] += reward_result.question_metrics.overall_score
            stats["avg_solution_score"] += reward_result.solution_metrics.overall_score
            
            if reward_result.question_metrics.solution_arithmetic_ok:
                stats["solvable_questions"] += 1
            
            if reward_result.solution_metrics.format_valid:
                stats["valid_solutions"] += 1
    
    # Calculate averages
    if num_samples > 0:
        stats["avg_combined_score"] /= num_samples
        stats["avg_question_score"] /= num_samples
        stats["avg_solution_score"] /= num_samples
    
    print(f"\nCompleted {num_samples} samples.")
    print(f"\nStatistics:")
    print(f"  Avg Combined Score: {stats['avg_combined_score']:.3f}")
    print(f"  Avg Question Score: {stats['avg_question_score']:.3f}")
    print(f"  Avg Solution Score: {stats['avg_solution_score']:.3f}")
    print(f"  Solvable Questions: {stats['solvable_questions']}/{num_samples} ({stats['solvable_questions']/num_samples:.1%})")
    print(f"  Valid Solutions: {stats['valid_solutions']}/{num_samples} ({stats['valid_solutions']/num_samples:.1%})")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Self-play generation for RL training"
    )
    parser.add_argument(
        "--adapter",
        type=str,
        required=True,
        help="Path to dual-task model adapter"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL path for trajectories"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of question-solution pairs to generate"
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=1,
        help="Iteration number (for metadata)"
    )
    parser.add_argument(
        "--reference-questions",
        type=str,
        help="Path to reference questions JSONL for novelty scoring"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B-Instruct"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max tokens for generation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature (higher = more diverse)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95
    )
    parser.add_argument(
        "--bnb-compute-dtype",
        type=str,
        default="bfloat16"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load model
    adapter_path = Path(args.adapter)
    model, tokenizer = load_model_and_tokenizer(
        adapter_path, args.base_model, args.bnb_compute_dtype
    )
    
    # Load reference questions for novelty scoring
    reference_questions = []
    if args.reference_questions:
        ref_path = Path(args.reference_questions)
        if ref_path.exists():
            print(f"\nLoading reference questions from {ref_path}...")
            reference_questions = load_reference_questions(str(ref_path))
            print(f"  Loaded {len(reference_questions)} reference questions")
        else:
            print(f"\nWarning: Reference questions file not found: {ref_path}")
            print("  Novelty scoring will not be accurate.")
    
    # Initialize reward calculator
    reward_calculator = RewardCalculator(
        reference_questions=reference_questions
    )
    
    # Run self-play
    output_path = Path(args.output)
    stats = run_self_play(
        model=model,
        tokenizer=tokenizer,
        reward_calculator=reward_calculator,
        num_samples=args.num_samples,
        iteration=args.iteration,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        output_path=output_path,
    )
    
    # Save summary
    summary_path = output_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "iteration": args.iteration,
                "num_samples": args.num_samples,
                "adapter": str(adapter_path),
                "statistics": stats,
                "config": {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_new_tokens": args.max_new_tokens,
                },
            },
            f,
            indent=2,
        )
    
    print(f"\nTrajectories saved to: {output_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
