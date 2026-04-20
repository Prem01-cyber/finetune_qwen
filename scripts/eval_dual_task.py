#!/usr/bin/env python3
"""
Evaluate dual-task model on both question generation and solution tasks.

This script evaluates:
1. Question Generation: Quality, diversity, and format of generated questions
2. Solution Generation: Accuracy, format compliance, and verification on GSM8K

Examples
--------
  # Evaluate both tasks
  python scripts/eval_dual_task.py \
      --adapter checkpoints/dual_task_v1 \
      --question-prompts data/eval/question_gen_prompts.jsonl \
      --solution-data data/sft/gsm8k_sft.jsonl \
      --output-json reports/dual_task_eval.json

  # Evaluate only solution task
  python scripts/eval_dual_task.py \
      --adapter checkpoints/dual_task_v1 \
      --solution-data data/sft/gsm8k_sft.jsonl \
      --skip-question-eval

  # Evaluate only question generation task
  python scripts/eval_dual_task.py \
      --adapter checkpoints/dual_task_v1 \
      --question-prompts data/eval/question_gen_prompts.jsonl \
      --skip-solution-eval
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if "HF_HUB_DISABLE_XET" not in os.environ:
    os.environ["HF_HUB_DISABLE_XET"] = "1"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from peft import PeftModel
from sympy import simplify
from sympy.parsing.sympy_parser import parse_expr
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from scripts.convert_gsm8k_to_sft import parse_gsm8k_answer
from src.sft.solution_format import extract_final_answer_numeric_str, validate_sympy_solution_format

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

USER_WRAPPER = (
    "Solve the following problem. Show your reasoning as numbered steps, "
    "then give the final numeric answer on the last line.\n\nProblem:\n{question}"
)


@dataclass
class SolutionEvalRow:
    index: int
    question: str
    gold_final: str
    pred_final: str
    exact_match: bool | None
    format_ok: bool
    step_count: int
    output_text: str


@dataclass
class QuestionEvalRow:
    index: int
    prompt: str
    generated_question: str
    is_valid: bool
    has_numbers: bool
    has_question_marker: bool
    no_solution_leak: bool
    length: int
    word_count: int


def load_model_and_tokenizer(adapter_path: Path, base_model: str, compute_dtype_str: str):
    """Load the dual-task model."""
    compute_dtype = getattr(torch, compute_dtype_str)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    # Check for metadata
    meta_path = adapter_path / "pipeline_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        base_model = meta.get("base_model", base_model)
        print(f"Loaded metadata: pipeline_type={meta.get('pipeline_type')}")

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model {base_model} with adapter from {adapter_path}...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()

    return model, tokenizer


def generate_solution(
    model: Any,
    tokenizer: Any,
    problem: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    greedy: bool,
) -> str:
    """Generate a solution for a given problem."""
    user_content = f"{SOLVE_TASK_PREFIX}{USER_WRAPPER.format(question=problem.strip())}"
    messages = [
        {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=not greedy,
            pad_token_id=tokenizer.pad_token_id,
        )

    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def generate_question(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    """Generate a question for a given prompt."""
    user_content = f"{GENERATE_TASK_PREFIX}{prompt.strip()}"
    messages = [
        {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,  # Always sample for question generation
            pad_token_id=tokenizer.pad_token_id,
        )

    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def _norm_expr(s: str) -> str:
    """Normalize expression for comparison."""
    s = s.strip()
    s = s.replace("^", "**")
    s = re.sub(r"[,$€£\s]+", "", s)
    return s


def _equiv_expr(a: str, b: str) -> bool | None:
    """Check if two expressions are equivalent."""
    if not a or not b:
        return None
    a_n = _norm_expr(a)
    b_n = _norm_expr(b)
    try:
        return bool(simplify(parse_expr(a_n) - parse_expr(b_n)) == 0)
    except Exception:
        return a_n == b_n


def validate_question(question: str) -> QuestionEvalRow:
    """Validate a generated question."""
    # Check for solution leakage
    solution_markers = ["step 1:", "step 2:", "final answer:", "solution:"]
    no_solution_leak = not any(marker in question.lower() for marker in solution_markers)
    
    # Check for numbers
    has_numbers = bool(re.search(r'\d+', question))
    
    # Check for question markers
    question_markers = ["?", "how many", "how much", "what is", "calculate", "find"]
    has_question_marker = any(marker in question.lower() for marker in question_markers)
    
    # Overall validity
    is_valid = (
        no_solution_leak and
        has_numbers and
        has_question_marker and
        20 <= len(question) <= 1000
    )
    
    return {
        "is_valid": is_valid,
        "has_numbers": has_numbers,
        "has_question_marker": has_question_marker,
        "no_solution_leak": no_solution_leak,
        "length": len(question),
        "word_count": len(question.split()),
    }


def eval_solution_generation(
    model: Any,
    tokenizer: Any,
    test_data: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    greedy: bool,
) -> dict[str, Any]:
    """Evaluate solution generation task."""
    print("\n" + "=" * 60)
    print("Evaluating Solution Generation")
    print("=" * 60)
    
    results = []
    n_correct = 0
    n_format_ok = 0
    
    for i, example in enumerate(test_data):
        print(f"\rProcessing {i+1}/{len(test_data)}...", end="", flush=True)
        
        question = example["question"]
        gold_final = example["gold_final"]
        
        # Generate solution
        output = generate_solution(
            model, tokenizer, question,
            max_new_tokens, temperature, top_p, greedy
        )
        
        # Extract predicted answer
        pred_final = extract_final_answer_numeric_str(output) or ""
        
        # Check exact match
        exact_match = _equiv_expr(pred_final, gold_final)
        if exact_match:
            n_correct += 1
        
        # Validate format
        format_result = validate_sympy_solution_format(output)
        if format_result.valid:
            n_format_ok += 1
        
        # Count steps
        step_count = len(re.findall(r'Step \d+:', output))
        
        results.append(SolutionEvalRow(
            index=i,
            question=question,
            gold_final=gold_final,
            pred_final=pred_final,
            exact_match=exact_match,
            format_ok=format_result.valid,
            step_count=step_count,
            output_text=output,
        ))
    
    print()  # New line after progress
    
    # Calculate metrics
    n_total = len(results)
    accuracy = n_correct / n_total if n_total > 0 else 0
    format_rate = n_format_ok / n_total if n_total > 0 else 0
    avg_steps = sum(r.step_count for r in results) / n_total if n_total > 0 else 0
    
    summary = {
        "task": "solution_generation",
        "total_examples": n_total,
        "exact_match_accuracy": accuracy,
        "format_compliance_rate": format_rate,
        "avg_step_count": avg_steps,
        "correct": n_correct,
        "format_ok": n_format_ok,
    }
    
    print(f"\nSolution Generation Results:")
    print(f"  Exact Match Accuracy: {accuracy:.2%} ({n_correct}/{n_total})")
    print(f"  Format Compliance: {format_rate:.2%} ({n_format_ok}/{n_total})")
    print(f"  Avg Steps per Solution: {avg_steps:.1f}")
    
    return {
        "summary": summary,
        "examples": [asdict(r) for r in results],
    }


def eval_question_generation(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    """Evaluate question generation task."""
    print("\n" + "=" * 60)
    print("Evaluating Question Generation")
    print("=" * 60)
    
    results = []
    all_questions = []
    
    for i, prompt in enumerate(prompts):
        print(f"\rProcessing {i+1}/{len(prompts)}...", end="", flush=True)
        
        # Generate question
        question = generate_question(
            model, tokenizer, prompt,
            max_new_tokens, temperature, top_p
        )
        
        # Validate
        validation = validate_question(question)
        
        results.append(QuestionEvalRow(
            index=i,
            prompt=prompt,
            generated_question=question,
            **validation,
        ))
        
        if validation["is_valid"]:
            all_questions.append(question.lower())
    
    print()  # New line after progress
    
    # Calculate metrics
    n_total = len(results)
    n_valid = sum(1 for r in results if r.is_valid)
    n_has_numbers = sum(1 for r in results if r.has_numbers)
    n_has_qmark = sum(1 for r in results if r.has_question_marker)
    n_no_leak = sum(1 for r in results if r.no_solution_leak)
    
    avg_length = sum(r.length for r in results) / n_total if n_total > 0 else 0
    avg_words = sum(r.word_count for r in results) / n_total if n_total > 0 else 0
    
    # Diversity: unique bigrams
    all_text = " ".join(all_questions)
    words = all_text.split()
    bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
    unique_bigrams = len(set(bigrams))
    total_bigrams = len(bigrams)
    diversity_score = unique_bigrams / total_bigrams if total_bigrams > 0 else 0
    
    summary = {
        "task": "question_generation",
        "total_examples": n_total,
        "valid_questions": n_valid,
        "valid_rate": n_valid / n_total if n_total > 0 else 0,
        "has_numbers_rate": n_has_numbers / n_total if n_total > 0 else 0,
        "has_question_marker_rate": n_has_qmark / n_total if n_total > 0 else 0,
        "no_solution_leak_rate": n_no_leak / n_total if n_total > 0 else 0,
        "avg_length_chars": avg_length,
        "avg_word_count": avg_words,
        "diversity_score_bigrams": diversity_score,
    }
    
    print(f"\nQuestion Generation Results:")
    print(f"  Valid Questions: {n_valid}/{n_total} ({summary['valid_rate']:.1%})")
    print(f"  Has Numbers: {n_has_numbers}/{n_total} ({summary['has_numbers_rate']:.1%})")
    print(f"  Has Question Marker: {n_has_qmark}/{n_total} ({summary['has_question_marker_rate']:.1%})")
    print(f"  No Solution Leak: {n_no_leak}/{n_total} ({summary['no_solution_leak_rate']:.1%})")
    print(f"  Avg Length: {avg_length:.0f} chars, {avg_words:.0f} words")
    print(f"  Diversity Score: {diversity_score:.2%}")
    
    return {
        "summary": summary,
        "examples": [asdict(r) for r in results],
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate dual-task model")
    parser.add_argument("--adapter", type=str, required=True, help="Path to trained adapter")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--bnb-compute-dtype", type=str, default="bfloat16")
    
    # Question generation eval
    parser.add_argument("--question-prompts", type=str, help="JSONL file with generation prompts")
    parser.add_argument("--skip-question-eval", action="store_true", help="Skip question generation eval")
    parser.add_argument("--max-question-samples", type=int, default=100, help="Max prompts to evaluate")
    
    # Solution generation eval
    parser.add_argument("--solution-data", type=str, help="JSONL file with test problems")
    parser.add_argument("--skip-solution-eval", action="store_true", help="Skip solution generation eval")
    parser.add_argument("--max-solution-samples", type=int, default=100, help="Max problems to evaluate")
    parser.add_argument("--solution-source", choices=["jsonl"], default="jsonl")
    
    # Generation params
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding for solutions")
    
    # Output
    parser.add_argument("--output-json", type=str, help="Output JSON file for detailed results")
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.skip_question_eval and args.skip_solution_eval:
        raise SystemExit("Error: Cannot skip both evaluations")
    
    if not args.skip_question_eval and not args.question_prompts:
        raise SystemExit("Error: --question-prompts required unless --skip-question-eval")
    
    if not args.skip_solution_eval and not args.solution_data:
        raise SystemExit("Error: --solution-data required unless --skip-solution-eval")
    
    # Load model
    adapter_path = Path(args.adapter)
    model, tokenizer = load_model_and_tokenizer(
        adapter_path, args.base_model, args.bnb_compute_dtype
    )
    
    results = {}
    
    # Evaluate question generation
    if not args.skip_question_eval:
        prompt_path = Path(args.question_prompts)
        if not prompt_path.exists():
            raise SystemExit(f"Question prompts file not found: {prompt_path}")
        
        prompts = []
        with prompt_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    prompts.append(obj.get("prompt", ""))
                    if len(prompts) >= args.max_question_samples:
                        break
        
        results["question_generation"] = eval_question_generation(
            model, tokenizer, prompts,
            args.max_new_tokens, args.temperature, args.top_p
        )
    
    # Evaluate solution generation
    if not args.skip_solution_eval:
        solution_path = Path(args.solution_data)
        if not solution_path.exists():
            raise SystemExit(f"Solution data file not found: {solution_path}")
        
        test_data = []
        with solution_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    # Extract question and gold answer
                    if "messages" in obj:
                        for msg in obj["messages"]:
                            if msg["role"] == "user":
                                content = msg["content"]
                                if "Problem:" in content:
                                    question = content.split("Problem:")[-1].strip()
                                else:
                                    question = content.strip()
                                # Remove task prefix if present
                                question = question.replace(SOLVE_TASK_PREFIX, "").strip()
                            elif msg["role"] == "assistant":
                                gold_final = extract_final_answer_numeric_str(msg["content"]) or ""
                        
                        test_data.append({"question": question, "gold_final": gold_final})
                        if len(test_data) >= args.max_solution_samples:
                            break
        
        results["solution_generation"] = eval_solution_generation(
            model, tokenizer, test_data,
            args.max_new_tokens, args.temperature, args.top_p, args.greedy
        )
    
    # Print summary
    print("\n" + "=" * 60)
    print("Dual-Task Evaluation Summary")
    print("=" * 60)
    
    if "question_generation" in results:
        qgen = results["question_generation"]["summary"]
        print(f"\nQuestion Generation:")
        print(f"  Valid Rate: {qgen['valid_rate']:.1%}")
        print(f"  Diversity: {qgen['diversity_score_bigrams']:.2%}")
    
    if "solution_generation" in results:
        sol = results["solution_generation"]["summary"]
        print(f"\nSolution Generation:")
        print(f"  Accuracy: {sol['exact_match_accuracy']:.1%}")
        print(f"  Format Compliance: {sol['format_compliance_rate']:.1%}")
    
    # Save detailed results
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
