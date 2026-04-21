#!/usr/bin/env python3
"""
End-to-end pipeline: Question → Solution → SymPy Verification → Verdict.

This script:
1. Loads a fine-tuned QLoRA adapter
2. Generates a math question (or uses a provided question)
3. Generates a step-by-step solution
4. Normalizes the solution for SymPy
5. Verifies each step's arithmetic using SymPy
6. Produces a final verdict with detailed scoring

Usage
-----
  # Model generates question, solves it, and verifies (fully autonomous)
  python scripts/full_solve_verify_pipeline.py \
      --adapter checkpoints/gsm8k_sft

  # Guide question generation with a hint
  python scripts/full_solve_verify_pipeline.py \
      --adapter checkpoints/gsm8k_sft \
      --question-prompt "Generate a problem about area of rectangles"

  # Use a specific question (skip generation)
  python scripts/full_solve_verify_pipeline.py \
      --adapter checkpoints/gsm8k_sft \
      --question "Find the derivative of f(x) = 3x^4 - 5x^3 + 2x^2 - 7x + 4"

  # Force question generation even when --question is provided
  python scripts/full_solve_verify_pipeline.py \
      --adapter checkpoints/gsm8k_sft \
      --question "ignored" \
      --generate-question

  # JSON mode: saves under reports/ by default, optional custom path
  python scripts/full_solve_verify_pipeline.py \
      --adapter checkpoints/gsm8k_sft \
      --json

  python scripts/full_solve_verify_pipeline.py \
      --adapter checkpoints/gsm8k_sft \
      --json --output runs/my_run.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# HF Hub XET workaround
if "HF_HUB_DISABLE_XET" not in os.environ:
    os.environ["HF_HUB_DISABLE_XET"] = "1"

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.sft.solution_format import validate_sympy_solution_format
from src.sft.step_verify_sympy import verify_solution_text


QUESTION_GENERATOR_SYSTEM_PROMPT = (
    "You are a math problem generator. "
    "Generate a single, clear math word problem or calculation problem. "
    "The problem should be solvable with arithmetic, algebra, or basic calculus. "
    "Output only the problem statement, nothing else."
)

SOLVER_SYSTEM_PROMPT = (
    "You are a step-by-step math solver. "
    "Solve the given problem one step at a time. "
    "Each step must be on its own line, starting with 'Step N:'. "
    "End with a line starting with 'Final Answer:'. "
    "Write every mathematical expression in Python/SymPy syntax "
    "so it can be verified programmatically."
)

QUESTION_GENERATION_PROMPTS = [
    "Generate a word problem about shopping or money.",
    "Generate a problem about distance, speed, and time.",
    "Generate an arithmetic calculation problem with multiple operations.",
    "Generate a problem about area or perimeter of shapes.",
    "Generate a problem involving fractions or percentages.",
    "Generate a simple algebra problem with variables.",
    "Generate a problem about combining or distributing items.",
    "Generate a basic calculus problem about derivatives or integrals.",
]


@dataclass
class PipelineResult:
    """Complete pipeline result with verdict."""

    question: str
    solution_text: str
    format_check: dict[str, Any]
    verification: dict[str, Any]
    verdict: dict[str, Any]


def load_model_and_tokenizer(
    adapter_path: Path,
    base_model: str,
    bnb_compute_dtype: str = "bfloat16",
):
    """Load base model + QLoRA adapter with 4-bit quantization."""
    print(f"Loading base model {base_model} + adapter {adapter_path} ...")

    compute_dtype = getattr(torch, bnb_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

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
    model,
    tokenizer,
    prompt_hint: str | None = None,
    max_new_tokens: int = 150,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> str:
    """Generate a math question using the fine-tuned model."""
    if prompt_hint is None:
        prompt_hint = random.choice(QUESTION_GENERATION_PROMPTS)

    messages = [
        {"role": "system", "content": QUESTION_GENERATOR_SYSTEM_PROMPT},
        {"role": "user", "content": prompt_hint},
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

    gen_ids = out[0, inputs["input_ids"].shape[1] :]
    question = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    
    # Clean up: remove any "Problem:" prefix if model added it
    question = question.replace("Problem:", "").strip()
    
    return question


def generate_solution(
    model,
    tokenizer,
    question: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    greedy: bool = False,
) -> str:
    """Generate step-by-step solution for a given question."""
    user_content = (
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
            do_sample=not greedy,
            pad_token_id=tokenizer.pad_token_id,
        )

    gen_ids = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def compute_verdict(format_result, verification_report) -> dict[str, Any]:
    """
    Compute final verdict based on format check and SymPy verification.

    Scoring:
    - Format: pass/fail (required structure)
    - Verification: steps_ok / steps_total (arithmetic correctness)
    - Overall: combination of both

    Returns verdict dict with status, score, and details.
    """
    # Format check
    format_ok = format_result["ok"]
    has_final = format_result["has_final_line"]
    step_count = format_result["step_count"]

    # Verification summary
    verif_summary = verification_report["summary"]
    steps_total = verif_summary["steps_total"]
    steps_verified_ok = verif_summary["steps_verified_ok"]
    steps_failed = verif_summary["steps_failed"]
    steps_skipped = verif_summary["steps_skipped_no_equality"]
    final_answer_status = verif_summary["final_answer"]

    # Compute scores
    format_score = 1.0 if format_ok else 0.0

    if steps_total > 0:
        verification_score = steps_verified_ok / steps_total
    else:
        verification_score = 0.0

    # Overall score: weighted combination
    # Format is required (50%), verification of steps (50%)
    overall_score = 0.5 * format_score + 0.5 * verification_score

    # Determine overall status
    if overall_score >= 0.9:
        status = "excellent"
    elif overall_score >= 0.7:
        status = "good"
    elif overall_score >= 0.5:
        status = "acceptable"
    elif overall_score >= 0.3:
        status = "poor"
    else:
        status = "failed"

    # Build detailed feedback
    issues = []
    if not format_ok:
        issues.extend(format_result.get("errors", []))
    if steps_failed > 0:
        issues.append(f"{steps_failed} step(s) contain arithmetic errors")
    if steps_skipped == steps_total and steps_total > 0:
        issues.append("No steps could be verified (all skipped)")
    if final_answer_status != "ok":
        issues.append(f"Final answer status: {final_answer_status}")

    return {
        "status": status,
        "overall_score": round(overall_score, 3),
        "format_score": format_score,
        "verification_score": round(verification_score, 3),
        "steps_total": steps_total,
        "steps_verified_ok": steps_verified_ok,
        "steps_failed": steps_failed,
        "steps_skipped": steps_skipped,
        "format_ok": format_ok,
        "has_final_answer": has_final,
        "final_answer_status": final_answer_status,
        "issues": issues,
    }


def run_pipeline(
    model,
    tokenizer,
    question: str | None = None,
    question_prompt: str | None = None,
    generate_question_flag: bool = False,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    greedy: bool = False,
) -> PipelineResult:
    """
    Run the full pipeline: [generate question] → generate solution → verify → verdict.
    """
    # Step 0: Generate question if requested
    if generate_question_flag or question is None:
        print("\n" + "=" * 70)
        print("GENERATING QUESTION")
        print("=" * 70)
        if question_prompt:
            print(f"Prompt hint: {question_prompt}")
        question = generate_question(model, tokenizer, question_prompt)
        print(f"Generated: {question}")
    
    print("\n" + "=" * 70)
    print("QUESTION")
    print("=" * 70)
    print(question)

    # Step 1: Generate solution
    print("\n" + "=" * 70)
    print("GENERATING SOLUTION")
    print("=" * 70)
    solution_text = generate_solution(
        model, tokenizer, question, max_new_tokens, temperature, top_p, greedy
    )
    print(solution_text)

    # Step 2: Format validation
    print("\n" + "=" * 70)
    print("FORMAT VALIDATION")
    print("=" * 70)
    format_result = validate_sympy_solution_format(solution_text)
    print(f"Format OK: {format_result.ok}")
    print(f"Steps: {format_result.step_count}")
    print(f"Has Final Answer: {format_result.has_final_line}")
    if format_result.errors:
        print("Errors:", format_result.errors)

    # Step 3: SymPy verification
    print("\n" + "=" * 70)
    print("SYMPY VERIFICATION")
    print("=" * 70)
    verification_report = verify_solution_text(solution_text)
    print(json.dumps(verification_report.summary, indent=2))

    # Show step-by-step verification details
    for step_check in verification_report.steps:
        status_symbol = {
            "ok": "✓",
            "fail": "✗",
            "skipped": "·",
            "mixed": "~",
        }.get(step_check.status, "?")
        print(f"\n  Step {step_check.step_index} [{status_symbol}]: {step_check.detail}")
        for line_check in step_check.lines:
            line_symbol = {
                "ok": "✓",
                "fail": "✗",
                "skipped": "·",
            }.get(line_check.status, "?")
            print(f"    {line_symbol} Line {line_check.line_index}: {line_check.detail}")

    # Step 4: Compute verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    verdict = compute_verdict(asdict(format_result), asdict(verification_report))
    print(f"Status: {verdict['status'].upper()}")
    print(f"Overall Score: {verdict['overall_score']:.1%}")
    print(f"Format Score: {verdict['format_score']:.1%}")
    print(f"Verification Score: {verdict['verification_score']:.1%}")
    print(f"Steps Verified OK: {verdict['steps_verified_ok']}/{verdict['steps_total']}")
    if verdict["issues"]:
        print("\nIssues:")
        for issue in verdict["issues"]:
            print(f"  - {issue}")

    return PipelineResult(
        question=question,
        solution_text=solution_text,
        format_check=asdict(format_result),
        verification=asdict(verification_report),
        verdict=verdict,
    )


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: Question → Solution → SymPy Verification → Verdict"
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        required=True,
        help="Path to trained QLoRA adapter directory",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B-Instruct",
        help="Base model name (default: Qwen/Qwen2.5-Math-1.5B-Instruct)",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Specific question to solve (if not provided, model generates one)",
    )
    parser.add_argument(
        "--generate-question",
        action="store_true",
        help="Force model to generate a new question even if --question is provided",
    )
    parser.add_argument(
        "--question-prompt",
        type=str,
        default=None,
        help="Hint for question generation (e.g., 'Generate a geometry problem')",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max tokens to generate for solution (default: 512)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling (default: 0.95)",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding (overrides temperature/top-p)",
    )
    parser.add_argument(
        "--bnb-compute-dtype",
        type=str,
        default="bfloat16",
        help="BitsAndBytes compute dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "Write full result as JSON to a file (default: reports/full_solve_verify_<timestamp>.json) "
            "and print the path. Use --output to set the path explicitly."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output path (used with --json; overrides default under reports/)",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="With --json, also print the full JSON to stdout (default: off, file only)",
    )

    args = parser.parse_args()

    # Load model metadata if available
    meta_path = args.adapter / "pipeline_meta.json"
    base_model = args.base_model
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        base_model = meta.get("base_model", base_model)
        print(f"Loaded base model from metadata: {base_model}")

    # Load model
    model, tokenizer = load_model_and_tokenizer(
        args.adapter, base_model, args.bnb_compute_dtype
    )

    # Run pipeline (with optional question generation)
    result = run_pipeline(
        model,
        tokenizer,
        question=args.question,
        question_prompt=args.question_prompt,
        generate_question_flag=args.generate_question,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        greedy=args.greedy,
    )

    # Output: --json or --output persists JSON to disk; --json without --output uses reports/ts.json
    if args.json or args.output:
        result_dict = asdict(result)
        json_text = json.dumps(result_dict, indent=2, ensure_ascii=False)

        if args.output is not None:
            out_path: Path = args.output
        elif args.json:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = ROOT / "reports" / f"full_solve_verify_{ts}.json"
        else:
            out_path = None

        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json_text, encoding="utf-8")
            print("\n" + "=" * 70)
            print("JSON SAVED")
            print("=" * 70)
            print(f"Written to: {out_path.resolve()}")

        if args.print_json:
            print("\n" + "=" * 70)
            print("JSON OUTPUT (stdout)")
            print("=" * 70)
            print(json_text)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
