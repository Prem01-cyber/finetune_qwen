#!/usr/bin/env python3
"""
Inspect dual-task (question + solve) model quality with SymPy verification.

For each sample this script:
  1. Samples a question-generation instruction (same pool as self-play)
  2. Generates a question with the model
  3. Generates a solution with the model
  4. Prints structural format check (Step N: / Final Answer:)
  5. Prints full SymPy step verification (equality chains, final answer)
  6. Prints RewardCalculator breakdown (Q / S components)

Examples
--------
  # Quick visual inspection (3 samples, verbose SymPy)
  python scripts/inspect_dual_task_quality.py \\
      --adapter checkpoints/dual_task_v1 \\
      --num-samples 3

  # More samples, save JSON for later review
  python scripts/inspect_dual_task_quality.py \\
      --adapter checkpoints/dual_task_v1 \\
      --num-samples 10 \\
      --json-out reports/dual_task_inspect.json \\
      --reference-questions data/sft/gsm8k_sft.jsonl

  # Less console output (SymPy summary only)
  python scripts/inspect_dual_task_quality.py \\
      --adapter checkpoints/dual_task_v1 \\
      --num-samples 5 \\
      --sympy-brief
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

from scripts.self_play_generate import (
    QUESTION_PROMPTS,
    generate_question,
    generate_solution,
    load_model_and_tokenizer,
)
from src.rl.reward_calculator import RewardCalculator, load_reference_questions
from src.sft.solution_format import validate_sympy_solution_format
from src.sft.step_verify_sympy import VerificationReport, print_report, report_to_dict, verify_solution_text


def _print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def _print_sample(
    idx: int,
    generation_prompt: str,
    question: str,
    solution: str,
    format_ok: bool,
    format_errors: list[str],
    reward_dict: dict[str, Any],
    ver: VerificationReport,
    sympy_brief: bool,
) -> None:
    _print_section(f"SAMPLE {idx + 1}")
    print("\n--- Generation instruction (user) ---\n")
    print(generation_prompt.strip())
    print("\n--- Generated question ---\n")
    print(question.strip() if question else "(empty)")
    print("\n--- Generated solution ---\n")
    print(solution.strip() if solution else "(empty)")

    print("\n--- Format check (Step N: / Final Answer:) ---")
    print(f"  ok: {format_ok}")
    if format_errors:
        for e in format_errors:
            print(f"  note: {e}")

    print("\n--- RewardCalculator (same as RL / self-play) ---")
    print(json.dumps(reward_dict, indent=2, ensure_ascii=False))

    print("\n--- SymPy verification (arithmetic along '=' chains) ---")
    if sympy_brief:
        print(json.dumps(ver.summary, indent=2))
        for s in ver.steps:
            short = s.detail.replace("\n", " ")[:140]
            print(f"  Step {s.step_index}: [{s.status}] {short}...")
        fa = ver.final_answer
        print(f"  Final: [{fa.status}] {fa.detail}")
    else:
        print_report(ver, verbose=True)


def run_inspection(args: argparse.Namespace) -> dict[str, Any]:
    adapter_path = Path(args.adapter)
    model, tokenizer = load_model_and_tokenizer(
        adapter_path, args.base_model, args.bnb_compute_dtype
    )

    reference_questions: list[str] = []
    if args.reference_questions:
        ref_path = Path(args.reference_questions)
        if ref_path.is_file():
            reference_questions = load_reference_questions(str(ref_path))
            print(f"Loaded {len(reference_questions)} reference questions for novelty.")

    reward_calculator = RewardCalculator(reference_questions=reference_questions)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for i in range(args.num_samples):
        prompt_template = random.choice(QUESTION_PROMPTS)
        target_steps = random.randint(2, 5)
        generation_prompt = prompt_template.format(steps=target_steps)

        try:
            question = generate_question(
                model,
                tokenizer,
                generation_prompt,
                args.max_question_tokens,
                args.temperature,
                args.top_p,
            )
        except Exception as e:
            print(f"\n[sample {i}] error generating question: {e}")
            continue

        try:
            solution = generate_solution(
                model,
                tokenizer,
                question,
                args.max_solution_tokens,
                args.temperature,
                args.top_p,
            )
        except Exception as e:
            print(f"\n[sample {i}] error generating solution: {e}")
            continue

        fmt = validate_sympy_solution_format(solution)
        format_errors = list(fmt.errors)

        ver = verify_solution_text(solution)
        ver_dict = report_to_dict(ver)

        rr = reward_calculator.calculate_reward(
            generated_question=question,
            generated_solution=solution,
        )
        reward_dict = rr.to_dict()

        record = {
            "sample_index": i,
            "generation_prompt": generation_prompt,
            "generated_question": question,
            "generated_solution": solution,
            "format_check": {
                "ok": fmt.ok,
                "step_count": fmt.step_count,
                "has_final_line": fmt.has_final_line,
                "errors": format_errors,
            },
            "sympy_verification": ver_dict,
            "reward": reward_dict,
        }
        records.append(record)
        summaries.append(ver.summary)

        if not args.quiet:
            _print_sample(
                idx=i,
                generation_prompt=generation_prompt,
                question=question,
                solution=solution,
                format_ok=fmt.ok,
                format_errors=format_errors,
                reward_dict=reward_dict,
                ver=ver,
                sympy_brief=args.sympy_brief,
            )

    # Aggregate summary
    n = len(records)
    aggregate: dict[str, Any] = {
        "num_samples_requested": args.num_samples,
        "num_samples_ok": n,
        "adapter": str(adapter_path),
        "seed": args.seed,
    }
    if n:
        avg_combined = sum(r["reward"]["combined_score"] for r in records) / n
        fmt_ok_rate = sum(1 for r in records if r["format_check"]["ok"]) / n
        sympy_ok = sum(
            1 for r in records if r["sympy_verification"]["summary"].get("final_answer") == "ok"
        ) / n
        aggregate["avg_combined_reward"] = avg_combined
        aggregate["format_ok_rate"] = fmt_ok_rate
        aggregate["sympy_final_answer_ok_rate"] = sympy_ok

    _print_section("AGGREGATE (this run)")
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"aggregate": aggregate, "samples": records}
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote full JSON to {out_path}")

    return aggregate


def main() -> None:
    p = argparse.ArgumentParser(
        description="Inspect dual-task Q+S generation and SymPy verification (human-readable)."
    )
    p.add_argument("--adapter", type=Path, required=True, help="Dual-task adapter dir.")
    p.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-Math-7B-Instruct",
        help="Base model id (overridden by pipeline_meta.json if present).",
    )
    p.add_argument("--num-samples", type=int, default=3, help="Number of Q+S pairs to generate.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-question-tokens", type=int, default=256)
    p.add_argument("--max-solution-tokens", type=int, default=512)
    p.add_argument(
        "--reference-questions",
        type=Path,
        default=None,
        help="Optional JSONL (e.g. gsm8k_sft) for novelty in reward breakdown.",
    )
    p.add_argument(
        "--sympy-brief",
        action="store_true",
        help="Shorter SymPy section (per-step one line + summary).",
    )
    p.add_argument("--quiet", action="store_true", help="No per-sample print; only aggregate.")
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write full JSON report (all samples + aggregate).",
    )
    p.add_argument("--bnb-compute-dtype", type=str, default="bfloat16")
    args = p.parse_args()
    run_inspection(args)


if __name__ == "__main__":
    main()
