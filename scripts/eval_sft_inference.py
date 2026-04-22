#!/usr/bin/env python3
"""
Run batch inference for a trained QLoRA adapter and report quality metrics.

This helps decide whether another SFT epoch is needed before RL.

Examples
--------
  # Evaluate on GSM8K test split (first 100 samples)
  python scripts/eval_sft_inference.py \
      --adapter checkpoints/gsm8k_sft \
      --max-samples 100

  # Evaluate on local JSONL with {question, answer} rows
  python scripts/eval_sft_inference.py \
      --adapter checkpoints/gsm8k_sft \
      --source jsonl \
      --input data/raw/gsm8k_test.jsonl \
      --max-samples 50 \
      --output-json reports/sft_eval.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

# Prefer classic HTTP Hub downloads by default.
if "HF_HUB_DISABLE_XET" not in os.environ:
    os.environ["HF_HUB_DISABLE_XET"] = "1"

# Ensure project-root imports work when invoked as `python scripts/...`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from datasets import load_dataset
from peft import PeftModel
from sympy import simplify
from sympy.parsing.sympy_parser import parse_expr
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from scripts.convert_gsm8k_to_sft import parse_gsm8k_answer
from src.sft.solution_format import extract_final_answer_numeric_str, validate_sympy_solution_format

SOLVER_SYSTEM_PROMPT = (
    "You are a step-by-step math solver. "
    "Solve the given problem one step at a time. "
    "Each step must be on its own line, starting with 'Step N:'. "
    "End with a line starting with 'Final Answer:'. "
    "Write every mathematical expression in Python/SymPy syntax "
    "so it can be verified programmatically."
)

USER_WRAPPER = (
    "Solve the following problem. Show your reasoning as numbered steps, "
    "then give the final numeric answer on the last line.\n\nProblem:\n{question}"
)


@dataclass
class EvalRow:
    index: int
    question: str
    gold_final: str
    pred_final: str
    exact_match: Optional[bool]
    format_ok: bool
    step_count: int
    scratchpad_leak: bool
    output_text: str


def _norm_expr(s: str) -> str:
    s = s.strip()
    s = s.replace("^", "**")
    s = re.sub(r"[,$€£\s]+", "", s)
    return s


def _equiv_expr(a: str, b: str) -> Optional[bool]:
    if not a or not b:
        return None
    a_n = _norm_expr(a)
    b_n = _norm_expr(b)
    try:
        return bool(simplify(parse_expr(a_n) - parse_expr(b_n)) == 0)
    except Exception:
        return a_n == b_n


def _iter_examples(args: argparse.Namespace) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if args.source == "hf":
        ds = load_dataset(args.dataset, args.config, split=args.split)
        if args.max_samples > 0:
            ds = ds.select(range(min(args.max_samples, len(ds))))
        for row in ds:
            _, final = parse_gsm8k_answer(row["answer"])
            rows.append({"question": row["question"].strip(), "gold_final": final})
        return rows

    in_path = Path(args.input)
    if not in_path.is_file():
        raise SystemExit(f"Input JSONL not found: {in_path}")
    with in_path.open(encoding="utf-8") as f:
        for line in f:
            if args.max_samples > 0 and len(rows) >= args.max_samples:
                break
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if "question" in o and "answer" in o:
                _, final = parse_gsm8k_answer(o["answer"])
                rows.append({"question": o["question"].strip(), "gold_final": final})
                continue
            if "messages" in o:
                user = next((m["content"] for m in o["messages"] if m.get("role") == "user"), "").strip()
                asst = next((m["content"] for m in o["messages"] if m.get("role") == "assistant"), "")
                gold = extract_final_answer_numeric_str(asst) or ""
                user = re.sub(r"^Solve the following problem\..*?Problem:\n", "", user, flags=re.S)
                rows.append({"question": user.strip(), "gold_final": gold.strip()})
                continue
            raise SystemExit("JSONL rows must contain either {question, answer} or {messages}.")
    return rows


def _generate(
    model: Any,
    tokenizer: Any,
    problem: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    greedy: bool,
) -> str:
    user_content = USER_WRAPPER.format(question=problem.strip())
    messages = [
        {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # HuggingFace warns once-per-call when `temperature`/`top_p` are passed
    # alongside `do_sample=False`.  Skip those kwargs entirely in greedy mode
    # so long eval loops don't spam the log.
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": not greedy,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if not greedy:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    gen_ids = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def main() -> None:
    p = argparse.ArgumentParser(description="Batch eval for SFT adapter inference.")
    p.add_argument("--adapter", type=Path, required=True, help="Adapter directory from training step.")
    p.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    p.add_argument("--source", choices=("hf", "jsonl"), default="hf")
    p.add_argument("--dataset", type=str, default="openai/gsm8k")
    p.add_argument("--config", type=str, default="main")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--input", type=Path, help="JSONL path for --source jsonl")
    p.add_argument("--max-samples", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--greedy", action="store_true", default=True)
    p.add_argument("--no-greedy", dest="greedy", action="store_false")
    p.add_argument("--bnb-compute-dtype", type=str, default="bfloat16")
    p.add_argument("--show-samples", type=int, default=3)
    p.add_argument("--output-json", type=Path, default=None)
    args = p.parse_args()

    if args.source == "jsonl" and not args.input:
        raise SystemExit("--input is required when --source jsonl")

    meta_path = args.adapter / "pipeline_meta.json"
    base_model = args.base_model
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        base_model = meta.get("base_model", base_model)

    rows = _iter_examples(args)
    if not rows:
        raise SystemExit("No evaluation examples loaded.")
    print(f"Loaded {len(rows)} evaluation examples.")

    compute_dtype = getattr(torch, args.bnb_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading base {base_model} + adapter {args.adapter} …")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(args.adapter))
    model.eval()

    results: list[EvalRow] = []
    for i, row in enumerate(rows):
        text = _generate(
            model=model,
            tokenizer=tokenizer,
            problem=row["question"],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            greedy=args.greedy,
        )
        fmt = validate_sympy_solution_format(text)
        pred_final = extract_final_answer_numeric_str(text) or ""
        exact = _equiv_expr(pred_final, row["gold_final"])
        results.append(
            EvalRow(
                index=i,
                question=row["question"],
                gold_final=row["gold_final"],
                pred_final=pred_final,
                exact_match=exact,
                format_ok=fmt.ok,
                step_count=fmt.step_count,
                scratchpad_leak=("<<" in text and ">>" in text),
                output_text=text,
            )
        )
        if i < args.show_samples:
            print(f"\n=== Sample {i} ===")
            print("Q:", row["question"])
            print("Gold:", row["gold_final"])
            print("Pred:", pred_final)
            print("Format OK:", fmt.ok, "| Steps:", fmt.step_count)
            print(text)

    n = len(results)
    n_format_ok = sum(1 for r in results if r.format_ok)
    n_scratch = sum(1 for r in results if r.scratchpad_leak)
    em_scored = [r for r in results if r.exact_match is not None]
    n_em = sum(1 for r in em_scored if r.exact_match)

    print("\n=== Summary ===")
    print(f"Samples: {n}")
    print(f"Format OK: {n_format_ok}/{n} ({100.0 * n_format_ok / n:.2f}%)")
    print(f"Scratchpad leakage (<< >>): {n_scratch}/{n} ({100.0 * n_scratch / n:.2f}%)")
    if em_scored:
        print(f"Exact match (final answer): {n_em}/{len(em_scored)} ({100.0 * n_em / len(em_scored):.2f}%)")
    else:
        print("Exact match (final answer): N/A (missing gold labels)")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": {
                "samples": n,
                "format_ok": n_format_ok,
                "format_ok_rate": n_format_ok / n,
                "scratchpad_leakage": n_scratch,
                "scratchpad_leakage_rate": n_scratch / n,
                "exact_match_scored": len(em_scored),
                "exact_match": n_em,
                "exact_match_rate": (n_em / len(em_scored)) if em_scored else None,
            },
            "results": [asdict(r) for r in results],
        }
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote detailed report to {args.output_json}")


def evaluate_gsm8k(
    model: Any,
    tokenizer: Any,
    data_path: str = "data/sft/gsm8k_test.jsonl",
    max_samples: int = 500,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> dict:
    """
    Evaluate *model* on a GSM8K-formatted JSONL file.

    Called by ``scripts/run_ppo_training.py`` at each eval step.

    Args:
        model          : AutoModelForCausalLM (already loaded, on correct device).
        tokenizer      : Matching AutoTokenizer.
        data_path      : Path to JSONL with {question, answer} rows.
                         Falls back to the HuggingFace hub split when not found.
        max_samples    : Evaluation cap (for speed during training).
        max_new_tokens : Generation budget per problem.
        temperature    : Sampling temperature (0 → greedy).
        top_p          : Nucleus sampling p.

    Returns:
        dict with keys: accuracy, correct, total, exact_match_rate
    """
    import logging as _logging
    _logger = _logging.getLogger(__name__)

    greedy = temperature == 0.0
    rows: list[dict] = []

    p = Path(data_path)
    if p.is_file():
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                if max_samples > 0 and len(rows) >= max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "question" in obj and "answer" in obj:
                    _, final = parse_gsm8k_answer(obj["answer"])
                    rows.append({"question": obj["question"].strip(), "gold_final": final})
                elif "messages" in obj:
                    user = next(
                        (m["content"] for m in obj["messages"] if m.get("role") == "user"), ""
                    ).strip()
                    asst = next(
                        (m["content"] for m in obj["messages"] if m.get("role") == "assistant"), ""
                    )
                    gold = extract_final_answer_numeric_str(asst) or ""
                    user = re.sub(r"^Solve the following problem\..*?Problem:\n", "", user, flags=re.S)
                    rows.append({"question": user.strip(), "gold_final": gold.strip()})
    else:
        _logger.warning(
            f"evaluate_gsm8k: {data_path} not found; loading openai/gsm8k from Hub."
        )
        try:
            ds = load_dataset("openai/gsm8k", "main", split="test")
            if max_samples > 0:
                ds = ds.select(range(min(max_samples, len(ds))))
            for row in ds:
                _, final = parse_gsm8k_answer(row["answer"])
                rows.append({"question": row["question"].strip(), "gold_final": final})
        except Exception as exc:
            _logger.error(f"Could not load GSM8K: {exc}")
            return {"accuracy": 0.0, "correct": 0, "total": 0, "exact_match_rate": 0.0}

    if not rows:
        return {"accuracy": 0.0, "correct": 0, "total": 0, "exact_match_rate": 0.0}

    correct = 0
    total = len(rows)

    # tqdm gives us a live progress bar with ETA; the `postfix` surfaces the
    # running accuracy so long runs are interpretable mid-flight.  Falls back
    # gracefully in non-TTY environments (tqdm.auto picks notebook vs plain).
    pbar = tqdm(
        rows,
        total=total,
        desc="GSM8K eval",
        unit="q",
        dynamic_ncols=True,
        leave=True,
    )
    for i, row in enumerate(pbar):
        try:
            pred_text = _generate(
                model=model,
                tokenizer=tokenizer,
                problem=row["question"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                greedy=greedy,
            )
            pred_final = extract_final_answer_numeric_str(pred_text) or ""
            gold_final = row["gold_final"]
            match = _equiv_expr(pred_final, gold_final)
            if match:
                correct += 1
        except Exception as exc:
            _logger.debug(f"Sample {i} error: {exc}")

        done = i + 1
        pbar.set_postfix(
            acc=f"{correct / done:.1%}",
            correct=f"{correct}/{done}",
            refresh=False,
        )

    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "exact_match_rate": accuracy,
    }


if __name__ == "__main__":
    main()
