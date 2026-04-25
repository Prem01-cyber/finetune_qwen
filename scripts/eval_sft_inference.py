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
from src.config.prompts import create_solver_messages
from src.sft.solution_format import extract_final_answer_numeric_str, validate_sympy_solution_format
from src.sft.sympy_normalize import normalize_for_parse_expr


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
    """Check if two answer strings are mathematically equivalent.

    Uses the same normalization as CurriculumMathEnvironment._answers_equivalent
    so eval and training agree on what counts as "correct".
    """
    if not a or not b:
        return None
    a_n = normalize_for_parse_expr(_norm_expr(a))
    b_n = normalize_for_parse_expr(_norm_expr(b))
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
    # Use the canonical solver prompt (same system + user format as GRPO training)
    # so eval measures the model under the exact distribution it was trained on.
    messages = create_solver_messages(problem.strip())
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


def _infer_dataset_name(data_path: str) -> str:
    """Derive a short human-readable dataset label from the file path."""
    stem = Path(data_path).stem.lower()   # e.g. "aqua_validation", "gsm8k_test"
    if "aqua" in stem:
        return "AQuA-RAT"
    if "math" in stem:
        return "MATH"
    if "gsm" in stem:
        return "GSM8K"
    return Path(data_path).stem          # fallback: raw filename stem


def evaluate_gsm8k(
    model: Any,
    tokenizer: Any,
    data_path: str = "data/sft/gsm8k_test.jsonl",
    max_samples: int = 500,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    reward_fn: Any = None,
    pass_at_k: int = 0,
    dataset_name: str = "",
    pass_at_k_temperature: float = 0.8,
) -> dict:
    """
    Evaluate *model* on a math JSONL file using the SAME scoring
    function used during GRPO training.

    Args:
        model        : AutoModelForCausalLM (already on correct device).
        tokenizer    : Matching AutoTokenizer.
        data_path    : Path to JSONL with {question, answer} rows.
        max_samples  : Evaluation cap.
        max_new_tokens / temperature / top_p : generation hyper-params.
        reward_fn    : callable(question: str, solution: str, gold: str) -> dict
                       Must return at minimum {"combined_score": float} and
                       optionally {"gt_match": bool, "prm_mean_score": float,
                       "sympy_score": float, "format_score": float}.
                       When supplied the primary accuracy metric becomes the
                       mean combined_score — identical to the GRPO training
                       objective — so every component (correctness, PRM step
                       quality, SymPy verification, format) contributes and
                       improvements in any of them show up immediately.
                       When None the function falls back to final-answer
                       exact-match accuracy (coarse binary).

    Returns dict keys:
        accuracy          – mean combined_score per solution (or exact-match if no reward_fn)
        combined_score    – same as accuracy (alias)
        correct_rate      – fraction of solutions with gt_match == True
        prm_mean          – mean PRM step-quality score per solution
        sympy_mean        – mean SymPy verification score
        format_mean       – mean format compliance score
        n_scored          – solutions successfully scored by reward_fn
        total             – total solutions evaluated
        # fallback (no reward_fn):
        exact_match_rate  – fraction of final answers matching gold
    """
    import logging as _logging
    _logger = _logging.getLogger(__name__)

    greedy = temperature < 1e-6
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
                if "question" in obj and "gold_final" in obj and obj["gold_final"]:
                    # Pre-extracted format (our gsm8k_test.jsonl)
                    rows.append({"question": obj["question"].strip(), "gold_final": obj["gold_final"].strip()})
                elif "question" in obj and "answer" in obj:
                    _, final = parse_gsm8k_answer(obj["answer"])
                    if final:
                        rows.append({"question": obj["question"].strip(), "gold_final": final})
                elif "messages" in obj:
                    task_type = obj.get("task_type", "solve")
                    if task_type != "solve":
                        continue   # skip question-generation entries
                    user = next(
                        (m["content"] for m in obj["messages"] if m.get("role") == "user"), ""
                    ).strip()
                    asst = next(
                        (m["content"] for m in obj["messages"] if m.get("role") == "assistant"), ""
                    )
                    gold = extract_final_answer_numeric_str(asst) or ""
                    if not gold:
                        continue   # skip entries with no parseable gold answer
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
    total   = len(rows)
    _n_errors = 0
    _MAX_ERROR_WARNINGS = 3

    # Per-solution reward accumulators (populated when reward_fn is supplied).
    _combined:  list[float] = []
    _gt_match:  list[float] = []
    _prm_comp:  list[float] = []
    _prm_final: list[float] = []
    _step_acc:  list[float] = []   # fraction of steps rated correct by PRM (>0.5)
    _lccp:      list[float] = []   # longest correct consecutive prefix ratio
    _sympy_comp:list[float] = []
    _fmt_comp:  list[float] = []

    # Pass@K accumulators: for each problem, did ANY of K samples get it right?
    _pak_any_correct: list[int] = []   # 1 if any of K samples correct, else 0

    _eval_label = dataset_name or _infer_dataset_name(data_path)
    pbar = tqdm(
        rows, total=total, desc=f"{_eval_label} eval",
        unit="q", dynamic_ncols=True, leave=True,
    )
    for i, row in enumerate(pbar):
        pred_text = ""
        try:
            pred_text = _generate(
                model=model, tokenizer=tokenizer,
                problem=row["question"],
                max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p, greedy=greedy,
            )
            pred_final = extract_final_answer_numeric_str(pred_text) or ""
            if _equiv_expr(pred_final, row["gold_final"]):
                correct += 1
        except Exception as exc:
            _n_errors += 1
            if _n_errors <= _MAX_ERROR_WARNINGS:
                _logger.warning(
                    "evaluate_gsm8k: sample %d raised %s: %s. "
                    "If all fail check that tokenizer has a chat_template.",
                    i, type(exc).__name__, exc,
                )
            elif _n_errors == _MAX_ERROR_WARNINGS + 1:
                _logger.warning(
                    "evaluate_gsm8k: suppressing further errors (%d so far).",
                    _n_errors,
                )
            _logger.debug("Sample %d error: %s", i, exc, exc_info=True)

        # ── Pass@K: sample K solutions at T=0.8 and check if any is correct ─
        # This is the fair comparison to batch_acc during training (also K samples
        # at T=0.8). Greedy (pass@1) is pessimistic; pass@k shows the upper bound
        # the model can achieve with sampling, matching the training regime.
        if pass_at_k > 1 and row.get("gold_final"):
            _any = 0
            for _ in range(pass_at_k):
                try:
                    s = _generate(
                        model=model, tokenizer=tokenizer,
                        problem=row["question"],
                        max_new_tokens=max_new_tokens,
                        temperature=pass_at_k_temperature,
                        top_p=top_p, greedy=False,
                    )
                    pf = extract_final_answer_numeric_str(s) or ""
                    if _equiv_expr(pf, row["gold_final"]):
                        _any = 1
                        break
                except Exception:
                    pass
            _pak_any_correct.append(_any)

        # ── Apply the SAME reward function used during GRPO training ──────────
        if reward_fn is not None and pred_text:
            try:
                r = reward_fn(row["question"], pred_text, row["gold_final"])
                _combined.append(float(r.get("combined_score",   0.0)))
                _gt_match.append(1.0 if r.get("gt_match", False) else 0.0)
                _prm_comp.append(float(r.get("prm_mean_score",   0.0)))
                _prm_final.append(float(r.get("prm_final_score", 0.0)))
                _step_acc.append(float(r.get("step_accuracy",    0.0)))
                _lccp.append(float(r.get("lccp",                 0.0)))
                _sympy_comp.append(float(r.get("sympy_score",    0.0)))
                _fmt_comp.append(float(r.get("format_score",     0.0)))
            except Exception as rfn_exc:
                _logger.debug("reward_fn failed for sample %d: %s", i, rfn_exc)

        done = i + 1
        # Periodically flush the CUDA allocator's free-block pool so that
        # fragmentation from large KV-cache + PRM tensors doesn't accumulate
        # and cause per-sample allocation time to grow throughout the run.
        if done % 20 == 0:
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Live bar: show training-objective score when available, else acc.
        if _combined:
            _pf: dict = dict(
                score=f"{sum(_combined) / len(_combined):.3f}",
                correct=f"{sum(_gt_match):.0f}/{len(_combined)}",
                step_acc=f"{sum(_step_acc)/len(_step_acc):.1%}" if _step_acc else "—",
                lccp=f"{sum(_lccp)/len(_lccp):.1%}" if _lccp else "—",
            )
        else:
            _pf = dict(acc=f"{correct / done:.1%}", correct=f"{correct}/{done}")
        pbar.set_postfix(**_pf, refresh=False)

    # ── Aggregate ──────────────────────────────────────────────────────────
    n_scored = len(_combined)
    _avg = lambda lst: round(sum(lst) / len(lst), 4) if lst else 0.0

    # Pass@K: fraction of problems where any of K sampled solutions was correct.
    pass_at_k_score = _avg(_pak_any_correct) if _pak_any_correct else None

    if reward_fn is not None:
        combined_score = _avg(_combined)
        result: dict = {
            # PRIMARY: mean training-objective score.
            # Formula: 0.50×correct + 0.40×process(prm_final, prm_mean) + 0.10×format
            "accuracy":       combined_score,
            "combined_score": combined_score,
            # PROCESS metrics — improve before correct_rate does
            "step_accuracy":  _avg(_step_acc),
            "lccp":           _avg(_lccp),   # chain integrity: how far into solution stays correct
            # Answer correctness
            "correct_rate":   _avg(_gt_match),
            # PRM components
            "prm_mean":       _avg(_prm_comp),
            "prm_final":      _avg(_prm_final),
            # Format / SymPy (informational)
            "sympy_mean":     _avg(_sympy_comp),
            "format_mean":    _avg(_fmt_comp),
            "n_scored":       n_scored,
            "total":          total,
            "final_answer_correct":  correct,
            "final_answer_accuracy": correct / total if total else 0.0,
        }
    else:
        _logger.warning(
            "evaluate_gsm8k: no reward_fn provided — using final-answer accuracy. "
            "Pass reward_fn=math_env.compute_grounded_reward for full training-objective eval."
        )
        fa_acc = correct / total if total else 0.0
        result = {
            "accuracy":              fa_acc,
            "combined_score":        fa_acc,
            "correct_rate":          fa_acc,
            "prm_mean":              0.0,
            "sympy_mean":            0.0,
            "format_mean":           0.0,
            "n_scored":              0,
            "total":                 total,
            "final_answer_correct":  correct,
            "final_answer_accuracy": fa_acc,
        }
    # Attach pass@k if it was computed
    if pass_at_k_score is not None:
        result["pass_at_k"]     = pass_at_k_score
        result["pass_at_k_k"]   = pass_at_k
    return result


if __name__ == "__main__":
    main()
