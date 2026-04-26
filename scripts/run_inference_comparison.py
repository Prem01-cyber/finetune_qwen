#!/usr/bin/env python3
"""
Inference comparison pipeline: Base Qwen vs RL fine-tuned model on GSM8K.

Loads both models, runs the same prompt through each, scores every answer,
and writes three report artefacts to reports/<run_id>/:

    results.json        — full per-question data (machine-readable)
    report.html         — rich HTML report with side-by-side comparisons
    summary.md          — markdown summary table for README / docs

Usage
-----
# Compare base vs SFT-only adapter (default)
python scripts/run_inference_comparison.py

# Point at a specific RL checkpoint (best_policy from a GRPO run)
python scripts/run_inference_comparison.py \
    --finetuned checkpoints/grpo/grpo_20260425_151304/best_policy

# More samples, both models greedy-decode with temperature 0
python scripts/run_inference_comparison.py \
    --max-samples 200 --temperature 0.0

# Skip the slow base model load (compare two fine-tuned checkpoints)
python scripts/run_inference_comparison.py \
    --base-checkpoint checkpoints/dual_task_v1 \
    --finetuned       checkpoints/grpo/grpo_20260425_151304/best_policy \
    --base-label      "SFT only" \
    --finetuned-label "GRPO (iter 10)"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import textwrap
import time
import types
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config.prompts import create_solver_messages
from src.sft.solution_format import extract_final_answer_numeric_str

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Answer normalisation / matching  (same logic as CurriculumMathEnvironment)
# ---------------------------------------------------------------------------

def _strip(s: str) -> str:
    return re.sub(r"[,$€£%\s]+", "", s.strip()).replace("^", "**")


def _answers_match(pred: str, gold: str) -> bool:
    """Numeric/symbolic equivalence, identical to training reward."""
    p, g = _strip(pred), _strip(gold)
    if not p or not g:
        return False
    if p == g:
        return True
    # try float comparison with tolerance
    try:
        return abs(float(p) - float(g)) < 1e-6
    except ValueError:
        pass
    # SymPy fallback
    try:
        from sympy import simplify
        from sympy.parsing.sympy_parser import parse_expr
        from src.sft.sympy_normalize import normalize_for_parse_expr
        return bool(simplify(parse_expr(normalize_for_parse_expr(p))
                              - parse_expr(normalize_for_parse_expr(g))) == 0)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Step parsing and scoring
# ---------------------------------------------------------------------------

# A "step" is any line matching "Step N:" (case-insensitive).
_STEP_RE = re.compile(r"^\s*Step\s+(\d+)\s*:(.*)", re.IGNORECASE)
_FINAL_RE = re.compile(r"(?:^|\n)\s*Final Answer\s*:\s*(.+)", re.IGNORECASE)


def _parse_steps(text: str) -> List[Dict[str, Any]]:
    """
    Split solution text into a list of step dicts:
        { "number": int, "text": str, "has_math": bool, "numbered_ok": bool }

    Also appends a synthetic "Final Answer" step if that line is present.
    """
    steps: List[Dict[str, Any]] = []
    current_num: Optional[int] = None
    current_lines: List[str] = []

    def _flush():
        if current_num is not None and current_lines:
            body = " ".join(l.strip() for l in current_lines if l.strip())
            steps.append({
                "number":      current_num,
                "text":        body,
                "has_math":    bool(re.search(r"[\d\+\-\*\/\=\(\)]", body)),
                "numbered_ok": True,          # came from a valid "Step N:" line
            })

    for line in text.splitlines():
        m = _STEP_RE.match(line)
        if m:
            _flush()
            current_num = int(m.group(1))
            current_lines = [m.group(2)]
        elif current_num is not None:
            # continuation line of the current step
            current_lines.append(line)

    _flush()

    # Append the Final Answer as a special step
    fa_match = _FINAL_RE.search(text)
    if fa_match:
        steps.append({
            "number":      len(steps) + 1,
            "text":        "Final Answer: " + fa_match.group(1).strip(),
            "has_math":    True,
            "numbered_ok": True,
            "is_final":    True,
        })

    return steps


def _score_steps(
    steps: List[Dict[str, Any]],
    ref_steps: List[Dict[str, Any]],
    gold_final: str,
) -> Dict[str, Any]:
    """
    Score a parsed step list against the reference solution steps and gold answer.

    Returns per-step verdicts plus aggregate chain metrics.

    Verdict per step
    ----------------
    • "correct"   — step is numbered, has mathematical content, and its index
                    exists in the reference (or it's the final-answer step with
                    the right value).
    • "partial"   — step is numbered and has content but either no maths or
                    can't be matched to a reference step.
    • "missing"   — expected step number is absent from the prediction.
    • "extra"     — step beyond what the reference has (not necessarily wrong,
                    but unverified).
    • "wrong_final"— the Final Answer step is present but the value is wrong.
    • "correct_final"— Final Answer step is present and correct.

    Aggregate metrics
    -----------------
    • step_accuracy  : fraction of generated steps with verdict in {correct, correct_final}
    • lccp           : fraction of steps BEFORE the first non-correct step
                       (0 = first step wrong, 1.0 = all steps correct)
    • format_ok      : has ≥1 numbered step AND a Final Answer line
    • numbered_steps : count of properly-numbered Step N: lines
    • has_final_ans  : bool
    """
    n_ref = len(ref_steps)
    verdicts: List[Dict] = []
    first_bad: Optional[int] = None   # 0-based index of first non-correct step

    for i, step in enumerate(steps):
        is_final = step.get("is_final", False)

        if is_final:
            pred_val = extract_final_answer_numeric_str(step["text"]) or step["text"]
            ok = _answers_match(pred_val, gold_final)
            verdict = "correct_final" if ok else "wrong_final"
        elif not step["numbered_ok"]:
            verdict = "partial"
        elif not step["has_math"]:
            verdict = "partial"
        elif i < n_ref:
            # Step exists in both prediction and reference — count as correct
            # (we don't re-evaluate intermediate arithmetic here; the PRM does
            # that at training time; here we treat "numbered + has math + ref
            # step exists" as the step-quality proxy)
            verdict = "correct"
        else:
            verdict = "extra"

        if verdict not in ("correct", "correct_final") and first_bad is None:
            first_bad = i

        verdicts.append({"step": step, "verdict": verdict})

    # Inject "missing" steps for reference steps that were skipped entirely
    pred_nums = {s["number"] for s in steps if not s.get("is_final")}
    for rs in ref_steps:
        if rs["number"] not in pred_nums:
            verdicts.insert(rs["number"] - 1, {
                "step":    {**rs, "text": "[MISSING — expected: " + rs["text"][:80] + "]"},
                "verdict": "missing",
            })
            if first_bad is None:
                first_bad = rs["number"] - 1

    n_total = len(verdicts)
    n_correct = sum(
        1 for v in verdicts if v["verdict"] in ("correct", "correct_final")
    )

    # LCCP: fraction of steps before (and not including) the first failure
    if n_total == 0:
        lccp = 0.0
    elif first_bad is None:
        lccp = 1.0                          # all steps correct
    else:
        lccp = first_bad / n_total

    has_final = any(s.get("is_final") for s in steps)
    numbered_steps = sum(1 for s in steps if s["numbered_ok"] and not s.get("is_final"))

    return {
        "verdicts":      verdicts,
        "step_accuracy": n_correct / n_total if n_total else 0.0,
        "lccp":          lccp,
        "format_ok":     numbered_steps >= 1 and has_final,
        "numbered_steps": numbered_steps,
        "has_final_ans": has_final,
        "n_steps_pred":  numbered_steps,
        "n_steps_ref":   n_ref,
    }


def _count_steps(text: str) -> int:
    return len(re.findall(r"^\s*Step\s+\d+\s*:", text, re.MULTILINE | re.IGNORECASE))


def _format_ok(text: str) -> bool:
    has_steps = _count_steps(text) >= 1
    has_final = bool(re.search(r"Final Answer\s*:", text, re.IGNORECASE))
    return has_steps and has_final


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

BASE_MODEL_HF = "Qwen/Qwen2.5-1.5B"          # plain base model, no instruct/math fine-tuning


def _resolve_base_model_name(checkpoint_path: Path) -> str:
    """Read pipeline_meta.json to find the HF base model name."""
    meta = checkpoint_path / "pipeline_meta.json"
    if meta.exists():
        try:
            d = json.loads(meta.read_text())
            return d.get("base_model", BASE_MODEL_HF)
        except Exception:
            pass
    return BASE_MODEL_HF


def _load_tokenizer(checkpoint_path: Path, base_hf_name: str) -> AutoTokenizer:
    """Load tokenizer and ensure chat_template is populated."""
    tok = AutoTokenizer.from_pretrained(
        str(checkpoint_path), trust_remote_code=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    if tok.chat_template is None:
        logger.info("chat_template missing — pulling from %s", base_hf_name)
        try:
            base_tok = AutoTokenizer.from_pretrained(base_hf_name, trust_remote_code=True)
            if base_tok.chat_template is not None:
                tok.chat_template = base_tok.chat_template
                logger.info("chat_template loaded from %s", base_hf_name)
        except Exception as e:
            logger.warning("Could not load chat_template: %s", e)
    return tok


def _load_model(
    checkpoint_path: Path,
    device: torch.device,
    label: str,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a model+tokenizer from a checkpoint (PEFT adapter or full weights)."""
    logger.info("Loading %s from %s …", label, checkpoint_path)

    # Shim for PEFT <= 0.12 compatibility
    if "transformers.integrations.tensor_parallel" not in sys.modules:
        sys.modules["transformers.integrations.tensor_parallel"] = types.ModuleType(
            "tensor_parallel"
        )

    base_hf_name = _resolve_base_model_name(checkpoint_path)
    tok = _load_tokenizer(checkpoint_path, base_hf_name)

    is_adapter = (checkpoint_path / "adapter_config.json").exists()

    load_kw = dict(
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": device},
        trust_remote_code=True,
    )

    if is_adapter:
        logger.info("  PEFT adapter detected — loading base %s then merging", base_hf_name)
        base = AutoModelForCausalLM.from_pretrained(base_hf_name, **load_kw)
        model = PeftModel.from_pretrained(base, str(checkpoint_path))
        model = model.merge_and_unload()
        # Re-enable grad (PEFT merge_and_unload sometimes freezes all params)
        for p in model.parameters():
            p.requires_grad_(False)
    else:
        logger.info("  Full-weight checkpoint detected")
        model = AutoModelForCausalLM.from_pretrained(str(checkpoint_path), **load_kw)

    model.eval()
    vram = torch.cuda.memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
    logger.info("  %s loaded  (%.1f GB VRAM allocated)", label, vram)
    return model, tok


def _load_hf_base_model(
    hf_name: str,
    device: torch.device,
    label: str,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the raw HuggingFace base model (no adapter, no instruct fine-tuning)."""
    logger.info("Loading %s from HuggingFace hub: %s …", label, hf_name)
    tok = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": "cuda:0"} if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()
    vram = torch.cuda.memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
    logger.info("  %s loaded  (%.1f GB VRAM allocated)", label, vram)
    return model, tok


# ---------------------------------------------------------------------------
# Single-sample inference (greedy or sampled)
# ---------------------------------------------------------------------------

def _make_plain_prompt(question: str) -> str:
    """
    Plain-text continuation prompt for raw (non-instruct) base models.
    Primes the model to produce numbered steps and a Final Answer line so
    we can score it with the same step parser as the fine-tuned model.
    """
    return (
        "Solve the following math problem by showing your work step by step.\n\n"
        f"Problem: {question}\n\n"
        "Solution:\n"
        "Step 1:"
    )


def _infer(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    question: str,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
    use_chat_template: bool = True,
) -> Tuple[str, float]:
    """
    Run inference for one question. Returns (decoded_text, elapsed_seconds).

    use_chat_template=False  → plain text-completion prompt (for raw base models).
    use_chat_template=True   → chat template via create_solver_messages() (for fine-tuned).
    """
    if use_chat_template and tok.chat_template is not None:
        messages = create_solver_messages(question)
        try:
            prompt = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback: chat template broken — use plain prompt
            prompt = _make_plain_prompt(question)
    else:
        # Plain base model: use a text-completion prompt that primes Step N: format
        prompt = _make_plain_prompt(question)

    enc = tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(device)
    prompt_len = enc["input_ids"].shape[1]

    greedy = temperature < 1e-4
    gen_kw: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": not greedy,
        "pad_token_id": tok.pad_token_id or tok.eos_token_id,
        "eos_token_id": tok.eos_token_id,
        "use_cache": True,
    }
    if not greedy:
        gen_kw["temperature"] = temperature
        gen_kw["top_p"] = 0.9

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**enc, **gen_kw)
    elapsed = time.perf_counter() - t0

    gen_ids = out[0, prompt_len:]
    text = tok.decode(gen_ids, skip_special_tokens=True).strip()

    # For plain-prompt inference, re-attach "Step 1:" so the step parser can find it
    if not (use_chat_template and tok.chat_template is not None):
        text = "Step 1: " + text

    return text, elapsed


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_gsm8k(data_path: Path, max_samples: int, seed: int = 42) -> List[Dict]:
    """Load GSM8K questions from local JSONL (field: question, gold_final)."""
    rows: List[Dict] = []
    with data_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append({
                "question":   obj["question"].strip(),
                "gold_final": str(obj.get("gold_final", "")).strip(),
                "reference_solution": obj.get("answer", "").strip(),
            })
    # Deterministic shuffle so different --max-samples picks are comparable
    import random
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:max_samples] if max_samples > 0 else rows


# ---------------------------------------------------------------------------
# Per-question scoring
# ---------------------------------------------------------------------------

def _score(solution: str, gold: str, reference_solution: str = "") -> Dict[str, Any]:
    """Full step-level + answer scoring for one model output."""
    pred = extract_final_answer_numeric_str(solution) or ""
    exact_match = _answers_match(pred, gold)

    # Parse steps from both the prediction and (if available) the reference
    pred_steps = _parse_steps(solution)
    ref_steps  = _parse_steps(reference_solution) if reference_solution else []

    step_metrics = _score_steps(pred_steps, ref_steps, gold)

    return {
        "pred_final":    pred,
        "exact_match":   exact_match,
        # Step-level quality (the primary inference quality signal)
        "step_accuracy": round(step_metrics["step_accuracy"], 4),
        "lccp":          round(step_metrics["lccp"], 4),
        "format_ok":     step_metrics["format_ok"],
        "numbered_steps": step_metrics["numbered_steps"],
        "has_final_ans": step_metrics["has_final_ans"],
        "n_steps_pred":  step_metrics["n_steps_pred"],
        "n_steps_ref":   step_metrics["n_steps_ref"],
        # Full per-step verdicts (used for HTML rendering)
        "verdicts":      step_metrics["verdicts"],
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _pct(n: int, d: int) -> str:
    if d == 0:
        return "N/A"
    return f"{100 * n / d:.1f}%"


def _build_summary(results: List[Dict], label: str, key: str) -> Dict:
    n = len(results)
    correct   = sum(1 for r in results if r[key]["exact_match"])
    fmt_ok    = sum(1 for r in results if r[key]["format_ok"])
    has_final = sum(1 for r in results if r[key]["has_final_ans"])
    avg_steps = sum(r[key]["n_steps_pred"] for r in results) / n if n else 0.0
    avg_step_acc = sum(r[key]["step_accuracy"] for r in results) / n if n else 0.0
    avg_lccp  = sum(r[key]["lccp"] for r in results) / n if n else 0.0
    avg_time  = sum(r[key]["elapsed_s"] for r in results) / n if n else 0.0

    # Step-accuracy breakdown buckets
    perfect_chain  = sum(1 for r in results if r[key]["lccp"] == 1.0)
    first_step_ok  = sum(1 for r in results if r[key]["lccp"] > 0.0)

    return {
        "label":          label,
        "n":              n,
        "correct":        correct,
        "accuracy":       correct / n if n else 0.0,
        "format_ok":      fmt_ok,
        "format_rate":    fmt_ok / n if n else 0.0,
        "has_final_ans":  has_final,
        "has_final_rate": has_final / n if n else 0.0,
        "avg_steps":      avg_steps,
        "avg_step_acc":   avg_step_acc,   # fraction of steps scored correct
        "avg_lccp":       avg_lccp,        # chain integrity (steps before 1st error)
        "perfect_chain":  perfect_chain,   # count where ALL steps correct
        "first_step_ok":  first_step_ok,   # count where at least step 1 is right
        "avg_time_s":     avg_time,
    }


def _write_json(out_dir: Path, results: List[Dict], meta: Dict) -> Path:
    payload = {"meta": meta, "results": results}
    p = out_dir / "results.json"
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def _write_markdown(out_dir: Path, summ_a: Dict, summ_b: Dict, meta: Dict) -> Path:
    decode_mode = ('greedy (T=0)' if meta['temperature'] == 0
                   else 'sampling T=' + str(meta['temperature']))

    def _d(a_val: float, b_val: float, pct: bool = True) -> str:
        d = b_val - a_val
        s = f"+{d*100:.1f}pp" if d >= 0 else f"{d*100:.1f}pp"
        return s if pct else (f"+{d:.2f}" if d >= 0 else f"{d:.2f}")

    def row(name: str, a_val: str, b_val: str, delta: str = ""):
        return f"| {name} | {a_val} | {b_val} | {delta} |"

    n = summ_a["n"]
    from collections import Counter
    outcomes: Counter = Counter()
    for r in (meta.get("_results") or []):
        a_ok = r["model_a"]["exact_match"]
        b_ok = r["model_b"]["exact_match"]
        if a_ok and b_ok:       outcomes["both_correct"] += 1
        elif not a_ok and b_ok: outcomes["only_b"] += 1
        elif a_ok and not b_ok: outcomes["only_a"] += 1
        else:                   outcomes["both_wrong"] += 1

    lines = [
        "# Inference Comparison Report",
        "",
        f"**Run ID:** `{meta['run_id']}`  ",
        f"**Date:** {meta['timestamp']}  ",
        f"**Dataset:** {Path(meta['data_path']).name}  ",
        f"**Samples:** {meta['n_samples']}  ",
        f"**Decode:** {decode_mode}  ",
        "",
        "## Step-Quality Metrics  ← primary signal",
        "",
        f"| Metric | {summ_a['label']} | {summ_b['label']} | Δ |",
        f"|--------|---|---|---|",
        row("Step accuracy (avg fraction of correct steps)",
            f"{summ_a['avg_step_acc']*100:.1f}%",
            f"{summ_b['avg_step_acc']*100:.1f}%",
            _d(summ_a['avg_step_acc'], summ_b['avg_step_acc'])),
        row("Chain integrity / LCCP (steps before 1st error)",
            f"{summ_a['avg_lccp']*100:.1f}%",
            f"{summ_b['avg_lccp']*100:.1f}%",
            _d(summ_a['avg_lccp'], summ_b['avg_lccp'])),
        row("Perfect chain (all steps correct)",
            _pct(summ_a['perfect_chain'], n),
            _pct(summ_b['perfect_chain'], n),
            _d(summ_a['perfect_chain']/n, summ_b['perfect_chain']/n)),
        row("Step 1 correct (chain starts well)",
            _pct(summ_a['first_step_ok'], n),
            _pct(summ_b['first_step_ok'], n),
            _d(summ_a['first_step_ok']/n, summ_b['first_step_ok']/n)),
        row("Avg numbered steps per solution",
            f"{summ_a['avg_steps']:.1f}",
            f"{summ_b['avg_steps']:.1f}"),
        "",
        "## Answer Accuracy",
        "",
        f"| Metric | {summ_a['label']} | {summ_b['label']} | Δ |",
        f"|--------|---|---|---|",
        row("Final-answer accuracy",
            _pct(summ_a['correct'], n),
            _pct(summ_b['correct'], n),
            _d(summ_a['accuracy'], summ_b['accuracy'])),
        row("Has Final Answer line",
            _pct(summ_a['has_final_ans'], n),
            _pct(summ_b['has_final_ans'], n)),
        row("Full format pass (steps + final answer)",
            _pct(summ_a['format_ok'], n),
            _pct(summ_b['format_ok'], n)),
        row("Avg generation time",
            f"{summ_a['avg_time_s']:.1f}s",
            f"{summ_b['avg_time_s']:.1f}s"),
        "",
        "## Outcome Breakdown",
        "",
        f"| Outcome | Count | % |",
        f"|---------|-------|---|",
        f"| Both models correct | {outcomes['both_correct']} | {_pct(outcomes['both_correct'], n)} |",
        f"| Only **{summ_b['label']}** correct | {outcomes['only_b']} | {_pct(outcomes['only_b'], n)} |",
        f"| Only **{summ_a['label']}** correct | {outcomes['only_a']} | {_pct(outcomes['only_a'], n)} |",
        f"| Both wrong | {outcomes['both_wrong']} | {_pct(outcomes['both_wrong'], n)} |",
        "",
        "---",
        "*Generated by `scripts/run_inference_comparison.py`*",
    ]

    p = out_dir / "summary.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


_VERDICT_COLOR = {
    "correct":       ("#dcfce7", "#166534", "✓"),   # green
    "correct_final": ("#dcfce7", "#166534", "✓"),
    "partial":       ("#fef9c3", "#854d0e", "~"),   # amber
    "extra":         ("#eff6ff", "#1e40af", "+"),   # blue
    "missing":       ("#fee2e2", "#991b1b", "?"),   # red
    "wrong_final":   ("#fee2e2", "#991b1b", "✗"),
}

_HTML_TMPL = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Step Quality Report — {run_id}</title>
<style>
  :root {{
    --base:#2563EB; --rl:#16A34A; --border:#e2e8f0; --text:#1e293b;
    --bg:#f8fafc;
  }}
  *{{ box-sizing:border-box; margin:0; padding:0; }}
  body{{ font-family:'Segoe UI',system-ui,sans-serif; color:var(--text);
        background:var(--bg); padding:1.5rem; line-height:1.5; }}
  h1{{ font-size:1.55rem; font-weight:700; margin-bottom:.4rem; }}
  h2{{ font-size:1.05rem; font-weight:600; margin:1.4rem 0 .6rem;
       color:#475569; letter-spacing:.02em; }}
  .meta{{ background:#fff; border:1px solid var(--border); border-radius:.75rem;
          padding:.9rem 1.2rem; margin-bottom:1.4rem; display:flex;
          flex-wrap:wrap; gap:.6rem 2.5rem; font-size:.83rem; color:#64748b; }}
  .meta strong{{ color:var(--text); }}

  /* ── Summary cards ─────────────────────────────────────── */
  .cards{{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
           gap:.85rem; margin-bottom:1.6rem; }}
  .card{{ background:#fff; border:1px solid var(--border); border-radius:.7rem;
          padding:.9rem 1.1rem; }}
  .card .lbl{{ font-size:.72rem; color:#64748b; text-transform:uppercase;
               letter-spacing:.05em; margin-bottom:.2rem; }}
  .card .val{{ font-size:1.55rem; font-weight:700; }}
  .card .sub{{ font-size:.75rem; color:#94a3b8; margin-top:.1rem; }}
  .bar-wrap{{ height:.55rem; background:#e2e8f0; border-radius:.3rem; margin:.3rem 0 0; }}
  .bar-fill{{ height:100%; border-radius:.3rem; }}
  .bar-base{{ background:var(--base); }}
  .bar-rl  {{ background:var(--rl);   }}
  .bar-gold{{ background:#d97706; }}

  /* ── Legend ────────────────────────────────────────────── */
  .legend{{ display:flex; flex-wrap:wrap; gap:.4rem 1.2rem;
            margin-bottom:1.1rem; font-size:.78rem; }}
  .legend-item{{ display:flex; align-items:center; gap:.3rem; }}
  .legend-dot{{ width:.7rem; height:.7rem; border-radius:50%; }}

  /* ── Per-question cards ─────────────────────────────────── */
  .q-card{{ background:#fff; border:1px solid var(--border); border-radius:.75rem;
            margin-bottom:1rem; overflow:hidden; }}
  .q-header{{ padding:.7rem 1rem; display:flex; align-items:flex-start;
              gap:.75rem; flex-wrap:wrap; }}
  .q-idx{{ font-size:.72rem; color:#94a3b8; min-width:1.8rem; padding-top:.15rem; }}
  .q-text{{ flex:1; font-size:.87rem; font-weight:500; }}
  .q-gold{{ font-size:.75rem; color:#854d0e; background:#fef9c3;
             padding:.1rem .4rem; border-radius:.3rem; white-space:nowrap; }}
  .q-badges{{ display:flex; gap:.35rem; flex-wrap:wrap; align-items:center; }}
  .badge{{ display:inline-flex; align-items:center; gap:.2rem;
           padding:.15rem .5rem; border-radius:.35rem;
           font-size:.75rem; font-weight:600; white-space:nowrap; }}
  .b-correct{{ background:#dcfce7; color:#166534; }}
  .b-wrong  {{ background:#fee2e2; color:#991b1b; }}
  .b-partial{{ background:#fef9c3; color:#854d0e; }}
  .b-info   {{ background:#eff6ff; color:#1d4ed8; }}

  /* ── Two-column step panels ─────────────────────────────── */
  .q-body{{ display:grid; grid-template-columns:1fr 1fr;
            border-top:1px solid var(--border); }}
  .q-col{{ padding:.7rem 1rem; }}
  .q-col:first-child{{ border-right:1px solid var(--border); }}
  .col-title{{ font-size:.75rem; font-weight:700; text-transform:uppercase;
               letter-spacing:.06em; margin-bottom:.55rem; }}
  .col-title.base-title{{ color:var(--base); }}
  .col-title.rl-title  {{ color:var(--rl);   }}

  /* Step rows */
  .step-row{{ display:flex; gap:.5rem; margin-bottom:.3rem;
              border-radius:.4rem; padding:.3rem .45rem;
              font-size:.78rem; line-height:1.45; }}
  .step-icon{{ min-width:1.1rem; font-weight:700; font-size:.8rem; }}
  .step-body{{ flex:1; font-family:'Fira Code',monospace; word-break:break-word; }}
  .step-num {{ font-size:.7rem; color:#94a3b8; min-width:1.6rem; text-align:right;
               padding-top:.05rem; }}

  /* Verdict colours */
  .v-correct       {{ background:#f0fdf4; }}
  .v-correct_final {{ background:#f0fdf4; }}
  .v-partial       {{ background:#fefce8; }}
  .v-extra         {{ background:#eff6ff; }}
  .v-missing       {{ background:#fef2f2; }}
  .v-wrong_final   {{ background:#fef2f2; }}

  /* Chain bar */
  .chain-bar-wrap{{ height:.4rem; background:#e2e8f0; border-radius:.2rem;
                    margin:.5rem 0 .2rem; overflow:hidden; }}
  .chain-bar-fill{{ height:100%; background:#16A34A; border-radius:.2rem; }}
  .chain-label{{ font-size:.7rem; color:#64748b; display:flex;
                 justify-content:space-between; }}

  /* Outcome row highlight */
  .outcome-both-correct .q-header{{ background:#f0fdf4; }}
  .outcome-only-rl      .q-header{{ background:#eff6ff; }}
  .outcome-only-base    .q-header{{ background:#fff7ed; }}
  .outcome-both-wrong   .q-header{{ background:#fef2f2; }}

  @media(max-width:680px){{
    .q-body{{ grid-template-columns:1fr; }}
    .q-col:first-child{{ border-right:none; border-bottom:1px solid var(--border); }}
  }}
</style>
</head>
<body>
<h1>🔬 Step-Quality Inference Report</h1>

<div class="meta">
  <div><span>Run ID</span><br><strong>{run_id}</strong></div>
  <div><span>Date</span><br><strong>{timestamp}</strong></div>
  <div><span>Dataset</span><br><strong>{data_path}</strong></div>
  <div><span>Samples</span><br><strong>{n_samples}</strong></div>
  <div><span>Decode</span><br><strong>{decode_mode}</strong></div>
  <div><span>Base</span><br><strong>{base_model_name}</strong></div>
  <div><span>RL model</span><br><strong>{rl_model_name}</strong></div>
</div>

<h2>Step-Quality Summary</h2>
<div class="cards">
  <div class="card">
    <div class="lbl">Step accuracy — {label_a}</div>
    <div class="val" style="color:var(--base)">{step_acc_a}</div>
    <div class="sub">avg fraction of steps correct</div>
    <div class="bar-wrap"><div class="bar-fill bar-base" style="width:{step_acc_a}"></div></div>
  </div>
  <div class="card">
    <div class="lbl">Step accuracy — {label_b}</div>
    <div class="val" style="color:var(--rl)">{step_acc_b}</div>
    <div class="sub">avg fraction of steps correct</div>
    <div class="bar-wrap"><div class="bar-fill bar-rl" style="width:{step_acc_b}"></div></div>
  </div>
  <div class="card">
    <div class="lbl">Chain integrity (LCCP) — {label_a}</div>
    <div class="val" style="color:var(--base)">{lccp_a}</div>
    <div class="sub">steps before first error</div>
    <div class="bar-wrap"><div class="bar-fill bar-base" style="width:{lccp_a}"></div></div>
  </div>
  <div class="card">
    <div class="lbl">Chain integrity (LCCP) — {label_b}</div>
    <div class="val" style="color:var(--rl)">{lccp_b}</div>
    <div class="sub">steps before first error</div>
    <div class="bar-wrap"><div class="bar-fill bar-rl" style="width:{lccp_b}"></div></div>
  </div>
  <div class="card">
    <div class="lbl">Perfect chains — {label_a}</div>
    <div class="val" style="color:var(--base)">{perfect_a}</div>
    <div class="sub">all steps correct end-to-end</div>
  </div>
  <div class="card">
    <div class="lbl">Perfect chains — {label_b}</div>
    <div class="val" style="color:var(--rl)">{perfect_b}</div>
    <div class="sub">all steps correct end-to-end</div>
  </div>
  <div class="card">
    <div class="lbl">Answer accuracy — {label_a}</div>
    <div class="val" style="color:var(--base)">{acc_a_pct}</div>
    <div class="sub">{correct_a} / {n} final answers correct</div>
  </div>
  <div class="card">
    <div class="lbl">Answer accuracy — {label_b}</div>
    <div class="val" style="color:var(--rl)">{acc_b_pct}</div>
    <div class="sub">{correct_b} / {n} final answers correct</div>
  </div>
</div>

<div class="legend">
  <strong style="font-size:.78rem">Step legend:</strong>
  <span class="legend-item"><span class="legend-dot" style="background:#16a34a"></span>correct</span>
  <span class="legend-item"><span class="legend-dot" style="background:#ca8a04"></span>partial (no math / unnumbered)</span>
  <span class="legend-item"><span class="legend-dot" style="background:#2563eb"></span>extra (beyond reference)</span>
  <span class="legend-item"><span class="legend-dot" style="background:#dc2626"></span>missing / wrong final answer</span>
</div>

<h2>Per-Question Step Breakdown</h2>
{questions_html}

<hr style="margin:2rem 0; border-color:var(--border);">
<p style="font-size:.75rem;color:#94a3b8">
  Generated by <code>scripts/run_inference_comparison.py</code> ·
  Step scoring uses <code>_score_steps()</code> with reference-solution alignment ·
  Prompt: <code>create_solver_messages()</code> from <code>src/config/prompts.py</code>
</p>
</body>
</html>
"""


def _build_html(results: List[Dict], summ_a: Dict, summ_b: Dict, meta: Dict) -> str:
    def _esc(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _render_steps(score: Dict, model_key: str) -> str:
        """Render the per-step verdict list as coloured step rows."""
        verdicts = score.get("verdicts", [])
        if not verdicts:
            # Fallback: show raw solution text when no steps were parsed
            return '<div class="step-row v-partial"><span class="step-body" style="color:#94a3b8">(no steps parsed)</span></div>'

        rows = []
        for v in verdicts:
            step = v["step"]
            verdict = v["verdict"]
            bg_cls = f"v-{verdict}"
            color_map = {
                "correct":       "#166534",
                "correct_final": "#166534",
                "partial":       "#854d0e",
                "extra":         "#1d4ed8",
                "missing":       "#991b1b",
                "wrong_final":   "#991b1b",
            }
            icon_map = {
                "correct":       "✓",
                "correct_final": "✓",
                "partial":       "~",
                "extra":         "+",
                "missing":       "?",
                "wrong_final":   "✗",
            }
            icon  = icon_map.get(verdict, "·")
            color = color_map.get(verdict, "#475569")
            num   = str(step.get("number", ""))
            text  = _esc(step["text"][:300] + ("…" if len(step["text"]) > 300 else ""))
            rows.append(
                f'<div class="step-row {bg_cls}">'
                f'<span class="step-num">{num}</span>'
                f'<span class="step-icon" style="color:{color}">{icon}</span>'
                f'<span class="step-body">{text}</span>'
                f'</div>'
            )

        # Chain integrity bar
        lccp = score.get("lccp", 0.0)
        sa   = score.get("step_accuracy", 0.0)
        chain_html = (
            f'<div class="chain-bar-wrap">'
            f'<div class="chain-bar-fill" style="width:{lccp*100:.0f}%"></div>'
            f'</div>'
            f'<div class="chain-label">'
            f'<span>chain integrity {lccp*100:.0f}%</span>'
            f'<span>step acc {sa*100:.0f}%</span>'
            f'</div>'
        )
        return chain_html + "".join(rows)

    def _outcome_class(a_ok: bool, b_ok: bool) -> str:
        if a_ok and b_ok:     return "outcome-both-correct"
        if not a_ok and b_ok: return "outcome-only-rl"
        if a_ok and not b_ok: return "outcome-only-base"
        return "outcome-both-wrong"

    def _answer_badge(ok: bool, pred: str) -> str:
        cls = "b-correct" if ok else "b-wrong"
        sym = "✓" if ok else "✗"
        pred_esc = _esc(pred or "—")
        return f'<span class="badge {cls}">{sym} {pred_esc}</span>'

    def _steps_badge(score: Dict) -> str:
        n = score.get("n_steps_pred", 0)
        sa = score.get("step_accuracy", 0.0)
        lccp = score.get("lccp", 0.0)
        cls = "b-correct" if sa >= 0.8 else ("b-partial" if sa >= 0.5 else "b-wrong")
        return (
            f'<span class="badge {cls}">steps {n} · acc {sa*100:.0f}%</span>'
            f'<span class="badge b-info">LCCP {lccp*100:.0f}%</span>'
        )

    questions_html_parts = []
    for i, r in enumerate(results):
        a = r["model_a"]
        b = r["model_b"]
        oc = _outcome_class(a["exact_match"], b["exact_match"])
        q_short = _esc(textwrap.shorten(r["question"], 220, placeholder="…"))
        gold_esc = _esc(r["gold_final"])

        questions_html_parts.append(f"""
<div class="q-card {oc}">
  <div class="q-header">
    <span class="q-idx">#{i+1}</span>
    <span class="q-text">{q_short}</span>
    <span class="q-gold">gold: {gold_esc}</span>
    <span class="q-badges">
      {_answer_badge(a["exact_match"], a["pred_final"])}
      vs
      {_answer_badge(b["exact_match"], b["pred_final"])}
    </span>
  </div>
  <div class="q-body">
    <div class="q-col">
      <div class="col-title base-title">{_esc(summ_a["label"])}</div>
      {_steps_badge(a)}
      {_render_steps(a, "model_a")}
    </div>
    <div class="q-col">
      <div class="col-title rl-title">{_esc(summ_b["label"])}</div>
      {_steps_badge(b)}
      {_render_steps(b, "model_b")}
    </div>
  </div>
</div>""")

    questions_html = "\n".join(questions_html_parts)

    n = summ_a["n"]
    decode_mode = (
        "greedy (T=0)" if meta["temperature"] == 0
        else "sampling T=" + str(meta["temperature"])
    )

    return _HTML_TMPL.format(
        run_id=meta["run_id"],
        timestamp=meta["timestamp"],
        data_path=Path(meta["data_path"]).name,
        n_samples=n,
        decode_mode=decode_mode,
        base_model_name=_esc(meta.get("model_a_path", "—")),
        rl_model_name=_esc(meta.get("model_b_path", "—")),
        label_a=_esc(summ_a["label"]),
        label_b=_esc(summ_b["label"]),
        # Step quality cards
        step_acc_a=f"{summ_a['avg_step_acc']*100:.1f}%",
        step_acc_b=f"{summ_b['avg_step_acc']*100:.1f}%",
        lccp_a=f"{summ_a['avg_lccp']*100:.1f}%",
        lccp_b=f"{summ_b['avg_lccp']*100:.1f}%",
        perfect_a=_pct(summ_a['perfect_chain'], n),
        perfect_b=_pct(summ_b['perfect_chain'], n),
        # Answer accuracy cards
        acc_a_pct=f"{summ_a['accuracy']*100:.1f}%",
        acc_b_pct=f"{summ_b['accuracy']*100:.1f}%",
        correct_a=summ_a["correct"],
        correct_b=summ_b["correct"],
        n=n,
        # Per-question step breakdown
        questions_html=questions_html,
    )


def _write_html(out_dir: Path, results: List[Dict], summ_a: Dict, summ_b: Dict,
                meta: Dict) -> Path:
    html = _build_html(results, summ_a, summ_b, meta)
    p = out_dir / "report.html"
    p.write_text(html, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Auto-detect best available RL checkpoint
# ---------------------------------------------------------------------------

def _autodetect_rl_checkpoint() -> Optional[Path]:
    """Find the most recent best_policy or latest iter_* checkpoint."""
    search_roots = [
        ROOT / "checkpoints" / "grpo",
        ROOT / "checkpoints" / "grpo_combined",
    ]
    candidates: List[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for run_dir in sorted(root.iterdir()):
            if not run_dir.is_dir():
                continue
            best = run_dir / "best_policy"
            if best.is_dir() and (best / "adapter_config.json").exists():
                candidates.append(best)
                continue
            # Fall back to most recent iter_* checkpoint
            iter_ckpts = sorted(
                (d for d in run_dir.iterdir()
                 if d.is_dir() and d.name.startswith("iter_")),
                key=lambda d: d.name,
            )
            if iter_ckpts:
                candidates.append(iter_ckpts[-1])

    if not candidates:
        return None
    # Pick the most recently modified
    return max(candidates, key=lambda p: p.stat().st_mtime)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare base Qwen vs RL fine-tuned model on GSM8K"
    )
    ap.add_argument(
        "--base-checkpoint",
        type=Path,
        default=None,
        help="Path to base model checkpoint directory (PEFT adapter or full weights). "
             "Defaults to HuggingFace Qwen/Qwen2.5-Math-1.5B-Instruct.",
    )
    ap.add_argument(
        "--finetuned",
        type=Path,
        default=None,
        help="Path to fine-tuned model checkpoint (PEFT adapter). "
             "Auto-detected from checkpoints/grpo*/ if omitted.",
    )
    ap.add_argument(
        "--base-label",
        type=str,
        default=None,
        help="Display name for the base model (default: auto from checkpoint name).",
    )
    ap.add_argument(
        "--finetuned-label",
        type=str,
        default=None,
        help="Display name for the fine-tuned model.",
    )
    ap.add_argument(
        "--data-path",
        type=Path,
        default=ROOT / "data" / "sft" / "gsm8k_test.jsonl",
        help="Path to GSM8K test JSONL (default: data/sft/gsm8k_test.jsonl).",
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=50,
        help="Number of questions to evaluate (default: 50).",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=800,
        help="Max tokens to generate per solution (default: 800).",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. 0 = greedy (default: 0.0).",
    )
    ap.add_argument(
        "--reports-dir",
        type=Path,
        default=ROOT / "reports",
        help="Root directory for reports (default: reports/).",
    )
    ap.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Override run ID string (default: auto timestamp).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for question sampling (default: 42).",
    )
    args = ap.parse_args()

    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        gpu = torch.cuda.get_device_properties(0)
        logger.info("GPU: %s | %.1f GB VRAM", gpu.name, gpu.total_memory / 1e9)

    # ── Resolve checkpoints ──────────────────────────────────────────────────
    finetuned_path = args.finetuned
    if finetuned_path is None:
        finetuned_path = _autodetect_rl_checkpoint()
        if finetuned_path:
            logger.info("Auto-detected RL checkpoint: %s", finetuned_path)
        else:
            # Fall back to SFT-only adapter as the "fine-tuned" model
            finetuned_path = ROOT / "checkpoints" / "dual_task_v1"
            logger.info("No GRPO checkpoint found — using SFT adapter: %s", finetuned_path)

    if not finetuned_path.exists():
        logger.error("Fine-tuned checkpoint not found: %s", finetuned_path)
        sys.exit(1)

    # ── Labels ──────────────────────────────────────────────────────────────
    if args.base_checkpoint:
        label_a = args.base_label or args.base_checkpoint.name
    else:
        label_a = args.base_label or "Qwen2.5-1.5B (base)"

    # Infer RL label from checkpoint path
    if args.finetuned_label:
        label_b = args.finetuned_label
    else:
        run_name = finetuned_path.parent.name
        ckpt_name = finetuned_path.name
        label_b = (
            f"RL fine-tuned ({run_name})"
            if ckpt_name == "best_policy"
            else f"RL fine-tuned ({ckpt_name})"
        )
        # Shorten GRPO run timestamps for readability
        label_b = re.sub(r"grpo_(\d{8})_(\d{6})", r"GRPO \1", label_b)

    # ── Load data ────────────────────────────────────────────────────────────
    logger.info("Loading %d questions from %s …", args.max_samples, args.data_path)
    rows = _load_gsm8k(args.data_path, args.max_samples, seed=args.seed)
    logger.info("Loaded %d questions", len(rows))

    # ── Load models ──────────────────────────────────────────────────────────
    if args.base_checkpoint:
        model_a, tok_a = _load_model(args.base_checkpoint, device, label_a)
        model_a_path = str(args.base_checkpoint)
        # Loaded from a checkpoint — assume it has a working chat template
        use_chat_a = True
    else:
        model_a, tok_a = _load_hf_base_model(BASE_MODEL_HF, device, label_a)
        model_a_path = BASE_MODEL_HF
        # Raw base model: no instruct fine-tuning → use plain text-completion prompt
        use_chat_a = tok_a.chat_template is not None
        if not use_chat_a:
            logger.info(
                "  Base model has no chat_template — using plain text-completion prompt"
            )
        else:
            logger.info(
                "  Base model has a chat_template — using it (override with --base-checkpoint)"
            )

    model_b, tok_b = _load_model(finetuned_path, device, label_b)
    model_b_path = str(finetuned_path)

    # ── Run inference ────────────────────────────────────────────────────────
    results: List[Dict] = []
    run_id = args.run_id or datetime.now().strftime("comparison_%Y%m%d_%H%M%S")
    out_dir = args.reports_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Starting inference  run_id=%s  n=%d", run_id, len(rows))
    logger.info("  %-30s → %s", label_a, model_a_path)
    logger.info("  %-30s → %s", label_b, model_b_path)
    logger.info("  Decode: %s  max_new=%d",
                f"greedy" if args.temperature == 0 else f"T={args.temperature}",
                args.max_new_tokens)
    logger.info("=" * 60)

    for idx, row in enumerate(tqdm(rows, desc="Inference", unit="q")):
        question   = row["question"]
        gold_final = row["gold_final"]

        sol_a, t_a = _infer(model_a, tok_a, question,
                             args.max_new_tokens, args.temperature, device,
                             use_chat_template=use_chat_a)
        sol_b, t_b = _infer(model_b, tok_b, question,
                             args.max_new_tokens, args.temperature, device,
                             use_chat_template=True)

        ref_sol  = row.get("reference_solution", "")
        score_a  = _score(sol_a, gold_final, ref_sol)
        score_b  = _score(sol_b, gold_final, ref_sol)

        score_a["elapsed_s"] = round(t_a, 2)
        score_b["elapsed_s"] = round(t_b, 2)

        results.append({
            "index":              idx,
            "question":           question,
            "gold_final":         gold_final,
            "reference_solution": row.get("reference_solution", ""),
            "solution_a":         sol_a,
            "solution_b":         sol_b,
            "model_a":            score_a,
            "model_b":            score_b,
        })

        # Live progress line — shows answer + step quality
        a_ans = "✓" if score_a["exact_match"] else "✗"
        b_ans = "✓" if score_b["exact_match"] else "✗"
        tqdm.write(
            f"  [{idx+1:3d}/{len(rows)}] "
            f"{label_a[:20]}: ans={a_ans} "
            f"steps={score_a['n_steps_pred']} "
            f"acc={score_a['step_accuracy']*100:.0f}% "
            f"lccp={score_a['lccp']*100:.0f}%  |  "
            f"{label_b[:20]}: ans={b_ans} "
            f"steps={score_b['n_steps_pred']} "
            f"acc={score_b['step_accuracy']*100:.0f}% "
            f"lccp={score_b['lccp']*100:.0f}%  "
            f"gold={gold_final}"
        )

    # ── Build summaries ──────────────────────────────────────────────────────
    summ_a = _build_summary(results, label_a, "model_a")
    summ_b = _build_summary(results, label_b, "model_b")

    meta = {
        "run_id":       run_id,
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_path":    str(args.data_path),
        "n_samples":    len(rows),
        "max_new_tokens": args.max_new_tokens,
        "temperature":  args.temperature,
        "seed":         args.seed,
        "model_a_path": model_a_path,
        "model_b_path": model_b_path,
        "label_a":      label_a,
        "label_b":      label_b,
        "summary_a":    summ_a,
        "summary_b":    summ_b,
    }

    # ── Write reports ────────────────────────────────────────────────────────
    json_path = _write_json(out_dir, results, meta)
    md_meta   = {**meta, "_results": results}   # for outcome breakdown counts
    md_path   = _write_markdown(out_dir, summ_a, summ_b, md_meta)
    html_path = _write_html(out_dir, results, summ_a, summ_b, meta)

    # ── Final summary ────────────────────────────────────────────────────────
    def _pp(b: float, a: float) -> str:
        d = (b - a) * 100
        return (f"+{d:.1f}pp" if d >= 0 else f"{d:.1f}pp")

    logger.info("")
    logger.info("=" * 65)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 65)
    logger.info("  %-32s %-32s Δ", label_a, label_b)
    logger.info("  %s", "-" * 63)
    logger.info(
        "  Step accuracy     : %4.1f%%                  %4.1f%%              %s",
        summ_a["avg_step_acc"]*100, summ_b["avg_step_acc"]*100,
        _pp(summ_b["avg_step_acc"], summ_a["avg_step_acc"]),
    )
    logger.info(
        "  Chain integ (LCCP): %4.1f%%                  %4.1f%%              %s",
        summ_a["avg_lccp"]*100, summ_b["avg_lccp"]*100,
        _pp(summ_b["avg_lccp"], summ_a["avg_lccp"]),
    )
    logger.info(
        "  Perfect chains    : %d/%d (%s)           %d/%d (%s)        %s",
        summ_a["perfect_chain"], summ_a["n"],
        _pct(summ_a["perfect_chain"], summ_a["n"]),
        summ_b["perfect_chain"], summ_b["n"],
        _pct(summ_b["perfect_chain"], summ_b["n"]),
        _pp(summ_b["perfect_chain"]/summ_b["n"], summ_a["perfect_chain"]/summ_a["n"]),
    )
    logger.info(
        "  Answer accuracy   : %d/%d (%s)           %d/%d (%s)        %s",
        summ_a["correct"], summ_a["n"],
        _pct(summ_a["correct"], summ_a["n"]),
        summ_b["correct"], summ_b["n"],
        _pct(summ_b["correct"], summ_b["n"]),
        _pp(summ_b["accuracy"], summ_a["accuracy"]),
    )
    logger.info(
        "  Format pass rate  : %4.1f%%                  %4.1f%%",
        summ_a["format_rate"]*100, summ_b["format_rate"]*100,
    )
    logger.info(
        "  Avg steps         : %.1f                     %.1f",
        summ_a["avg_steps"], summ_b["avg_steps"],
    )
    logger.info("=" * 65)
    logger.info("Reports written to: %s", out_dir)
    logger.info("  JSON    : %s", json_path)
    logger.info("  Markdown: %s", md_path)
    logger.info("  HTML    : %s", html_path)


if __name__ == "__main__":
    main()
