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


def _count_steps(text: str) -> int:
    return len(re.findall(r"^\s*Step\s+\d+\s*:", text, re.MULTILINE | re.IGNORECASE))


def _format_ok(text: str) -> bool:
    has_steps = _count_steps(text) >= 1
    has_final = bool(re.search(r"Final Answer\s*:", text, re.IGNORECASE))
    return has_steps and has_final


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

BASE_MODEL_HF = "Qwen/Qwen2.5-Math-1.5B-Instruct"


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
    """Load the raw HuggingFace base model (no adapter)."""
    logger.info("Loading %s from HuggingFace hub: %s …", label, hf_name)
    tok = AutoTokenizer.from_pretrained(hf_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        hf_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": device},
        trust_remote_code=True,
    )
    model.eval()
    vram = torch.cuda.memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
    logger.info("  %s loaded  (%.1f GB VRAM allocated)", label, vram)
    return model, tok


# ---------------------------------------------------------------------------
# Single-sample inference (greedy or sampled)
# ---------------------------------------------------------------------------

def _infer(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    question: str,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> Tuple[str, float]:
    """Run inference for one question. Returns (decoded_text, elapsed_seconds)."""
    messages = create_solver_messages(question)
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
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

def _score(solution: str, gold: str) -> Dict[str, Any]:
    pred = extract_final_answer_numeric_str(solution) or ""
    return {
        "pred_final":  pred,
        "exact_match": _answers_match(pred, gold),
        "format_ok":   _format_ok(solution),
        "step_count":  _count_steps(solution),
        "has_final_answer_line": bool(
            re.search(r"Final Answer\s*:", solution, re.IGNORECASE)
        ),
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
    correct = sum(1 for r in results if r[key]["exact_match"])
    fmt_ok  = sum(1 for r in results if r[key]["format_ok"])
    avg_steps = (
        sum(r[key]["step_count"] for r in results) / n if n else 0.0
    )
    avg_time = (
        sum(r[key]["elapsed_s"] for r in results) / n if n else 0.0
    )
    return {
        "label":        label,
        "n":            n,
        "correct":      correct,
        "accuracy":     correct / n if n else 0.0,
        "format_ok":    fmt_ok,
        "format_rate":  fmt_ok / n if n else 0.0,
        "avg_steps":    avg_steps,
        "avg_time_s":   avg_time,
    }


def _write_json(out_dir: Path, results: List[Dict], meta: Dict) -> Path:
    payload = {"meta": meta, "results": results}
    p = out_dir / "results.json"
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def _write_markdown(out_dir: Path, summ_a: Dict, summ_b: Dict, meta: Dict) -> Path:
    lines = [
        f"# Inference Comparison Report",
        f"",
        f"**Run ID:** `{meta['run_id']}`  ",
        f"**Date:** {meta['timestamp']}  ",
        f"**Dataset:** {meta['data_path']}  ",
        f"**Samples:** {meta['n_samples']}  ",
        f"**Decode mode:** {'greedy' if meta['temperature'] == 0 else f'sampling T={meta[\"temperature\"]}'}  ",
        f"",
        f"## Summary",
        f"",
        f"| Metric | {summ_a['label']} | {summ_b['label']} | Δ |",
        f"|--------|{'---'*3}|{'---'*3}|{'---'}|",
    ]

    def row(name: str, a_val: str, b_val: str, delta: str = ""):
        return f"| {name} | {a_val} | {b_val} | {delta} |"

    a_acc = summ_a["accuracy"]
    b_acc = summ_b["accuracy"]
    delta_acc = f"+{(b_acc - a_acc)*100:.1f}pp" if b_acc >= a_acc else f"{(b_acc - a_acc)*100:.1f}pp"

    lines += [
        row("Final-answer accuracy",
            _pct(summ_a['correct'], summ_a['n']),
            _pct(summ_b['correct'], summ_b['n']),
            delta_acc),
        row("Format pass rate",
            _pct(summ_a['format_ok'], summ_a['n']),
            _pct(summ_b['format_ok'], summ_b['n'])),
        row("Avg steps / solution",
            f"{summ_a['avg_steps']:.1f}",
            f"{summ_b['avg_steps']:.1f}"),
        row("Avg generation time",
            f"{summ_a['avg_time_s']:.1f}s",
            f"{summ_b['avg_time_s']:.1f}s"),
        "",
        "## Outcome Breakdown",
        "",
        f"| Outcome | {summ_a['label']} | {summ_b['label']} |",
        f"|---------|{'---'*3}|{'---'*3}|",
    ]

    # Both correct / only RL correct / only base correct / both wrong
    from collections import Counter
    outcomes: Counter = Counter()
    for r in (meta.get("_results") or []):
        a_ok = r["model_a"]["exact_match"]
        b_ok = r["model_b"]["exact_match"]
        if a_ok and b_ok:
            outcomes["both_correct"] += 1
        elif not a_ok and b_ok:
            outcomes["only_b_correct"] += 1
        elif a_ok and not b_ok:
            outcomes["only_a_correct"] += 1
        else:
            outcomes["both_wrong"] += 1
    n = summ_a["n"]
    lines += [
        f"| Both correct | {outcomes['both_correct']} ({_pct(outcomes['both_correct'], n)}) | — |",
        f"| Only {summ_b['label']} correct | — | {outcomes['only_b_correct']} ({_pct(outcomes['only_b_correct'], n)}) |",
        f"| Only {summ_a['label']} correct | {outcomes['only_a_correct']} ({_pct(outcomes['only_a_correct'], n)}) | — |",
        f"| Both wrong | {outcomes['both_wrong']} ({_pct(outcomes['both_wrong'], n)}) | — |",
        "",
        "---",
        f"*Generated by `scripts/run_inference_comparison.py`*",
    ]

    p = out_dir / "summary.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


_HTML_TMPL = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Inference Comparison — {run_id}</title>
<style>
  :root {{
    --base:#2563EB; --rl:#16A34A; --correct:#dcfce7; --wrong:#fee2e2;
    --neutral:#f1f5f9; --border:#e2e8f0; --text:#1e293b;
    --step-bg:#f8fafc; --answer-bg:#fef9c3;
  }}
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:'Segoe UI',system-ui,sans-serif; color:var(--text);
         background:#f8fafc; padding:1.5rem; line-height:1.5; }}
  h1 {{ font-size:1.6rem; font-weight:700; margin-bottom:.5rem; }}
  h2 {{ font-size:1.15rem; font-weight:600; margin:1.5rem 0 .6rem; color:#475569; }}
  .meta {{ background:#fff; border:1px solid var(--border); border-radius:.75rem;
           padding:1rem 1.25rem; margin-bottom:1.5rem; display:flex;
           flex-wrap:wrap; gap:.75rem 2.5rem; font-size:.85rem; color:#475569; }}
  .meta strong {{ color:var(--text); }}

  /* Summary cards */
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
            gap:.9rem; margin-bottom:1.8rem; }}
  .card {{ background:#fff; border:1px solid var(--border); border-radius:.75rem;
           padding:1rem 1.2rem; }}
  .card .label {{ font-size:.78rem; color:#64748b; text-transform:uppercase;
                  letter-spacing:.04em; margin-bottom:.25rem; }}
  .card .val {{ font-size:1.65rem; font-weight:700; }}
  .card .sub {{ font-size:.82rem; color:#94a3b8; margin-top:.1rem; }}
  .card.base .val {{ color:var(--base); }}
  .card.rl   .val {{ color:var(--rl); }}
  .card.delta .val {{ color:#7c3aed; }}

  /* Comparison table */
  .cmp-table {{ width:100%; border-collapse:collapse; background:#fff;
                border:1px solid var(--border); border-radius:.75rem;
                overflow:hidden; margin-bottom:2rem; font-size:.83rem; }}
  .cmp-table thead tr {{ background:#f1f5f9; }}
  .cmp-table th {{ padding:.65rem .9rem; text-align:left; font-weight:600;
                   border-bottom:2px solid var(--border); }}
  .cmp-table td {{ padding:.55rem .9rem; border-bottom:1px solid var(--border);
                   vertical-align:top; }}
  .cmp-table tr:last-child td {{ border-bottom:none; }}
  .cmp-table tr:hover td {{ background:#f8fafc; }}
  .tick {{ color:#16a34a; font-weight:700; }}
  .cross {{ color:#dc2626; font-weight:700; }}
  .both-correct {{ background:#f0fdf4; }}
  .only-rl      {{ background:#eff6ff; }}
  .only-base    {{ background:#fff7ed; }}
  .both-wrong   {{ background:#fef2f2; }}

  /* Expandable detail panels */
  details {{ margin:.3rem 0; }}
  summary {{ cursor:pointer; font-weight:500; color:#475569; font-size:.8rem;
             user-select:none; }}
  summary:hover {{ color:#1e293b; }}
  .solution-box {{ background:var(--step-bg); border:1px solid var(--border);
                   border-radius:.5rem; padding:.75rem 1rem; margin-top:.4rem;
                   white-space:pre-wrap; font-family:'Fira Code',monospace;
                   font-size:.78rem; line-height:1.55; overflow-x:auto; }}
  .answer-badge {{ display:inline-block; padding:.15rem .55rem; border-radius:.35rem;
                   font-size:.78rem; font-weight:600; margin-left:.3rem; }}
  .badge-correct {{ background:#dcfce7; color:#166534; }}
  .badge-wrong   {{ background:#fee2e2; color:#991b1b; }}
  .badge-gold    {{ background:#fef9c3; color:#854d0e; }}

  /* Progress bar */
  .bar-wrap {{ height:.6rem; background:#e2e8f0; border-radius:.3rem; margin:.2rem 0 0; }}
  .bar-fill {{ height:100%; border-radius:.3rem; }}
  .bar-base {{ background:var(--base); }}
  .bar-rl   {{ background:var(--rl); }}

  /* Question text */
  .question-cell {{ max-width:340px; word-break:break-word; font-size:.82rem; }}
  .idx {{ color:#94a3b8; font-size:.75rem; }}
  .ref-sol {{ font-size:.75rem; color:#64748b; }}

  @media(max-width:700px) {{
    body {{ padding:.75rem; }}
    .cmp-table {{ font-size:.75rem; }}
    .card .val {{ font-size:1.35rem; }}
  }}
</style>
</head>
<body>
<h1>📊 Inference Comparison Report</h1>

<div class="meta">
  <div><span class="label">Run ID</span><br><strong>{run_id}</strong></div>
  <div><span class="label">Date</span><br><strong>{timestamp}</strong></div>
  <div><span class="label">Dataset</span><br><strong>{data_path}</strong></div>
  <div><span class="label">Samples</span><br><strong>{n_samples}</strong></div>
  <div><span class="label">Decode</span><br><strong>{decode_mode}</strong></div>
  <div><span class="label">Base model</span><br><strong>{base_model_name}</strong></div>
  <div><span class="label">RL model</span><br><strong>{rl_model_name}</strong></div>
</div>

<h2>Summary</h2>
<div class="cards">
  <div class="card base">
    <div class="label">{label_a} accuracy</div>
    <div class="val">{acc_a_pct}</div>
    <div class="sub">{correct_a} / {n} correct</div>
    <div class="bar-wrap"><div class="bar-fill bar-base" style="width:{acc_a_pct}"></div></div>
  </div>
  <div class="card rl">
    <div class="label">{label_b} accuracy</div>
    <div class="val">{acc_b_pct}</div>
    <div class="sub">{correct_b} / {n} correct</div>
    <div class="bar-wrap"><div class="bar-fill bar-rl" style="width:{acc_b_pct}"></div></div>
  </div>
  <div class="card delta">
    <div class="label">Improvement (Δ accuracy)</div>
    <div class="val">{delta_pp}</div>
    <div class="sub">percentage points</div>
  </div>
  <div class="card">
    <div class="label">Only RL correct</div>
    <div class="val" style="color:#7c3aed">{only_rl}</div>
    <div class="sub">questions RL solved, base didn't</div>
  </div>
  <div class="card">
    <div class="label">{label_a} format pass</div>
    <div class="val" style="color:#0891b2">{fmt_a_pct}</div>
    <div class="sub">Step N: + Final Answer: present</div>
  </div>
  <div class="card">
    <div class="label">{label_b} format pass</div>
    <div class="val" style="color:#0891b2">{fmt_b_pct}</div>
    <div class="sub">Step N: + Final Answer: present</div>
  </div>
  <div class="card">
    <div class="label">Avg steps ({label_a})</div>
    <div class="val" style="color:#d97706">{avg_steps_a}</div>
    <div class="sub">reasoning steps per solution</div>
  </div>
  <div class="card">
    <div class="label">Avg steps ({label_b})</div>
    <div class="val" style="color:#d97706">{avg_steps_b}</div>
    <div class="sub">reasoning steps per solution</div>
  </div>
</div>

<h2>Per-Question Results</h2>
{table_html}

<hr style="margin:2rem 0; border-color:var(--border);">
<p style="font-size:.78rem;color:#94a3b8">
  Generated by <code>scripts/run_inference_comparison.py</code> ·
  Prompt: <code>create_solver_messages()</code> from <code>src/config/prompts.py</code>
</p>
</body>
</html>
"""


def _build_html(results: List[Dict], summ_a: Dict, summ_b: Dict, meta: Dict) -> str:
    from collections import Counter
    outcomes: Counter = Counter()
    for r in results:
        a_ok = r["model_a"]["exact_match"]
        b_ok = r["model_b"]["exact_match"]
        if a_ok and b_ok:
            outcomes["both_correct"] += 1
        elif not a_ok and b_ok:
            outcomes["only_rl"] += 1
        elif a_ok and not b_ok:
            outcomes["only_base"] += 1
        else:
            outcomes["both_wrong"] += 1

    def _esc(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _badge(ok: bool, pred: str, gold: str) -> str:
        cls = "badge-correct" if ok else "badge-wrong"
        symbol = "✓" if ok else "✗"
        tip = f"pred: {_esc(pred)} | gold: {_esc(gold)}"
        return (f'<span class="answer-badge {cls}" title="{tip}">'
                f'{symbol} {_esc(pred) or "—"}</span>')

    def _solution_detail(label: str, sol: str, score: Dict) -> str:
        steps = score["step_count"]
        fmt = "✓ format" if score["format_ok"] else "✗ format"
        sol_esc = _esc(sol[:4000] + ("…" if len(sol) > 4000 else ""))
        return (
            f'<details><summary>{_esc(label)} · {steps} steps · {fmt}</summary>'
            f'<div class="solution-box">{sol_esc}</div></details>'
        )

    def _row_class(a_ok: bool, b_ok: bool) -> str:
        if a_ok and b_ok:     return "both-correct"
        if not a_ok and b_ok: return "only-rl"
        if a_ok and not b_ok: return "only-base"
        return "both-wrong"

    rows_html = []
    for i, r in enumerate(results):
        a = r["model_a"]
        b = r["model_b"]
        rc = _row_class(a["exact_match"], b["exact_match"])
        a_ok_sym = '<span class="tick">✓</span>' if a["exact_match"] else '<span class="cross">✗</span>'
        b_ok_sym = '<span class="tick">✓</span>' if b["exact_match"] else '<span class="cross">✗</span>'
        ref_snip = textwrap.shorten(r.get("reference_solution", ""), 80, placeholder="…")

        rows_html.append(f"""
<tr class="{rc}">
  <td class="idx">#{i+1}</td>
  <td class="question-cell">
    {_esc(textwrap.shorten(r["question"], 200, placeholder="…"))}
    <br><span class="badge-gold answer-badge">gold: {_esc(r["gold_final"])}</span>
    {f'<br><span class="ref-sol">ref: {_esc(ref_snip)}</span>' if ref_snip else ''}
  </td>
  <td>
    {a_ok_sym}
    {_badge(a["exact_match"], a["pred_final"], r["gold_final"])}
    <br>{_solution_detail(summ_a["label"], r["solution_a"], a)}
    <br><span style="font-size:.72rem;color:#94a3b8">{a['elapsed_s']:.1f}s</span>
  </td>
  <td>
    {b_ok_sym}
    {_badge(b["exact_match"], b["pred_final"], r["gold_final"])}
    <br>{_solution_detail(summ_b["label"], r["solution_b"], b)}
    <br><span style="font-size:.72rem;color:#94a3b8">{b['elapsed_s']:.1f}s</span>
  </td>
</tr>""")

    table_html = f"""
<table class="cmp-table">
<thead>
  <tr>
    <th>#</th>
    <th>Question</th>
    <th>{_esc(summ_a["label"])}</th>
    <th>{_esc(summ_b["label"])}</th>
  </tr>
</thead>
<tbody>
{"".join(rows_html)}
</tbody>
</table>"""

    n = summ_a["n"]
    a_acc = summ_a["accuracy"]
    b_acc = summ_b["accuracy"]
    delta = b_acc - a_acc
    delta_str = f"+{delta*100:.1f}pp" if delta >= 0 else f"{delta*100:.1f}pp"

    decode_mode = (
        "greedy (T=0)"
        if meta["temperature"] == 0
        else f"sampling (T={meta['temperature']})"
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
        acc_a_pct=f"{a_acc*100:.1f}%",
        acc_b_pct=f"{b_acc*100:.1f}%",
        correct_a=summ_a["correct"],
        correct_b=summ_b["correct"],
        n=n,
        delta_pp=delta_str,
        only_rl=outcomes["only_rl"],
        fmt_a_pct=f"{summ_a['format_rate']*100:.1f}%",
        fmt_b_pct=f"{summ_b['format_rate']*100:.1f}%",
        avg_steps_a=f"{summ_a['avg_steps']:.1f}",
        avg_steps_b=f"{summ_b['avg_steps']:.1f}",
        table_html=table_html,
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
        label_a = args.base_label or "Qwen2.5-Math-1.5B (base)"

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
    else:
        model_a, tok_a = _load_hf_base_model(BASE_MODEL_HF, device, label_a)
        model_a_path = BASE_MODEL_HF

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
                             args.max_new_tokens, args.temperature, device)
        sol_b, t_b = _infer(model_b, tok_b, question,
                             args.max_new_tokens, args.temperature, device)

        score_a = _score(sol_a, gold_final)
        score_b = _score(sol_b, gold_final)

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

        # Live progress line
        a_sym = "✓" if score_a["exact_match"] else "✗"
        b_sym = "✓" if score_b["exact_match"] else "✗"
        tqdm.write(
            f"  [{idx+1:3d}/{len(rows)}]  {label_a[:22]}: {a_sym}  "
            f"{label_b[:22]}: {b_sym}  "
            f"gold={gold_final}  "
            f'pred_a={score_a["pred_final"] or "—"}  '
            f'pred_b={score_b["pred_final"] or "—"}'
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
    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info("  %-30s accuracy: %.1f%%  (%d/%d)",
                label_a, summ_a["accuracy"]*100, summ_a["correct"], summ_a["n"])
    logger.info("  %-30s accuracy: %.1f%%  (%d/%d)",
                label_b, summ_b["accuracy"]*100, summ_b["correct"], summ_b["n"])
    delta = (summ_b["accuracy"] - summ_a["accuracy"]) * 100
    sign  = "+" if delta >= 0 else ""
    logger.info("  Accuracy Δ (RL − base): %s%.1f pp", sign, delta)
    logger.info("")
    logger.info("  %-30s format: %.1f%%  avg_steps: %.1f",
                label_a, summ_a["format_rate"]*100, summ_a["avg_steps"])
    logger.info("  %-30s format: %.1f%%  avg_steps: %.1f",
                label_b, summ_b["format_rate"]*100, summ_b["avg_steps"])
    logger.info("=" * 60)
    logger.info("Reports written to: %s", out_dir)
    logger.info("  JSON    : %s", json_path)
    logger.info("  Markdown: %s", md_path)
    logger.info("  HTML    : %s", html_path)


if __name__ == "__main__":
    main()
