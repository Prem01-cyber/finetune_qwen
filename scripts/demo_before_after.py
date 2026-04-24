"""Before / after demo — baseline vs GRPO-trained policy.

Designed for hackathon judges: loads both models, runs greedy evaluation on
a fixed problem set, and prints a clean side-by-side comparison with full
solution text for the most interesting examples.

Features
--------
* Handles all checkpoint types: HF model IDs, GRPO full-weight saves,
  PEFT/LoRA adapter directories.
* Automatically loads the chat template from the base model when the
  checkpoint tokenizer doesn't have one (fixes the 0% accuracy bug that
  silently swallows TemplateErrors).
* Reads ``metrics.jsonl`` (if present) and prints the full accuracy curve,
  showing judges the training progression at a glance.
* Saves machine-readable JSON (for grading scripts) and prints a human-
  readable Markdown table.
* Shows full solution text for the best wins and worst regressions.

Quick-start
-----------
After a GRPO run, point at ``best_policy/``::

    python scripts/demo_before_after.py \\
        --baseline-model checkpoints/dual_task_v1 \\
        --trained-model  checkpoints/grpo/<run>/best_policy \\
        --problems       data/sft/gsm8k_sft.jsonl \\
        --max-samples    100

Include the training curve::

    python scripts/demo_before_after.py \\
        --baseline-model checkpoints/dual_task_v1 \\
        --trained-model  checkpoints/grpo/<run>/best_policy \\
        --metrics-jsonl  checkpoints/grpo/<run>/metrics.jsonl \\
        --problems       data/sft/gsm8k_sft.jsonl \\
        --max-samples    100 \\
        --records-out    results/demo.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.sft.solution_format import extract_final_answer_numeric_str
from src.utils.attn_backend import select_attn_implementation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

_SEP = "=" * 78
_SEP2 = "-" * 78


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class Problem:
    question: str
    gold_final: str


def _parse_gold(answer: str) -> str:
    m = re.search(r"####\s*([-0-9.,/ ]+)", answer)
    if m:
        return m.group(1).strip().replace(",", "")
    return answer.strip().splitlines()[-1].strip()


def _load_problems(path: Path, max_samples: int) -> List[Problem]:
    """Accept GSM8K ``{question, answer}`` or SFT ``{messages}`` JSONL."""
    out: List[Problem] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if max_samples > 0 and len(out) >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "question" in obj and "answer" in obj:
                out.append(Problem(
                    question=obj["question"].strip(),
                    gold_final=_parse_gold(obj["answer"]),
                ))
            elif "messages" in obj:
                user = next(
                    (m["content"] for m in obj["messages"] if m.get("role") == "user"), ""
                ).strip()
                asst = next(
                    (m["content"] for m in obj["messages"] if m.get("role") == "assistant"), ""
                )
                gold = extract_final_answer_numeric_str(asst) or ""
                out.append(Problem(question=user, gold_final=gold.strip()))
    return out


# ---------------------------------------------------------------------------
# Model loading — handles HF IDs, full-weight saves, and PEFT adapters
# ---------------------------------------------------------------------------

def _ensure_chat_template(
    tokenizer: AutoTokenizer,
    fallback_model: str = "Qwen/Qwen2.5-Math-1.5B-Instruct",
) -> None:
    """Load chat template from *fallback_model* when the checkpoint lacks one.

    SFT adapter checkpoints often omit the chat_template from their tokenizer
    config.  Without it, ``apply_chat_template`` raises a TemplateError that
    is silently swallowed inside ``evaluate_gsm8k``, returning 0% accuracy.
    """
    if tokenizer.chat_template is not None:
        return
    logger.info("Tokenizer missing chat_template — loading from %s", fallback_model)
    try:
        _base_tok = AutoTokenizer.from_pretrained(fallback_model, trust_remote_code=True)
        if _base_tok.chat_template is not None:
            tokenizer.chat_template = _base_tok.chat_template
            logger.info("Chat template loaded.")
    except Exception as exc:
        logger.warning("Could not load chat template: %s", exc)


def _load_model(
    checkpoint: str,
    base_model_id: str,
    device: torch.device,
    dtype: torch.dtype,
    attn_impl: str,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model + tokenizer from any checkpoint style.

    Handles:
    * HuggingFace model ID  (e.g. ``Qwen/Qwen2.5-Math-1.5B-Instruct``)
    * GRPO full-weight save (directory with ``model.safetensors`` / pytorch_model*)
    * PEFT/LoRA adapter dir (directory with ``adapter_config.json``)
    """
    # PEFT shim — prevents crash in merge_and_unload on some versions.
    if "transformers.integrations.tensor_parallel" not in sys.modules:
        sys.modules["transformers.integrations.tensor_parallel"] = types.ModuleType(
            "tensor_parallel"
        )

    ckpt_path = Path(checkpoint)
    is_adapter = ckpt_path.is_dir() and (ckpt_path / "adapter_config.json").exists()
    is_local_full = ckpt_path.is_dir() and not is_adapter

    # Tokenizer
    tok_src = checkpoint if (ckpt_path.is_dir() and (ckpt_path / "tokenizer_config.json").exists()) else base_model_id
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # standard for generation
    _ensure_chat_template(tokenizer, fallback_model=base_model_id)

    load_kw = dict(
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map={"": device},
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )

    if is_adapter:
        # Read base model from pipeline_meta.json if present
        meta_file = ckpt_path / "pipeline_meta.json"
        _base = base_model_id
        if meta_file.exists():
            _base = json.loads(meta_file.read_text()).get("base_model", _base)
        logger.info("PEFT adapter — loading base %s then merging %s", _base, checkpoint)
        _base_mdl = AutoModelForCausalLM.from_pretrained(_base, **load_kw)
        model = PeftModel.from_pretrained(_base_mdl, checkpoint).merge_and_unload()
        model = model.to(device)
    else:
        # Full weights (GRPO save) or HF model ID
        src = checkpoint if is_local_full else checkpoint
        logger.info("Loading full-weight model from %s", src)
        model = AutoModelForCausalLM.from_pretrained(src, **load_kw)

    # Re-enable requires_grad isn't needed for eval, but ensure eval mode.
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    logger.info("Loaded: %s  (%.2fB params, %.1f GB VRAM est.)",
                checkpoint, n / 1e9, n * 2 / 1e9)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _build_prompt(tokenizer: AutoTokenizer, question: str) -> str:
    """Format question using the model's chat template (matches training format)."""
    if tokenizer.chat_template is None:
        return question
    msgs = [
        {"role": "system", "content": "You are a helpful math assistant. Solve the problem step-by-step and end with 'Final Answer: <number>'."},
        {"role": "user",   "content": question},
    ]
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return question


def _stop_ids(tokenizer: AutoTokenizer) -> List[int]:
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end not in ids:
        ids.append(im_end)
    return ids or None  # type: ignore[return-value]


@torch.no_grad()
def _generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    prompt = _build_prompt(tokenizer, question)
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(device)
    prompt_len = enc["input_ids"].shape[1]

    out = model.generate(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=False,       # greedy — deterministic for reproducibility
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=_stop_ids(tokenizer),
        use_cache=True,
    )
    return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _normalize(x: str) -> str:
    if not x:
        return ""
    s = x.strip().replace(",", "").replace("$", "").strip()
    try:
        f = float(s)
        return f"{int(f)}" if f == int(f) else f"{f}"
    except ValueError:
        return s


@dataclass
class Record:
    question: str
    gold: str
    pred: str
    correct: bool
    solution_text: str


def _score_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    problems: List[Problem],
    max_new_tokens: int,
    device: torch.device,
    label: str,
) -> Tuple[int, List[Record]]:
    records: List[Record] = []
    correct = 0
    for prob in tqdm(problems, desc=f"Scoring {label}", unit="q", dynamic_ncols=True):
        try:
            text = _generate(model, tokenizer, prob.question, max_new_tokens, device)
        except Exception as exc:
            text = f"[generation error: {exc}]"
        pred = extract_final_answer_numeric_str(text) or ""
        ok = bool(pred) and _normalize(pred) == _normalize(prob.gold_final)
        if ok:
            correct += 1
        records.append(Record(
            question=prob.question,
            gold=prob.gold_final,
            pred=pred,
            correct=ok,
            solution_text=text,
        ))
    return correct, records


# ---------------------------------------------------------------------------
# Metrics curve
# ---------------------------------------------------------------------------

def _load_metrics_curve(path: Path) -> List[Dict]:
    """Read metrics.jsonl and return rows that contain GSM8K accuracy."""
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "accuracy" in obj or "iteration" in obj:
                    rows.append(obj)
            except json.JSONDecodeError:
                pass
    return rows


def _print_curve(rows: List[Dict]) -> None:
    if not rows:
        return
    print(f"\n{_SEP}")
    print("TRAINING ACCURACY CURVE  (from metrics.jsonl)")
    print(_SEP)
    print(f"{'Iter':>5}  {'GSM8K%':>7}  {'Reward':>7}  {'Batch%':>7}  {'LR':>10}  {'Time(s)':>8}")
    print(_SEP2)
    for r in rows:
        it  = r.get("iteration", "")
        acc = r.get("accuracy", None)
        rwd = r.get("mean_reward", None)
        bat = r.get("batch_accuracy", None)
        lr  = r.get("learning_rate", None)
        ts  = r.get("iter_time_s", None)
        acc_s = f"{100*acc:.1f}%" if acc is not None else "—"
        rwd_s = f"{rwd:.3f}"      if rwd is not None else "—"
        bat_s = f"{100*bat:.1f}%" if bat is not None else "—"
        lr_s  = f"{lr:.2e}"       if lr  is not None else "—"
        ts_s  = f"{ts:.1f}"       if ts  is not None else "—"
        print(f"{it:>5}  {acc_s:>7}  {rwd_s:>7}  {bat_s:>7}  {lr_s:>10}  {ts_s:>8}")
    print()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _print_summary(
    base_correct: int,
    tr_correct: int,
    base_records: List[Record],
    tr_records: List[Record],
    baseline_name: str,
    trained_name: str,
    n_solutions: int = 3,
) -> None:
    n = len(base_records)
    wins   = [(p, b, t) for p, b, t in zip(base_records, base_records, tr_records) if not b.correct and t.correct]
    losses = [(p, b, t) for p, b, t in zip(base_records, base_records, tr_records) if b.correct and not t.correct]
    both_wrong  = sum(1 for b, t in zip(base_records, tr_records) if not b.correct and not t.correct)
    both_right  = sum(1 for b, t in zip(base_records, tr_records) if b.correct and t.correct)

    delta = tr_correct - base_correct
    sign  = "+" if delta >= 0 else ""

    print(f"\n{_SEP}")
    print("BEFORE  vs  AFTER — GSM8K accuracy (greedy decoding, fixed seed)")
    print(_SEP)
    print(f"  Baseline  : {baseline_name}")
    print(f"  Trained   : {trained_name}")
    print(_SEP2)
    print(f"  Baseline accuracy  : {base_correct}/{n}  ({100*base_correct/n:.1f}%)")
    print(f"  Trained  accuracy  : {tr_correct}/{n}  ({100*tr_correct/n:.1f}%)")
    print(f"  Delta              : {sign}{delta} problems  ({sign}{100*delta/n:.1f} pp)")
    print(_SEP2)
    print(f"  Newly correct (wins)   : {len(wins)}")
    print(f"  Newly wrong  (losses)  : {len(losses)}")
    print(f"  Both correct           : {both_right}")
    print(f"  Both wrong             : {both_wrong}")
    print(_SEP)

    if wins:
        print(f"\n{'='*78}")
        print(f"WINS — problems the RL model now solves that the baseline could not")
        print(f"{'='*78}")
        for i, (_, base_r, tr_r) in enumerate(wins[:n_solutions]):
            print(f"\n[Win {i+1}/{min(n_solutions, len(wins))}]")
            _print_problem(base_r, tr_r)

    if losses:
        print(f"\n{'='*78}")
        print(f"REGRESSIONS — problems the baseline solved but the RL model now misses")
        print(f"{'='*78}")
        for i, (_, base_r, tr_r) in enumerate(losses[:min(2, len(losses))]):
            print(f"\n[Regression {i+1}/{min(2, len(losses))}]")
            _print_problem(base_r, tr_r, is_regression=True)

    print(f"\n{_SEP}")
    pct_gain = 100 * delta / max(n - base_correct, 1)
    print(f"SUMMARY: RL training fixed {len(wins)} problems, regressed {len(losses)}.")
    print(f"         Net: {sign}{delta} pts.  Relative gain on previously-wrong: {pct_gain:+.1f}%")
    print(_SEP)


def _print_problem(base_r: Record, tr_r: Record, is_regression: bool = False) -> None:
    q = base_r.question
    # Truncate long questions
    if len(q) > 250:
        q = q[:247] + "..."
    print(f"  Q : {q}")
    print(f"  Gold   : {base_r.gold}")
    if not is_regression:
        print(f"  Before : {base_r.pred!r:30s}  ✗")
        print(f"  After  : {tr_r.pred!r:30s}  ✓")
        # Show trained solution (truncated)
        sol = tr_r.solution_text.strip()
        if sol:
            lines = sol.splitlines()
            show = "\n    ".join(lines[:12])
            if len(lines) > 12:
                show += f"\n    ... ({len(lines)-12} more lines)"
            print(f"\n  Solution (trained model):\n    {show}")
    else:
        print(f"  Before : {base_r.pred!r:30s}  ✓")
        print(f"  After  : {tr_r.pred!r:30s}  ✗")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--baseline-model", default="checkpoints/dual_task_v1",
        help="Pre-RL checkpoint. HF model ID, full-weight dir, or PEFT adapter dir.",
    )
    parser.add_argument(
        "--trained-model", required=True,
        help="Post-RL checkpoint (GRPO best_policy/ dir, or iteration checkpoint).",
    )
    parser.add_argument(
        "--base-model-for-adapter", default="Qwen/Qwen2.5-Math-1.5B-Instruct",
        help="Base model used when loading a PEFT adapter checkpoint.",
    )
    parser.add_argument(
        "--problems", type=Path, default=Path("data/sft/gsm8k_sft.jsonl"),
        help="JSONL eval set. Defaults to GSM8K training split (first --max-samples rows).",
    )
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--metrics-jsonl", type=Path, default=None,
        help="Path to metrics.jsonl from a GRPO run — prints the accuracy curve.",
    )
    parser.add_argument(
        "--n-solutions", type=int, default=3,
        help="Number of win/loss examples to print in full.",
    )
    parser.add_argument(
        "--records-out", type=Path, default=None,
        help="Save full per-problem JSON records here (for judge grading scripts).",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype", default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    args = parser.parse_args()

    if not args.problems.is_file():
        logger.error("Problems file not found: %s", args.problems)
        return 2

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype  = dtype_map[args.dtype]
    device = torch.device(args.device)
    attn   = select_attn_implementation()
    logger.info("Device: %s | dtype: %s | attn: %s", device, args.dtype, attn)

    # Print training curve if available
    if args.metrics_jsonl:
        curve = _load_metrics_curve(args.metrics_jsonl)
        _print_curve(curve)

    problems = _load_problems(args.problems, args.max_samples)
    if not problems:
        logger.error("No problems loaded from %s", args.problems)
        return 2
    logger.info("Evaluating on %d problems from %s", len(problems), args.problems)

    # ── Baseline ──────────────────────────────────────────────────────────
    logger.info("%s\nScoring BASELINE: %s\n%s", _SEP, args.baseline_model, _SEP)
    t0 = time.perf_counter()
    base_model, base_tok = _load_model(
        args.baseline_model, args.base_model_for_adapter, device, dtype, attn
    )
    base_correct, base_records = _score_model(
        base_model, base_tok, problems, args.max_new_tokens, device, "baseline"
    )
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Baseline done in %.1fs — accuracy: %d/%d (%.1f%%)",
                time.perf_counter() - t0,
                base_correct, len(problems),
                100 * base_correct / len(problems))

    # ── Trained ───────────────────────────────────────────────────────────
    logger.info("%s\nScoring TRAINED: %s\n%s", _SEP, args.trained_model, _SEP)
    t0 = time.perf_counter()
    tr_model, tr_tok = _load_model(
        args.trained_model, args.base_model_for_adapter, device, dtype, attn
    )
    tr_correct, tr_records = _score_model(
        tr_model, tr_tok, problems, args.max_new_tokens, device, "trained"
    )
    del tr_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Trained done in %.1fs — accuracy: %d/%d (%.1f%%)",
                time.perf_counter() - t0,
                tr_correct, len(problems),
                100 * tr_correct / len(problems))

    # ── Summary ───────────────────────────────────────────────────────────
    _print_summary(
        base_correct, tr_correct,
        base_records, tr_records,
        baseline_name=args.baseline_model,
        trained_name=args.trained_model,
        n_solutions=args.n_solutions,
    )

    # ── Save records ──────────────────────────────────────────────────────
    if args.records_out:
        args.records_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "baseline_model": args.baseline_model,
            "trained_model":  args.trained_model,
            "n_problems":     len(problems),
            "baseline": {
                "correct": base_correct,
                "accuracy": base_correct / len(problems),
                "records": [vars(r) for r in base_records],
            },
            "trained": {
                "correct": tr_correct,
                "accuracy": tr_correct / len(problems),
                "records": [vars(r) for r in tr_records],
            },
        }
        args.records_out.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Per-problem records saved to %s", args.records_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
