"""Before/after demo for the self-improvement math environment.

Rubric rule #19 asks for "a clear environment design, objective reward
functions, *evidence that the model improved*, prevention against
reward hacking, a reproducible deployment story, and a sharp demo."
This script is the *sharp demo* piece: it loads the baseline model and
the RL-trained model, runs both on a fixed set of GSM8K-style
problems, and renders a side-by-side accuracy comparison.

Why a standalone script and not a notebook?

* scripts are trivially diffable, reviewable, and runnable in CI,
* a deterministic run (greedy decoding + fixed seed) makes the "before
  vs after" delta reproducible for judges,
* the same script works locally, on a Space sidecar, or inside a
  container.

Example::

    python scripts/demo_before_after.py \\
        --baseline-model Qwen/Qwen2.5-Math-1.5B-Instruct \\
        --trained-model checkpoints/ppo_curriculum/iteration_050/policy \\
        --problems data/sft/gsm8k_test.jsonl \\
        --max-samples 20
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.eval_sft_inference import _generate as hf_generate
from src.sft.solution_format import extract_final_answer_numeric_str

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data / IO helpers
# ---------------------------------------------------------------------------
@dataclass
class Problem:
    question: str
    gold_final: str


def _parse_gold(answer: str) -> str:
    """Pull the final numeric answer out of a GSM8K-style ``answer`` field."""
    m = re.search(r"####\s*([-0-9.,/ ]+)", answer)
    if m:
        return m.group(1).strip().replace(",", "")
    return answer.strip().splitlines()[-1].strip()


def _load_problems(path: Path, max_samples: int) -> List[Problem]:
    """Accept either GSM8K ``{question, answer}`` or SFT ``{messages}`` JSONL."""
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
                out.append(
                    Problem(
                        question=obj["question"].strip(),
                        gold_final=_parse_gold(obj["answer"]),
                    )
                )
            elif "messages" in obj:
                user = next(
                    (m["content"] for m in obj["messages"] if m.get("role") == "user"),
                    "",
                ).strip()
                asst = next(
                    (m["content"] for m in obj["messages"] if m.get("role") == "assistant"),
                    "",
                )
                gold = extract_final_answer_numeric_str(asst) or ""
                out.append(Problem(question=user, gold_final=gold.strip()))
    return out


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _load_model(
    checkpoint: str,
    base_model: str,
    device: str,
    dtype: torch.dtype,
) -> tuple:
    """Return (model, tokenizer).

    Handles three cases transparently: a bare HF model id, a local
    full-weight checkpoint, and a LoRA/PEFT adapter directory.
    """
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    adapter = Path(checkpoint)
    if (adapter / "adapter_config.json").is_file():
        base = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=dtype, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(base, adapter)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, torch_dtype=dtype, low_cpu_mem_usage=True
        )
    return model.to(device).eval(), tok


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _normalize(x: str) -> str:
    if not x:
        return ""
    s = x.strip().replace(",", "")
    try:
        f = float(s)
    except ValueError:
        return s
    return f"{int(f)}" if f.is_integer() else f"{f}"


def _score(model, tokenizer, problems: List[Problem], max_new_tokens: int):
    correct = 0
    records = []
    for prob in tqdm(problems, desc="solving", unit="q", dynamic_ncols=True):
        try:
            text = hf_generate(
                model=model,
                tokenizer=tokenizer,
                problem=prob.question,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                greedy=True,
            )
        except Exception as exc:
            text = f"[generation failed: {exc}]"

        pred = extract_final_answer_numeric_str(text) or ""
        ok = _normalize(pred) == _normalize(prob.gold_final) and pred != ""
        if ok:
            correct += 1
        records.append(
            {
                "question": prob.question,
                "gold": prob.gold_final,
                "pred": pred,
                "correct": ok,
                "full_text": text,
            }
        )
    return correct, records


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------
def _fmt_pct(num: int, denom: int) -> str:
    if denom == 0:
        return "0/0 (0%)"
    return f"{num}/{denom} ({100.0 * num / denom:.1f}%)"


def _print_summary(
    baseline: tuple, trained: tuple, problems: List[Problem]
) -> None:
    base_correct, base_records = baseline
    tr_correct, tr_records = trained
    n = len(problems)

    print("\n" + "=" * 78)
    print("BEFORE  vs  AFTER — GSM8K-style accuracy (greedy)")
    print("=" * 78)
    print(f"Baseline model : {_fmt_pct(base_correct, n)}")
    print(f"Trained  model : {_fmt_pct(tr_correct,   n)}")
    delta = tr_correct - base_correct
    sign = "+" if delta >= 0 else ""
    print(f"Delta          : {sign}{delta} problems "
          f"({sign}{100.0 * delta / n:.1f} pp)\n")

    flipped_good = [
        (p, b, t)
        for p, b, t in zip(problems, base_records, tr_records)
        if (not b["correct"]) and t["correct"]
    ]
    flipped_bad = [
        (p, b, t)
        for p, b, t in zip(problems, base_records, tr_records)
        if b["correct"] and (not t["correct"])
    ]

    print(f"Fixed by RL : {len(flipped_good)}")
    print(f"Broken by RL: {len(flipped_bad)}")

    if flipped_good:
        print("\n--- Sample wins (baseline wrong -> trained right) ---")
        for p, b, t in flipped_good[:3]:
            print(f"Q: {p.question}")
            print(f"  gold  = {p.gold_final}")
            print(f"  before= {b['pred']!r}  |  after= {t['pred']!r}")
            print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-model",
        default="Qwen/Qwen2.5-Math-1.5B-Instruct",
        help="Pre-RL checkpoint (HF id or local path).  Defaults to the base "
        "Qwen math model so you always have *some* baseline.",
    )
    parser.add_argument(
        "--trained-model",
        required=True,
        help="Post-RL checkpoint.  Either a PEFT adapter directory or a "
        "full-weight save.",
    )
    parser.add_argument(
        "--base-model-for-adapter",
        default="Qwen/Qwen2.5-Math-1.5B-Instruct",
        help="Base model to attach the trained adapter to, if applicable.",
    )
    parser.add_argument(
        "--problems",
        type=Path,
        default=Path("data/sft/gsm8k_test.jsonl"),
        help="JSONL file with ``{question, answer}`` or SFT-style rows.",
    )
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument(
        "--records-out",
        type=Path,
        default=None,
        help="Optional path to dump per-problem JSON records for later "
        "inspection / grading.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if not args.problems.is_file():
        logger.error("Problems file not found: %s", args.problems)
        return 2

    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[
        args.dtype
    ]

    problems = _load_problems(args.problems, args.max_samples)
    if not problems:
        logger.error("No problems loaded from %s", args.problems)
        return 2
    logger.info("Loaded %d problems from %s", len(problems), args.problems)

    logger.info("Scoring baseline: %s", args.baseline_model)
    t0 = time.time()
    base_model, base_tok = _load_model(
        args.baseline_model, args.baseline_model, args.device, dtype
    )
    baseline = _score(base_model, base_tok, problems, args.max_new_tokens)
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Baseline scored in %.1fs", time.time() - t0)

    logger.info("Scoring trained: %s", args.trained_model)
    t0 = time.time()
    tr_model, tr_tok = _load_model(
        args.trained_model, args.base_model_for_adapter, args.device, dtype
    )
    trained = _score(tr_model, tr_tok, problems, args.max_new_tokens)
    del tr_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Trained scored in %.1fs", time.time() - t0)

    _print_summary(baseline, trained, problems)

    if args.records_out:
        args.records_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "baseline_model": args.baseline_model,
            "trained_model": args.trained_model,
            "baseline": {
                "correct": baseline[0],
                "total": len(problems),
                "records": baseline[1],
            },
            "trained": {
                "correct": trained[0],
                "total": len(problems),
                "records": trained[1],
            },
        }
        args.records_out.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Wrote per-problem records to %s", args.records_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
