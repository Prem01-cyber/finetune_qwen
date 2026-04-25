#!/usr/bin/env python3
"""
Download Chinar/AQuA-RAT from HuggingFace and convert it to the same JSONL
format used by gsm8k_sft.jsonl so the GRPO training script can consume it
directly via --gsm8k-data.

Chinar/AQuA-RAT schema (processed version)
-------------------------------------------
  prompt     : str  — the math question
  completion : str  — step-by-step reasoning ending with:
                      "The answer is X . Therefore, the correct answer is:  <value>"

Output schema (messages format expected by load_gsm8k)
-------------------------------------------------------
  {
    "id": "aqua_<idx>",
    "skill_id": "aqua_rat_algebra",
    "source": "Chinar/AQuA-RAT",
    "split": "train" | "validation",
    "messages": [
        {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
        {"role": "user",   "content": "Solve ... Problem:\\n<question>"},
        {"role": "assistant", "content": "Step 1: ...\\nFinal Answer: <value>"}
    ]
  }

The dataset has only a 'train' split — we reserve the last 500 rows as
a validation set and use the rest for training.

Usage
-----
  python scripts/prepare_aqua_dataset.py
  python scripts/prepare_aqua_dataset.py --val-size 300 --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Prompt constants (kept in sync with src/config/prompts.py)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

# The completion always ends with a variant of:
#   "The answer is E . Therefore, the correct answer is:  23"
_ANSWER_TAIL = re.compile(
    r"(?:The answer is\s+[A-Ea-e]\s*[.\-]?\s*)?"
    r"Therefore,?\s+the correct answer is\s*:?\s*(.+)$",
    re.IGNORECASE,
)


def _extract_answer_and_rationale(completion: str) -> Optional[tuple[str, str]]:
    """
    Split the completion into (rationale_lines, final_answer_str).
    Returns None if no extractable numeric answer is found.
    """
    # Find the tail marker
    m = _ANSWER_TAIL.search(completion)
    if not m:
        return None

    raw_answer = m.group(1).strip()
    # Everything before the tail is the rationale
    rationale = completion[: m.start()].strip()
    # Also strip a standalone "The answer is X ." line at the end of rationale
    rationale = re.sub(r"\s*The answer is\s+[A-Ea-e]\s*[.\-]?\s*$", "", rationale, flags=re.IGNORECASE).strip()

    # Normalise the answer to a clean numeric string
    final_answer = _normalise_answer(raw_answer)
    if final_answer is None:
        return None

    return rationale, final_answer


def _normalise_answer(raw: str) -> Optional[str]:
    """
    Extract a single numeric value from an answer string.

    "23"          → "23"
    "$ 1600"      → "1600"
    "8 seconds"   → "8"
    "5 and 1"     → None  (multi-value — skip)
    "I and II"    → None  (non-numeric — skip)
    "− 3 ≤ x ≤ 4" → None  (inequality — skip)
    """
    text = raw.strip()

    # Remove currency / whitespace
    text = text.replace("$", "").replace("Rs.", "").replace("Rs", "").replace(",", "").strip()

    # Handle unicode minus
    text = text.replace("\u2212", "-").replace("−", "-")

    # Skip if "and" still present (multi-value like "5 and 1")
    if re.search(r"\band\b", text, re.IGNORECASE):
        return None

    # Skip inequalities / expressions with variables
    if re.search(r"[a-zA-Z≤≥<>]", text):
        return None

    # Single number (integer or decimal, optionally negative)
    m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*(?:[a-zA-Z%°].*)?", text)
    if m:
        val_str = m.group(1)
        try:
            val = float(val_str)
            return str(int(val)) if val == int(val) else val_str
        except ValueError:
            pass

    return None


# ---------------------------------------------------------------------------
# Rationale → Step N: format
# ---------------------------------------------------------------------------

def _rationale_to_steps(rationale: str) -> list[str]:
    lines: list[str] = []
    for raw in rationale.splitlines():
        line = raw.strip()
        if line:
            line = line.replace("^", "**")
            lines.append(line)
    if not lines and rationale.strip():
        sentences = re.split(r"(?<=[.!?])\s+", rationale.strip())
        lines = [s.strip() for s in sentences if s.strip()]
    return lines


def _build_assistant(rationale: str, final_answer: str) -> str:
    steps = _rationale_to_steps(rationale)
    parts = [f"Step {i}: {line}" for i, line in enumerate(steps, 1)]
    body = "\n".join(parts)
    return f"{body}\nFinal Answer: {final_answer}" if body else f"Final Answer: {final_answer}"


# ---------------------------------------------------------------------------
# Row conversion
# ---------------------------------------------------------------------------

def convert_row(row: dict[str, Any], idx: int, split: str) -> Optional[dict[str, Any]]:
    question   = (row.get("prompt") or "").strip()
    completion = (row.get("completion") or "").strip()

    if not question or not completion:
        return None

    result = _extract_answer_and_rationale(completion)
    if result is None:
        return None

    rationale, final_answer = result
    assistant_text = _build_assistant(rationale, final_answer)

    return {
        "id": f"aqua_{split}_{idx}",
        "skill_id": "aqua_rat_algebra",
        "source": "Chinar/AQuA-RAT",
        "split": split,
        "messages": [
            {"role": "system",    "content": SOLVER_SYSTEM_PROMPT},
            {"role": "user",      "content": USER_WRAPPER.format(question=question)},
            {"role": "assistant", "content": assistant_text},
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/sft")
    parser.add_argument("--val-size",   type=int, default=500,
                        help="How many rows from the end of the dataset to use as validation.")
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets", file=sys.stderr)
        sys.exit(1)

    print("Downloading Chinar/AQuA-RAT …")
    ds = load_dataset("Chinar/AQuA-RAT")
    all_rows = list(ds["train"])
    total = len(all_rows)
    print(f"  Total rows: {total:,}")

    val_rows   = all_rows[-args.val_size:]
    train_rows = all_rows[: -args.val_size]

    splits = {
        "train":      train_rows,
        "validation": val_rows,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, rows in splits.items():
        if args.max_samples:
            rows = rows[: args.max_samples]

        records: list[dict] = []
        skipped = 0
        for idx, row in enumerate(rows):
            rec = convert_row(row, idx, split)
            if rec is None:
                skipped += 1
            else:
                records.append(rec)

        skip_pct = 100.0 * skipped / max(1, len(rows))

        if args.dry_run:
            print(f"\n── {split}: {len(records)} valid / {skipped} skipped ({skip_pct:.1f}%) ──")
            for rec in records[:3]:
                print(json.dumps(rec, indent=2))
            continue

        out_path = out_dir / f"aqua_{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"  [{split:12s}]  {len(records):6,d} valid  {skipped:5,d} skipped ({skip_pct:.1f}%)  →  {out_path}")

    if not args.dry_run:
        print("\nDone.  Launch continuation training with:")
        print("  bash launch_grpo_aqua.sh")


if __name__ == "__main__":
    main()
