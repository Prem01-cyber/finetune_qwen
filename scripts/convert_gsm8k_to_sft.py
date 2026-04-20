#!/usr/bin/env python3
"""
Convert OpenAI GSM8K to SFT JSONL aligned with MathAgent solver format:

  Step 1: ...
  Step 2: ...
  ...
  Final Answer: <integer>

Each record uses a chat messages list for Qwen-style fine-tuning.

Usage
-----
  # From Hugging Face (default; same data as in test.ipynb)
  python scripts/convert_gsm8k_to_sft.py \\
      --output data/sft/gsm8k_sft.jsonl \\
      --splits train test

  # From a saved JSONL with columns \"question\" and \"answer\" (GSM8K schema)
  python scripts/convert_gsm8k_to_sft.py \\
      --source jsonl \\
      --input path/to/file.jsonl \\
      --output data/sft/gsm8k_sft.jsonl

Requires: pip install datasets (and datasets will pull pyarrow as needed)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

# Keep in sync with src.agent.math_agent.SOLVER_SYSTEM_PROMPT
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


def parse_gsm8k_answer(raw_answer: str) -> tuple[str, str]:
    """
    Split GSM8K 'answer' field into reasoning text and final integer string.

    GSM8K ends solutions with a line like: #### 42
    """
    text = raw_answer.strip()
    parts = re.split(r"\s*####\s*", text, maxsplit=1)
    reasoning = parts[0].strip()
    final = parts[1].strip() if len(parts) > 1 else ""
    # Normalize final (sometimes extra whitespace or commas)
    final = re.sub(r"[,\s]+", "", final)
    final_match = re.search(r"-?\d+", final)
    final_clean = final_match.group(0) if final_match else final
    return reasoning, final_clean


def reasoning_to_step_lines(reasoning: str) -> list[str]:
    """Turn reasoning into non-empty lines; each line becomes one Step N:."""
    lines: list[str] = []
    for raw in reasoning.splitlines():
        line = raw.strip()
        if line:
            lines.append(line)
    if not lines:
        # Rare: single blob without newlines — split on sentence boundaries lightly
        blob = reasoning.strip()
        if blob:
            chunks = re.split(r"(?<=[.!?])\s+", blob)
            lines = [c.strip() for c in chunks if c.strip()]
    return lines


def build_assistant_content(reasoning: str, final_answer: str) -> str:
    lines = reasoning_to_step_lines(reasoning)
    out_parts: list[str] = []
    for i, line in enumerate(lines, start=1):
        # Prefer SymPy-friendly numerics: ** not ^, ascii-friendly
        cleaned = line.replace("^", "**")
        out_parts.append(f"Step {i}: {cleaned}")
    body = "\n".join(out_parts)
    if final_answer:
        body = f"{body}\nFinal Answer: {final_answer}" if body else f"Final Answer: {final_answer}"
    return body


def row_to_record(
    question: str,
    answer: str,
    example_id: str,
    split: str,
) -> dict[str, Any] | None:
    reasoning, final_answer = parse_gsm8k_answer(answer)
    if not final_answer and "####" not in answer:
        return None
    assistant = build_assistant_content(reasoning, final_answer)
    if not assistant.strip():
        return None

    user_content = USER_WRAPPER.format(question=question.strip())

    return {
        "id": f"gsm8k_{example_id}",
        "skill_id": "gsm8k_grade_school",
        "source": "openai/gsm8k",
        "split": split,
        "messages": [
            {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant},
        ],
        # Convenience for non-chat trainers
        "text": f"<|system|>\n{SOLVER_SYSTEM_PROMPT}\n<|user|>\n{user_content}\n<|assistant|>\n{assistant}",
    }


def iter_hf_rows(dataset_name: str, config: str, splits: list[str]) -> Iterator[tuple[str, str, dict]]:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, config)
    for split in splits:
        if split not in ds:
            raise KeyError(f"Split {split!r} not in dataset. Available: {list(ds.keys())}")
        for i, row in enumerate(ds[split]):
            yield f"{split}_{i}", split, row


def main() -> None:
    p = argparse.ArgumentParser(description="Convert GSM8K to SFT JSONL (chat messages).")
    p.add_argument(
        "--source",
        choices=("hf", "jsonl"),
        default="hf",
        help="Load from Hugging Face dataset or a local JSONL file.",
    )
    p.add_argument("--dataset", default="openai/gsm8k", help="HF dataset id when --source hf.")
    p.add_argument("--config", default="main", help="HF config name when --source hf.")
    p.add_argument("--splits", nargs="+", default=["train", "test"], help="HF splits to export.")
    p.add_argument("--input", type=Path, help="Local JSONL path when --source jsonl.")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/sft/gsm8k_sft.jsonl"),
        help="Output JSONL path.",
    )
    args = p.parse_args()

    if args.source == "jsonl" and not args.input:
        raise SystemExit("--input is required when --source jsonl")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    n_ok, n_skip = 0, 0

    def process(example_id: str, split: str, row: dict) -> None:
        nonlocal n_ok, n_skip
        q = row.get("question", "")
        a = row.get("answer", "")
        rec = row_to_record(q, a, example_id, split)
        if rec is None:
            n_skip += 1
            return
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_ok += 1

    with args.output.open("w", encoding="utf-8") as out_f:
        if args.source == "hf":
            for example_id, split, row in iter_hf_rows(args.dataset, args.config, args.splits):
                process(example_id, split, row)
        else:
            for i, line in enumerate(args.input.open(encoding="utf-8")):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                process(str(i), "jsonl", row)

    print(f"Wrote {n_ok} examples to {args.output} ({n_skip} skipped).")


if __name__ == "__main__":
    main()
