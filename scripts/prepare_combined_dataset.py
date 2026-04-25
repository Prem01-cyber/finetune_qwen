#!/usr/bin/env python3
"""
Combined dataset pipeline — NuminaMath-CoT + OpenMathInstruct-2
================================================================
Downloads, filters, normalises, and merges two large math datasets into a single
JSONL file (train / val / test) that the GRPO training script can consume directly
via --gsm8k-data.

Why these two datasets
----------------------
  NuminaMath-CoT  (AI-MO/NuminaMath-CoT)
      860 K problems.  Clean \\boxed{} answers.  7 rich topic categories that map
      directly to ZPD skill_ids.  Sources span AMC, AIME, Chinese HS, olympiads,
      and synthetic — giving natural difficulty diversity.

  OpenMathInstruct-2  (nvidia/OpenMathInstruct-2)
      14 M synthetic problems with step-level CoT.  `expected_answer` is pre-verified.
      Diverse surface forms prevent pattern memorisation.  We skip any row whose
      problem_source is "gsm8k" (already in prior training).

Output schema (identical to gsm8k_sft.jsonl / aqua_train.jsonl)
---------------------------------------------------------------
  {
    "id":       "<source>_<split>_<idx>",
    "skill_id": "<topic_slug>",        ← used by ZPD CurriculumManager
    "source":   "<hf_dataset_name>",
    "split":    "train" | "val" | "test",
    "difficulty": 1 | 2 | 3,          ← 1=easy 2=medium 3=hard (for ZPD)
    "task_type": "solve",
    "messages": [
        {"role": "system",    "content": SOLVER_SYSTEM_PROMPT},
        {"role": "user",      "content": "Solve ... Problem:\\n<question>"},
        {"role": "assistant", "content": "Step 1: ...\\nFinal Answer: <answer>"}
    ]
  }

Usage
-----
  # Quick test (no download, just show stats)
  python scripts/prepare_combined_dataset.py --dry-run

  # Full pipeline (default caps: 20 K numina + 15 K openmath)
  python scripts/prepare_combined_dataset.py

  # Larger run
  python scripts/prepare_combined_dataset.py --max-numina 40000 --max-openmath 30000

  # Only one source
  python scripts/prepare_combined_dataset.py --skip-openmath
  python scripts/prepare_combined_dataset.py --skip-numina

  # Custom output dir
  python scripts/prepare_combined_dataset.py --output-dir data/sft/combined
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — kept in sync with src/config/prompts.py
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
# Skill-ID mappings  (drives ZPD CurriculumManager per-topic mastery)
# ---------------------------------------------------------------------------

# NuminaMath-CoT  `type` field → skill_id
NUMINA_TYPE_TO_SKILL: Dict[str, str] = {
    "algebra":                   "numina_algebra",
    "intermediate_algebra":      "numina_algebra",
    "prealgebra":                "numina_prealgebra",
    "number_theory":             "numina_number_theory",
    "geometry":                  "numina_geometry",
    "counting_and_probability":  "numina_combinatorics",
    "precalculus":               "numina_calculus",
    "calculus":                  "numina_calculus",
    "statistics":                "numina_statistics",
    "probability":               "numina_statistics",
    # competition-source buckets (fallback when type not in map above)
    "cn_k12":                    "numina_algebra",
    "olympiads":                 "numina_olympiad",
    "amc_aime":                  "numina_competition",
    "synthetic_math":            "numina_synthetic",
}

# NuminaMath source → approximate difficulty (1=easy 2=medium 3=hard)
NUMINA_SOURCE_DIFFICULTY: Dict[str, int] = {
    "cn_k12":        1,
    "synthetic_math": 2,
    "amc_aime":       2,
    "olympiads":      3,
}

# OpenMathInstruct-2 problem_source → skill_id / difficulty
OPENMATH_SOURCE_TO_SKILL: Dict[str, str] = {
    "math":                  "openmath_algebra",   # overridden per-row by subject
    "amc_aime_1983_2024":    "openmath_competition",
    "synthetic_math":        "openmath_synthetic",
    "number_theory":         "openmath_number_theory",
}

OPENMATH_SOURCE_DIFFICULTY: Dict[str, int] = {
    "math":                 2,
    "amc_aime_1983_2024":   3,
    "synthetic_math":       1,
}

# OpenMathInstruct MATH-subject → skill_id (when problem_source == "math")
OPENMATH_MATH_SUBJECT_SKILL: Dict[str, str] = {
    "Algebra":                   "openmath_algebra",
    "Number Theory":             "openmath_number_theory",
    "Geometry":                  "openmath_geometry",
    "Counting & Probability":    "openmath_combinatorics",
    "Intermediate Algebra":      "openmath_algebra",
    "Prealgebra":                "openmath_prealgebra",
    "Precalculus":               "openmath_calculus",
    "Calculus":                  "openmath_calculus",
}

# ---------------------------------------------------------------------------
# Answer normalisation
# ---------------------------------------------------------------------------

_BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
_LATEX_FRAC = re.compile(r"\\frac\{(\d+)\}\{(\d+)\}")
_PLAIN_FRAC = re.compile(r"^(-?\d+)\s*/\s*(\d+)$")
_CURRENCY    = re.compile(r"(?:Rs\.?|USD|\$|€|£)\s*", re.IGNORECASE)
_UNICODE_MINUS = str.maketrans({"\u2212": "-", "−": "-"})


def extract_boxed(text: str) -> Optional[str]:
    """Return the last \\boxed{} contents from a solution string."""
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


def normalise_numeric(raw: str) -> Optional[str]:
    """
    Convert a raw answer string to a clean numeric string.

    Returns None for:
      - multi-value answers ("3 and 5")
      - symbolic expressions ("3\\sqrt{2}", "x+1")
      - inequalities
      - fractions where num/den exceed safe range
    """
    text = raw.strip()

    # Remove currency symbols and commas in numbers
    text = _CURRENCY.sub("", text)
    text = text.replace(",", "").translate(_UNICODE_MINUS).strip()

    # Skip if still contains words other than units
    if re.search(r"\b(and|or|none|no solution|undefined)\b", text, re.IGNORECASE):
        return None

    # Skip if contains letters (symbolic)
    if re.search(r"[a-zA-Z]", text):
        return None

    # Skip inequalities / ranges
    if re.search(r"[≤≥<>]", text):
        return None

    # Handle LaTeX fractions: \frac{3}{4}
    m = _LATEX_FRAC.fullmatch(text)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        if den:
            v = num / den
            return str(int(v)) if v == int(v) else f"{v:.4f}"
        return None

    # Handle plain fractions: 3/4
    m = _PLAIN_FRAC.match(text)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        if den:
            v = num / den
            return str(int(v)) if v == int(v) else f"{v:.4f}"
        return None

    # Handle percentage → decimal
    pct = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*%", text)
    if pct:
        v = float(pct.group(1))
        return str(int(v)) if v == int(v) else f"{v:.4f}"

    # Plain integer or decimal (possibly negative, possibly with trailing unit like "km")
    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*(?:[^0-9.\s].*)?\s*$", text)
    if m:
        val_str = m.group(1)
        try:
            v = float(val_str)
            return str(int(v)) if v == int(v) else val_str
        except ValueError:
            pass

    return None


# ---------------------------------------------------------------------------
# Solution → Step N: format
# ---------------------------------------------------------------------------

_SKIP_LINE_RE = re.compile(
    r"^\s*("
    r"\\boxed\{|"
    r"(Therefore|Thus|Hence|So),?\s+(the\s+)?(final\s+)?answer\s+is|"
    r"The\s+(final\s+)?answer\s+is|"
    r"Answer\s*[:=]"
    r")",
    re.IGNORECASE,
)


def solution_to_steps(solution: str, final_answer: str, max_steps: int = 18) -> str:
    """
    Convert an arbitrary CoT solution to the pipeline's Step N: format.

    Strategy:
      1. Split on newlines.
      2. Drop blank lines and lines that just announce the final answer
         (those are replaced by the explicit Final Answer: line).
      3. Strip any existing "Step N:" prefix to avoid double-numbering.
      4. Re-number as "Step 1:", "Step 2:", …
      5. Append "Final Answer: <answer>".
    """
    raw_lines = [l.strip() for l in solution.split("\n") if l.strip()]
    clean: List[str] = []
    for line in raw_lines:
        if _SKIP_LINE_RE.match(line):
            continue
        # Strip old step prefix
        line = re.sub(r"^Step\s*\d+\s*[:.)]\s*", "", line)
        if line:
            clean.append(line)

    # Cap to max_steps to keep token count reasonable
    clean = clean[:max_steps]

    if not clean:
        return f"Final Answer: {final_answer}"

    parts = [f"Step {i}: {line}" for i, line in enumerate(clean, 1)]
    return "\n".join(parts) + f"\nFinal Answer: {final_answer}"


# ---------------------------------------------------------------------------
# Record builders
# ---------------------------------------------------------------------------

def build_record(
    idx: int,
    split: str,
    source_name: str,
    skill_id: str,
    difficulty: int,
    question: str,
    solution_text: str,
    final_answer: str,
) -> Dict[str, Any]:
    assistant_content = solution_to_steps(solution_text, final_answer)
    return {
        "id":         f"{source_name.replace('/', '_')}_{split}_{idx}",
        "skill_id":   skill_id,
        "source":     source_name,
        "split":      split,
        "difficulty": difficulty,
        "task_type":  "solve",
        "messages": [
            {"role": "system",    "content": SOLVER_SYSTEM_PROMPT},
            {"role": "user",      "content": USER_WRAPPER.format(question=question.strip())},
            {"role": "assistant", "content": assistant_content},
        ],
    }


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def problem_hash(text: str) -> str:
    """Fast 16-char hash for near-dedup (exact-match on normalised text)."""
    normalised = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.md5(normalised.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# NuminaMath-CoT processing
# ---------------------------------------------------------------------------

def _numina_skill_and_difficulty(row: Dict) -> Tuple[str, int]:
    topic = (row.get("type") or "").lower().strip()
    source = (row.get("source") or "").lower().strip()

    skill = NUMINA_TYPE_TO_SKILL.get(topic)
    if skill is None:
        skill = NUMINA_TYPE_TO_SKILL.get(source, "numina_general")

    difficulty = NUMINA_SOURCE_DIFFICULTY.get(source, 2)
    return skill, difficulty


def iter_numina(
    max_samples: int,
    per_skill_cap: int,
    skip_olympiad: bool,
    seed: int,
) -> Iterator[Dict[str, Any]]:
    """
    Stream NuminaMath-CoT from HuggingFace and yield cleaned records.
    Uses per-skill quota to guarantee topic diversity.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        log.error("pip install datasets huggingface_hub")
        sys.exit(1)

    log.info("Streaming AI-MO/NuminaMath-CoT …")
    ds = load_dataset("AI-MO/NuminaMath-CoT", split="train", streaming=True,
                      trust_remote_code=True)

    skill_counts: Counter = Counter()
    seen_hashes: set = set()
    total_yielded = 0

    rng = random.Random(seed)

    for row in ds:
        if total_yielded >= max_samples:
            break

        problem  = (row.get("problem") or "").strip()
        solution = (row.get("solution") or "").strip()
        if not problem or not solution:
            continue

        # Extract and normalise answer from \boxed{}
        raw_answer = extract_boxed(solution)
        if raw_answer is None:
            continue
        final_answer = normalise_numeric(raw_answer)
        if final_answer is None:
            continue

        skill, difficulty = _numina_skill_and_difficulty(row)

        # Optionally skip very hard olympiad problems
        if skip_olympiad and skill == "numina_olympiad":
            continue

        # Per-skill cap to guarantee diversity
        if skill_counts[skill] >= per_skill_cap:
            continue

        # Dedup
        h = problem_hash(problem)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        skill_counts[skill] += 1
        total_yielded += 1

        yield build_record(
            idx=total_yielded,
            split="__assign__",
            source_name="AI-MO/NuminaMath-CoT",
            skill_id=skill,
            difficulty=difficulty,
            question=problem,
            solution_text=solution,
            final_answer=final_answer,
        )

    log.info("NuminaMath-CoT: yielded %d records | skill dist: %s",
             total_yielded, dict(skill_counts.most_common()))


# ---------------------------------------------------------------------------
# OpenMathInstruct-2 processing
# ---------------------------------------------------------------------------

def _openmath_skill_and_difficulty(row: Dict) -> Tuple[str, int]:
    src    = (row.get("problem_source") or "").lower().strip()
    subj   = (row.get("subject") or "").strip()

    if src == "math" and subj:
        skill = OPENMATH_MATH_SUBJECT_SKILL.get(subj, "openmath_algebra")
    else:
        skill = OPENMATH_SOURCE_TO_SKILL.get(src, "openmath_general")

    difficulty = OPENMATH_SOURCE_DIFFICULTY.get(src, 2)
    return skill, difficulty


def iter_openmath(
    max_samples: int,
    per_skill_cap: int,
    skip_gsm8k: bool,
    seed: int,
) -> Iterator[Dict[str, Any]]:
    """
    Stream OpenMathInstruct-2 from HuggingFace and yield cleaned records.
    Only yields rows where `is_correct_solution` is True (pre-verified by NVIDIA).
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        log.error("pip install datasets huggingface_hub")
        sys.exit(1)

    log.info("Streaming nvidia/OpenMathInstruct-2 (this may take a moment) …")
    ds = load_dataset(
        "nvidia/OpenMathInstruct-2",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    skill_counts: Counter = Counter()
    seen_hashes: set = set()
    total_yielded = 0

    for row in ds:
        if total_yielded >= max_samples:
            break

        # Filter: skip gsm8k (contamination risk)
        problem_src = (row.get("problem_source") or "").lower()
        if skip_gsm8k and "gsm8k" in problem_src:
            continue

        # Filter: only verified correct solutions
        if not row.get("is_correct_solution", True):
            continue

        problem  = (row.get("problem") or "").strip()
        solution = (row.get("generated_solution") or "").strip()
        expected = (row.get("expected_answer") or "").strip()

        if not problem or not solution or not expected:
            continue

        # Normalise the pre-extracted answer
        final_answer = normalise_numeric(expected)
        if final_answer is None:
            continue

        skill, difficulty = _openmath_skill_and_difficulty(row)

        # Per-skill cap
        if skill_counts[skill] >= per_skill_cap:
            continue

        # Dedup
        h = problem_hash(problem)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        skill_counts[skill] += 1
        total_yielded += 1

        yield build_record(
            idx=total_yielded,
            split="__assign__",
            source_name="nvidia/OpenMathInstruct-2",
            skill_id=skill,
            difficulty=difficulty,
            question=problem,
            solution_text=solution,
            final_answer=final_answer,
        )

    log.info("OpenMathInstruct-2: yielded %d records | skill dist: %s",
             total_yielded, dict(skill_counts.most_common()))


# ---------------------------------------------------------------------------
# Dataset stats printer
# ---------------------------------------------------------------------------

def print_stats(records: List[Dict], label: str) -> None:
    skill_c: Counter = Counter(r["skill_id"] for r in records)
    diff_c:  Counter = Counter(r["difficulty"] for r in records)
    src_c:   Counter = Counter(r["source"] for r in records)
    split_c: Counter = Counter(r["split"] for r in records)

    log.info("─── %s  (%d records) ───────────────────────────────", label, len(records))
    log.info("  by split:      %s", dict(split_c))
    log.info("  by source:     %s", dict(src_c))
    log.info("  by difficulty: %s", dict(sorted(diff_c.items())))
    log.info("  by skill_id:")
    for sk, cnt in skill_c.most_common():
        log.info("    %-40s  %5d", sk, cnt)


# ---------------------------------------------------------------------------
# Write JSONL
# ---------------------------------------------------------------------------

def write_jsonl(records: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log.info("Wrote %d records → %s", len(records), path)


# ---------------------------------------------------------------------------
# Train / val / test split  (stratified by skill_id)
# ---------------------------------------------------------------------------

def stratified_split(
    records: List[Dict],
    train_frac: float = 0.85,
    val_frac:   float = 0.10,
    seed:       int   = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Stratified split by skill_id so every skill appears in all three sets.
    Remaining fraction after train+val goes to test.
    """
    rng = random.Random(seed)

    by_skill: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        by_skill[r["skill_id"]].append(r)

    train_, val_, test_ = [], [], []
    for skill, items in by_skill.items():
        rng.shuffle(items)
        n = len(items)
        n_train = math.floor(n * train_frac)
        n_val   = math.floor(n * val_frac)
        train_ += items[:n_train]
        val_   += items[n_train: n_train + n_val]
        test_  += items[n_train + n_val:]

    for r in train_: r["split"] = "train"
    for r in val_:   r["split"] = "val"
    for r in test_:  r["split"] = "test"

    # Shuffle each split so skill interleaves during training
    rng.shuffle(train_)
    rng.shuffle(val_)
    rng.shuffle(test_)

    return train_, val_, test_


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build combined NuminaMath + OpenMathInstruct-2 training data."
    )
    p.add_argument("--output-dir",    default="data/sft",
                   help="Directory for output JSONL files.")
    p.add_argument("--max-numina",    type=int, default=20_000,
                   help="Max records from NuminaMath-CoT (default 20 000).")
    p.add_argument("--max-openmath",  type=int, default=15_000,
                   help="Max records from OpenMathInstruct-2 (default 15 000).")
    p.add_argument("--per-skill-cap", type=int, default=4_000,
                   help="Max records per skill_id to guarantee topic diversity.")
    p.add_argument("--skip-numina",   action="store_true",
                   help="Skip NuminaMath-CoT entirely.")
    p.add_argument("--skip-openmath", action="store_true",
                   help="Skip OpenMathInstruct-2 entirely.")
    p.add_argument("--skip-olympiad", action="store_true", default=True,
                   help="Skip numina_olympiad problems (too hard for 1.5B; default: True).")
    p.add_argument("--no-skip-olympiad", dest="skip_olympiad", action="store_false",
                   help="Include olympiad-level problems.")
    p.add_argument("--train-frac", type=float, default=0.85)
    p.add_argument("--val-frac",   type=float, default=0.10)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--dry-run",    action="store_true",
                   help="Process only 500 rows from each source and show stats (no write).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng  = random.Random(args.seed)

    if args.dry_run:
        args.max_numina   = min(args.max_numina,   500)
        args.max_openmath = min(args.max_openmath, 500)
        log.info("DRY RUN — capped at 500 samples per source, nothing written to disk.")

    all_records: List[Dict] = []

    # ── NuminaMath-CoT ────────────────────────────────────────────────────
    if not args.skip_numina:
        numina_recs = list(iter_numina(
            max_samples   = args.max_numina,
            per_skill_cap = args.per_skill_cap,
            skip_olympiad = args.skip_olympiad,
            seed          = args.seed,
        ))
        all_records.extend(numina_recs)
        log.info("NuminaMath-CoT collected: %d records", len(numina_recs))
    else:
        log.info("Skipping NuminaMath-CoT (--skip-numina).")

    # ── OpenMathInstruct-2 ────────────────────────────────────────────────
    if not args.skip_openmath:
        openmath_recs = list(iter_openmath(
            max_samples   = args.max_openmath,
            per_skill_cap = args.per_skill_cap,
            skip_gsm8k    = True,
            seed          = args.seed,
        ))
        all_records.extend(openmath_recs)
        log.info("OpenMathInstruct-2 collected: %d records", len(openmath_recs))
    else:
        log.info("Skipping OpenMathInstruct-2 (--skip-openmath).")

    if not all_records:
        log.error("No records collected — check dataset availability.")
        sys.exit(1)

    # ── Deduplicate across sources ─────────────────────────────────────────
    seen: set = set()
    deduped: List[Dict] = []
    for r in all_records:
        question = r["messages"][1]["content"]
        h = problem_hash(question)
        if h not in seen:
            seen.add(h)
            deduped.append(r)

    log.info("After cross-source dedup: %d → %d records  (removed %d dupes)",
             len(all_records), len(deduped), len(all_records) - len(deduped))

    # ── Stratified split ──────────────────────────────────────────────────
    train_recs, val_recs, test_recs = stratified_split(
        deduped, args.train_frac, args.val_frac, args.seed
    )

    print_stats(train_recs + val_recs + test_recs, "COMBINED DATASET")

    # ── Write outputs ─────────────────────────────────────────────────────
    if args.dry_run:
        log.info("DRY RUN complete — no files written.")
        log.info("  would write: combined_train.jsonl  (%d rows)", len(train_recs))
        log.info("  would write: combined_val.jsonl    (%d rows)", len(val_recs))
        log.info("  would write: combined_test.jsonl   (%d rows)", len(test_recs))
        log.info("Sample record:")
        print(json.dumps(train_recs[0], indent=2, ensure_ascii=False))
        return

    out = Path(args.output_dir)
    write_jsonl(train_recs, out / "combined_train.jsonl")
    write_jsonl(val_recs,   out / "combined_val.jsonl")
    write_jsonl(test_recs,  out / "combined_test.jsonl")

    log.info("")
    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║  Pipeline complete.  Next step:                              ║")
    log.info("║    bash launch_grpo_combined.sh                              ║")
    log.info("╚══════════════════════════════════════════════════════════════╝")
    log.info("  train : %6d rows  → %s/combined_train.jsonl", len(train_recs), out)
    log.info("  val   : %6d rows  → %s/combined_val.jsonl",   len(val_recs),   out)
    log.info("  test  : %6d rows  → %s/combined_test.jsonl",  len(test_recs),  out)
    log.info("")
    log.info("Skill coverage (for ZPD CurriculumManager):")
    skill_c = Counter(r["skill_id"] for r in train_recs)
    for sk, cnt in sorted(skill_c.items()):
        log.info("  %-40s  %5d train samples", sk, cnt)


if __name__ == "__main__":
    main()
