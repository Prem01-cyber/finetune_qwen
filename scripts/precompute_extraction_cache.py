"""
Offline step-chain extraction cache builder.

Run this once before training to pre-extract structured step chains from all
grounded training data (GSM8K + MATH).  The resulting cache file is passed to
run_grpo_training.py via --extraction-cache so the extractor LLM is never
called for fixed training examples — only novel self-play solutions require
live extraction during training.

Usage
-----
    python scripts/precompute_extraction_cache.py \\
        --gsm8k-data  data/sft/gsm8k_sft.jsonl \\
        --math-data   data/sft/math_sft.jsonl \\
        --output-cache data/extraction_cache.json \\
        --extractor-model Qwen/Qwen2.5-0.5B-Instruct \\
        --device cuda

Cache key: md5(question + "\\n" + solution) — keying on both prevents
collisions when two MATH problems share identical solution text.
Entries for solutions the extractor cannot parse are stored with
success=False so training never re-attempts and correctly penalises them.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import sys
from typing import List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def load_jsonl(path: str) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def collect_qa_pairs(records: list[dict]) -> List[Tuple[str, str]]:
    """
    Extract (question, solution) pairs from dataset records.

    Returns pairs where both fields are non-empty.  Falls back to empty
    string for the question when only the solution field is present.
    """
    pairs: List[Tuple[str, str]] = []
    for rec in records:
        sol = (
            rec.get("solution")
            or rec.get("output")
            or rec.get("response")
            or ""
        )
        q = (
            rec.get("question")
            or rec.get("problem")
            or rec.get("input")
            or ""
        )
        if sol.strip():
            pairs.append((q.strip(), sol.strip()))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-extract step chains for grounded training data."
    )
    parser.add_argument(
        "--gsm8k-data", required=True,
        help="Path to GSM8K training JSONL (e.g. data/sft/gsm8k_sft.jsonl).",
    )
    parser.add_argument(
        "--math-data", default=None,
        help="Optional path to MATH training JSONL. If provided, those solutions "
             "are also extracted and added to the cache.",
    )
    parser.add_argument(
        "--output-cache", required=True,
        help="Destination JSON file for the extraction cache.",
    )
    parser.add_argument(
        "--extractor-model", default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model ID for the step chain extractor. Default Qwen/Qwen2.5-0.5B-Instruct.",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device for the extractor model (default: cuda).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Reserved for future batched extraction. Currently always 1.",
    )
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading GSM8K data from: %s", args.gsm8k_data)
    gsm8k_records = load_jsonl(args.gsm8k_data)
    qa_pairs = collect_qa_pairs(gsm8k_records)
    logger.info("GSM8K: %d (question, solution) pairs", len(qa_pairs))

    if args.math_data:
        logger.info("Loading MATH data from: %s", args.math_data)
        math_records = load_jsonl(args.math_data)
        math_pairs = collect_qa_pairs(math_records)
        logger.info("MATH: %d (question, solution) pairs", len(math_pairs))
        qa_pairs += math_pairs

    if not qa_pairs:
        logger.error(
            "No solutions found in provided files. "
            "Check field names (question/problem/input + solution/output/response)."
        )
        sys.exit(1)

    # Deduplicate by (question, solution) content
    # Two different MATH problems can have identical solution text but different
    # questions — the question+solution key keeps them distinct in the cache.
    seen: set = set()
    unique_pairs: List[Tuple[str, str]] = []
    for q, sol in qa_pairs:
        key = (q, sol)
        if key not in seen:
            seen.add(key)
            unique_pairs.append((q, sol))

    logger.info(
        "Total: %d pairs (%d unique after dedup)", len(qa_pairs), len(unique_pairs)
    )

    # ── Load extractor ────────────────────────────────────────────────────────
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    from src.rl.unified_accuracy import StepChainExtractor

    extractor = StepChainExtractor(
        model_name=args.extractor_model,
        device=args.device,
        cache_path=args.output_cache,   # load existing cache if present (resume)
    )

    # ── Build cache ───────────────────────────────────────────────────────────
    already_cached = len(extractor._cache)
    if already_cached:
        logger.info("Resuming: %d entries already in cache", already_cached)

    extractor.build_cache(unique_pairs)

    # ── Save ──────────────────────────────────────────────────────────────────
    extractor.save_cache()
    logger.info(
        "Done. Cache contains %d entries → %s",
        len(extractor._cache),
        args.output_cache,
    )


if __name__ == "__main__":
    main()
