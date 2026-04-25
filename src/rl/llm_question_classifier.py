"""
LLM-backed question classifier that replaces the keyword-regex approach.

The already-loaded policy model (Qwen2.5-1.5B-Instruct) is used as the
classifier brain via a short structured prompt.  Inference runs under
``torch.no_grad()`` so it does not affect training gradients.

Interface is identical to ``QuestionClassifier``, so it is a drop-in
replacement for the ``classifier`` argument of ``QuestionQualityEvaluator``.

Fallback chain
--------------
  1. Cache hit        → instant (0 ms)
  2. LLM generation  → ~60-120 ms (8 new tokens, greedy, 1.5B model)
  3. Regex fallback   → ~1 ms  (on any error or unparseable output)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import torch

from src.rl.question_classifier import TOPIC_LIST, QuestionClassifier

logger = logging.getLogger(__name__)

# ── Prompt constants ─────────────────────────────────────────────────────────

_TOPIC_CSV = "\n".join(f"  {t}" for t in TOPIC_LIST)

_SYSTEM_PROMPT = "You are a precise math topic classifier. Reply with exactly one topic name."

_USER_TEMPLATE = (
    "Classify the math problem below into EXACTLY ONE topic from this list:\n"
    "{topics}\n\n"
    "Problem:\n{problem}\n\n"
    "Reply with only the topic name, nothing else."
)

_TOPIC_SET = set(TOPIC_LIST)

# Normalise common LLM output variations → canonical topic names
_ALIAS_MAP: Dict[str, str] = {
    # spacing / dash variants
    "competition math":      "competition_math",
    "competition-math":      "competition_math",
    "basic arithmetic":      "basic_arithmetic",
    "number theory":         "number_theory",
    "single step":           "single_step_word_problems",
    "single-step":           "single_step_word_problems",
    "word problems":         "single_step_word_problems",
    "word problem":          "single_step_word_problems",
    "multi step":            "multi_step_reasoning",
    "multi-step":            "multi_step_reasoning",
    "time distance":         "time_distance",
    "time-distance":         "time_distance",
    "money problems":        "money_problems",
    "profit loss":           "profit_loss",
    "profit and loss":       "profit_loss",
    "work time":             "work_time",
    "work rate":             "work_time",
    "mixed operations":      "mixed_operations",
    "mixed-operations":      "mixed_operations",
    "comparison problems":   "comparison_problems",
    "optimization problems": "optimization_problems",
    # common shorthand
    "geo":       "geometry",
    "calc":      "calculus",
    "stats":     "statistics",
    "stat":      "statistics",
    "arith":     "basic_arithmetic",
    "combi":     "combinatorics",
    "combo":     "combinatorics",
    "prob":      "probability",
    "seq":       "sequences",
    "percent":   "percentages",
    "alg":       "algebra",
}


def _parse_topic(raw: str) -> Optional[str]:
    """
    Extract a canonical topic name from raw LLM output.

    Returns None if the output cannot be mapped to any known topic.
    """
    text = raw.strip().lower()
    # Take first line only (model sometimes adds explanation after newline)
    first_line = text.split("\n")[0].strip()
    # Remove surrounding quotes or punctuation
    first_line = re.sub(r'^["\']|["\',.:;]$', "", first_line).strip()

    if first_line in _TOPIC_SET:
        return first_line

    normalised = first_line.replace(" ", "_").replace("-", "_")
    if normalised in _TOPIC_SET:
        return normalised

    if first_line in _ALIAS_MAP:
        return _ALIAS_MAP[first_line]
    if normalised in _ALIAS_MAP:
        return _ALIAS_MAP[normalised]

    # Substring scan: accept if exactly one topic is contained
    matches = [t for t in TOPIC_LIST if t in first_line or first_line in t]
    if len(matches) == 1:
        return matches[0]

    return None


# ── LLM Classifier ────────────────────────────────────────────────────────────


class LLMQuestionClassifier(QuestionClassifier):
    """
    Uses the loaded policy model to classify math problem topics.

    Inherits all ``estimate_difficulty``, ``check_clarity``, and
    ``_infer_topic_from_solution`` methods from ``QuestionClassifier`` —
    only ``classify_topic`` is overridden with LLM inference.

    Parameters
    ----------
    model       : The loaded CausalLM policy model (already in VRAM).
    tokenizer   : Matching tokenizer.
    device      : torch.device or str.
    cache_size  : LRU-style cache capacity (number of questions).
    max_retries : Number of greedy attempts before regex fallback.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: Any,
        cache_size: int = 10_000,
        max_retries: int = 1,
    ) -> None:
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
        self._device = torch.device(device) if isinstance(device, str) else device
        self._cache: Dict[str, Dict] = {}
        self._cache_size = cache_size
        self._max_retries = max_retries
        self._stats = {"llm_hits": 0, "cache_hits": 0, "fallback_hits": 0}
        logger.info(
            "LLMQuestionClassifier ready  (model=%s, cache=%d, topics=%d)",
            type(model).__name__,
            cache_size,
            len(TOPIC_LIST),
        )

    # ------------------------------------------------------------------
    # Public API (same signature as QuestionClassifier)
    # ------------------------------------------------------------------

    def classify_topic(
        self,
        question: str,
        solution: Optional[str] = None,
    ) -> Dict[str, object]:
        """
        Classify *question* into one of the 24 curriculum topics.

        Uses the LLM for fresh questions and a cache for repeated ones.
        Falls back to regex keyword matching on any error.
        """
        cache_key = (question or "")[:300]

        if cache_key in self._cache:
            self._stats["cache_hits"] += 1
            return self._cache[cache_key]

        result = self._classify_with_llm(question, solution)

        # Evict oldest entry when cache is full (FIFO approximation)
        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = result
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_with_llm(
        self,
        question: str,
        solution: Optional[str],
    ) -> Dict[str, object]:
        """Run LLM inference and parse the topic; fall back to regex."""
        try:
            topic = self._llm_infer_topic(question)
            if topic is not None:
                self._stats["llm_hits"] += 1
                return {
                    "primary_topic":    topic,
                    "secondary_topics": self._llm_secondary(topic, question, solution),
                    "confidence":       0.92,
                    "signals_used":     ["llm"],
                    "keyword_scores":   {topic: 0.92},
                }
        except Exception as exc:
            logger.debug("LLM classifier error: %s — using regex fallback.", exc)

        # Regex fallback (inherited from QuestionClassifier)
        self._stats["fallback_hits"] += 1
        return super().classify_topic(question, solution)

    @torch.no_grad()
    def _llm_infer_topic(self, question: str) -> Optional[str]:
        """
        Generate a topic prediction using the policy model (greedy, 8 tokens).

        Returns a canonical topic string, or None if the output can't be parsed.
        """
        prompt_text = _USER_TEMPLATE.format(
            topics=_TOPIC_CSV,
            problem=(question or "")[:400],  # truncate very long problems
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt_text},
        ]
        input_text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = self._tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self._device)
        prompt_len = enc["input_ids"].shape[1]

        out = self._model.generate(
            **enc,
            max_new_tokens=12,
            do_sample=False,
            temperature=1.0,
            pad_token_id=self._tokenizer.eos_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
        )

        new_ids = out[0][prompt_len:]
        raw = self._tokenizer.decode(new_ids, skip_special_tokens=True)
        return _parse_topic(raw)

    def _llm_secondary(
        self,
        primary: str,
        question: str,
        solution: Optional[str],
    ) -> List[str]:
        """
        Cheap secondary topics via regex (not worth a second LLM call).
        Re-uses the parent's keyword_scores to find runner-up topics.
        """
        text = (question or "").lower()
        kw_scores = {
            t: self._keyword_score(text, words)
            for t, words in __import__(
                "src.rl.question_classifier", fromlist=["TOPIC_KEYWORDS"]
            ).TOPIC_KEYWORDS.items()
        }
        secondary = [
            t for t, sc in sorted(kw_scores.items(), key=lambda x: x[1], reverse=True)
            if t != primary and sc >= 0.2
        ][:3]
        return secondary

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def log_stats(self) -> None:
        total = sum(self._stats.values())
        if total == 0:
            return
        logger.info(
            "LLMClassifier  cache=%.0f%%  llm=%.0f%%  fallback=%.0f%%  (cache_size=%d/%d)",
            100 * self._stats["cache_hits"]    / total,
            100 * self._stats["llm_hits"]      / total,
            100 * self._stats["fallback_hits"] / total,
            len(self._cache),
            self._cache_size,
        )
