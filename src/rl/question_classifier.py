"""
Question classification and difficulty estimation utilities.

This module provides a deterministic, low-latency classifier for:
- Primary/secondary topic detection
- Post-hoc difficulty estimation from generated solutions
- Basic question clarity checks
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


TOPIC_KEYWORDS = {
    "basic_arithmetic": [
        "add",
        "sum",
        "subtract",
        "difference",
        "total",
        "altogether",
    ],
    "single_step_word_problems": [
        "how many",
        "left",
        "remain",
        "altogether",
    ],
    "fractions": [
        "fraction",
        "fractions",
        "numerator",
        "denominator",
        "half",
        "quarter",
        "third",
        "fourth",
        "fifth",
    ],
    "percentages": [
        "percent",
        "percentage",
        "% ",
        "discount",
        "tax",
        "increase",
        "decrease",
    ],
    "ratios": [
        "ratio",
        "proportion",
        "per",
        "for every",
        "rate",
    ],
    "money_problems": [
        "dollar",
        "dollars",
        "cents",
        "$",
        "price",
        "cost",
        "buy",
        "sell",
    ],
    "time_distance": [
        "hour",
        "minute",
        "second",
        "km",
        "mile",
        "speed",
        "distance",
        "travel",
    ],
    "multi_step_reasoning": [
        "then",
        "after",
        "before",
        "remaining",
        "each",
        "twice",
        "three times",
    ],
    "algebra": [
        "solve for",
        "equation",
        "variable",
        "x",
        "y",
        "unknown",
    ],
    "mixed_operations": [
        "combined",
        "multiple operations",
        "in total",
    ],
    "comparison_problems": [
        "more than",
        "less than",
        "difference",
        "compared",
    ],
    "optimization_problems": [
        "maximum",
        "minimum",
        "optimize",
        "best",
    ],
    # ── AQuA-RAT additions ────────────────────────────────────────────────
    "number_theory": [
        "prime",
        "divisible",
        "remainder",
        "factor",
        "multiple",
        "divisor",
        "integer divisible",
        "mod",
    ],
    "profit_loss": [
        "profit",
        "loss",
        "cost price",
        "selling price",
        "markup",
        "gain",
        "cp",
        "sp",
    ],
    "interest": [
        "simple interest",
        "compound interest",
        "principal",
        "rate of interest",
        "annually",
        "quarterly",
        "semi-annually",
        "p.a.",
    ],
    "sets": [
        "neither",
        "both",
        "only one",
        "union",
        "intersection",
        "venn",
        "at least one",
    ],
    "combinatorics": [
        "combination",
        "permutation",
        "arrangement",
        "ways to select",
        "ways to choose",
        "how many ways",
        "nCr",
        "nPr",
    ],
    "sequences": [
        "sequence",
        "series",
        "arithmetic progression",
        "geometric progression",
        "nth term",
        "common difference",
        "common ratio",
    ],
    "probability": [
        "probability",
        "chance",
        "likely",
        "favorable",
        "event",
        "random",
        "draw",
    ],
    "work_time": [
        "work together",
        "working together",
        "alone in",
        "complete the job",
        "working rate",
        "finish the work",
        "days to complete",
        "rate of work",
    ],
    # ── NuminaMath / OpenMathInstruct additions ───────────────────────────
    "geometry": [
        "triangle",
        "circle",
        "rectangle",
        "polygon",
        "area",
        "perimeter",
        "angle",
        "radius",
        "diameter",
        "hypotenuse",
        "coordinate",
        "tangent",
        "bisector",
        "congruent",
        "similar",
        "parallel",
        "perpendicular",
        "volume",
        "surface area",
        "right angle",
    ],
    "calculus": [
        "derivative",
        "differentiate",
        "integrate",
        "dy/dx",
        "f'(x)",
        "definite integral",
        "indefinite integral",
        "slope of the tangent",
        "rate of change",
        "inflection point",
    ],
    "statistics": [
        "mean",
        "median",
        "mode",
        "standard deviation",
        "variance",
        "average",
        "data set",
        "frequency",
        "histogram",
        "distribution",
        "normal distribution",
        "expected value",
        "outlier",
        "quartile",
        "range of data",
    ],
    "competition_math": [
        "positive integers",
        "integer solutions",
        "divisible by",
        "remainder when",
        "relatively prime",
        "greatest common divisor",
        "least common multiple",
        "prove that",
        "diophantine",
        "congruent modulo",
        "sum of digits",
    ],
}

TOPIC_LIST = list(TOPIC_KEYWORDS.keys())


@dataclass
class TopicClassification:
    primary_topic: str
    secondary_topics: List[str]
    confidence: float
    signals_used: List[str]
    keyword_scores: Dict[str, float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "primary_topic": self.primary_topic,
            "secondary_topics": self.secondary_topics,
            "confidence": self.confidence,
            "signals_used": self.signals_used,
            "keyword_scores": self.keyword_scores,
        }


class QuestionClassifier:
    """Deterministic classifier for curriculum-guided question generation."""

    _step_pattern = re.compile(r"^\s*step\s+\d+\s*:", re.IGNORECASE | re.MULTILINE)
    _number_pattern = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")
    _fraction_pattern = re.compile(r"\d+\s*/\s*\d+")
    _nested_op_pattern = re.compile(r"\([^()]*[+\-*/][^()]*\)")

    # High-confidence single-phrase signals that override the scoring formula.
    # Ordered: more specific first.  If ANY of these patterns match, the
    # corresponding topic wins regardless of keyword counts.
    _PRIORITY_SIGNALS: List[Tuple[re.Pattern, str]] = [
        # Calculus — "integrate" before ratios can steal "rate" as a substring
        (re.compile(r"\b(derivative|differentiate|integrate|d/dx|dy/dx|f'\s*\(|indefinite integral|definite integral|rate of change|inflection point)\b", re.I), "calculus"),
        # Geometry
        (re.compile(r"\b(triangle|rectangle|polygon|perimeter|circumference|hypotenuse|right angle|surface area|volume of|radius|diameter)\b", re.I), "geometry"),
        # Statistics
        (re.compile(r"\b(standard deviation|variance|median|normal distribution|expected value)\b", re.I), "statistics"),
        # Competition math
        (re.compile(r"\b(divisible by|remainder when|relatively prime|greatest common divisor|least common multiple|diophantine|congruent modulo|sum of digits)\b", re.I), "competition_math"),
        (re.compile(r"\bpositive integers?\b.{0,40}\bdivisible\b", re.I), "competition_math"),
        # Time-distance (speeds? covers plural; match across short gap)
        (re.compile(r"\bspeeds?\b.{0,80}\b(meet|distance|time|arrive|travel)\b", re.I), "time_distance"),
        (re.compile(r"\b(km/h|mph|miles per hour|km per hour)\b", re.I), "time_distance"),
        # Combinatorics — "how many ways" beats single_step "how many"
        (re.compile(r"\bhow many ways\b", re.I), "combinatorics"),
        (re.compile(r"\b(arrangements?|permutations?|combinations?) of\b", re.I), "combinatorics"),
        # Probability — "probability" contains "y" which would otherwise hit algebra
        (re.compile(r"\b(probability|the chance that|likelihood of)\b", re.I), "probability"),
    ]

    def classify_topic(self, question: str, solution: Optional[str] = None) -> Dict[str, object]:
        """Return primary/secondary topics with confidence."""
        text = (question or "").lower()

        # Fast path: high-confidence priority signals bypass scoring
        for pattern, topic in self._PRIORITY_SIGNALS:
            if pattern.search(text):
                return TopicClassification(
                    primary_topic=topic,
                    secondary_topics=[],
                    confidence=0.95,
                    signals_used=["priority"],
                    keyword_scores={topic: 0.95},
                ).to_dict()

        keyword_scores = {topic: self._keyword_score(text, words) for topic, words in TOPIC_KEYWORDS.items()}

        signals_used = ["keyword"]
        primary_topic = max(keyword_scores, key=keyword_scores.get)
        confidence = keyword_scores[primary_topic]

        if self._fraction_pattern.search(text):
            keyword_scores["fractions"] += 0.25
            primary_topic = max(keyword_scores, key=keyword_scores.get)
            confidence = max(confidence, min(1.0, keyword_scores[primary_topic]))
            signals_used.append("pattern")

        if "%" in text:
            keyword_scores["percentages"] += 0.25
            primary_topic = max(keyword_scores, key=keyword_scores.get)
            confidence = max(confidence, min(1.0, keyword_scores[primary_topic]))
            if "pattern" not in signals_used:
                signals_used.append("pattern")

        if solution:
            op_topic = self._infer_topic_from_solution(solution)
            if op_topic:
                primary_topic = op_topic
                confidence = max(confidence, 0.9)
                signals_used.append("solution_ops")

        secondary_topics = [
            topic
            for topic, score in sorted(keyword_scores.items(), key=lambda item: item[1], reverse=True)
            if topic != primary_topic and score >= 0.2
        ][:3]

        return TopicClassification(
            primary_topic=primary_topic,
            secondary_topics=secondary_topics,
            confidence=min(1.0, confidence),
            signals_used=signals_used,
            keyword_scores=keyword_scores,
        ).to_dict()

    def estimate_difficulty(
        self,
        question: str,
        solution: str,
        consensus_result: Optional[Dict[str, object]] = None,
    ) -> float:
        """
        Estimate difficulty using post-solution signals.

        40%: step complexity
        30%: numeric complexity
        30%: consensus disagreement complexity
        """
        step_score = self._step_complexity(solution)
        number_score = self._numeric_complexity(question, solution)
        consensus_score = self._consensus_difficulty(consensus_result)
        difficulty = 0.4 * step_score + 0.3 * number_score + 0.3 * consensus_score
        return max(0.0, min(1.0, difficulty))

    def check_clarity(self, question: str) -> float:
        """Score question clarity in [0, 1] from low-cost heuristics."""
        text = (question or "").strip()
        if not text:
            return 0.0

        lower = text.lower()
        has_numbers = 1.0 if self._number_pattern.search(lower) else 0.0
        has_question = 1.0 if ("?" in lower or re.search(r"\b(find|calculate|how many|what is|determine|compute|evaluate|express|simplify|solve)\b", lower)) else 0.0
        words = lower.split()
        length_ok = 1.0 if 8 <= len(words) <= 120 else 0.3
        contradiction = 1.0 if not re.search(r"\b(impossible|contradiction|undefined)\b", lower) else 0.0

        return max(0.0, min(1.0, 0.3 * has_numbers + 0.3 * has_question + 0.2 * length_ok + 0.2 * contradiction))

    def _keyword_score(self, text: str, keywords: List[str]) -> float:
        if not keywords:
            return 0.0
        hits = 0
        for kw in keywords:
            if kw in text:
                hits += 1
        return min(1.0, hits / max(2.0, len(keywords) * 0.6))

    def _infer_topic_from_solution(self, solution: str) -> Optional[str]:
        text = (solution or "").lower()
        if not text:
            return None

        has_fraction = bool(self._fraction_pattern.search(text))
        has_percent = "%" in text or "percent" in text
        has_variable = bool(re.search(r"\b[x-y]\b|\bsolve\b|\bequation\b", text))
        has_division = "/" in text or "divide" in text
        has_mul = "*" in text or "multiply" in text
        has_add_sub = any(op in text for op in ["+", "-", "add", "subtract"])

        # Higher-specificity signals come first
        if any(kw in text for kw in ["derivative", "dy/dx", "f'(", "differentiat", "integrat"]):
            return "calculus"
        if any(kw in text for kw in ["triangle", "circle", "area =", "perimeter", "radius", "angle", "coordinate"]):
            return "geometry"
        if any(kw in text for kw in ["modulo", "gcd", "lcm", "divisible by", "remainder", "prime"]):
            return "competition_math"
        if any(kw in text for kw in ["mean =", "median", "standard deviation", "variance"]):
            return "statistics"
        if has_variable:
            return "algebra"
        if has_percent:
            return "percentages"
        if has_fraction:
            return "fractions"
        if has_division and ("km" in text or "mile" in text or "hour" in text):
            return "time_distance"
        if has_division and has_mul and has_add_sub:
            return "mixed_operations"
        if has_division or has_mul:
            return "multi_step_reasoning"
        return None

    def _step_complexity(self, solution: str) -> float:
        text = solution or ""
        step_count = len(self._step_pattern.findall(text))
        if step_count == 0:
            step_count = max(1, text.count("\n") // 2)
        step_score = min(1.0, step_count / 5.0)

        lowered = text.lower()
        op_score = 0.0
        if any(token in lowered for token in ["+", "-", "add", "subtract"]):
            op_score = max(op_score, 0.3)
        if any(token in lowered for token in ["*", "multiply"]):
            op_score = max(op_score, 0.55)
        if any(token in lowered for token in ["/", "divide"]):
            op_score = max(op_score, 0.7)
        if self._nested_op_pattern.search(lowered):
            op_score = max(op_score, 0.85)

        return max(0.0, min(1.0, 0.6 * step_score + 0.4 * op_score))

    def _numeric_complexity(self, question: str, solution: str) -> float:
        text = f"{question or ''} {solution or ''}"
        numbers = self._number_pattern.findall(text)
        if not numbers:
            return 0.0

        max_abs = 0.0
        has_decimal = False
        has_fraction = False
        for token in numbers:
            if "/" in token:
                has_fraction = True
                parts = token.split("/")
                if len(parts) == 2 and parts[1] != "0":
                    try:
                        value = abs(float(parts[0]) / float(parts[1]))
                        max_abs = max(max_abs, value)
                    except ValueError:
                        pass
            else:
                if "." in token:
                    has_decimal = True
                try:
                    max_abs = max(max_abs, abs(float(token)))
                except ValueError:
                    pass

        magnitude_score = 0.2
        if max_abs >= 1000:
            magnitude_score = 0.8
        elif max_abs >= 100:
            magnitude_score = 0.6
        elif max_abs >= 20:
            magnitude_score = 0.4

        numeric_bonus = 0.0
        if has_decimal:
            numeric_bonus += 0.15
        if has_fraction:
            numeric_bonus += 0.2

        return max(0.0, min(1.0, magnitude_score + numeric_bonus))

    def _consensus_difficulty(self, consensus_result: Optional[Dict[str, object]]) -> float:
        if not consensus_result:
            return 0.5
        strength = float(consensus_result.get("consensus_strength", 0.0))
        return max(0.0, min(1.0, 1.0 - strength))
