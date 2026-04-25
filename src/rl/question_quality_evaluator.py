"""
Question quality evaluator for curriculum-guided dual-task training.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.rl.question_classifier import QuestionClassifier


@dataclass
class QuestionEvalResult:
    overall_score: float
    topic_match: float
    difficulty_score: float
    clarity: float
    solvability_score: float
    novelty_combined: float
    measured_difficulty: float
    detected_topic: Dict[str, object]
    novelty: Dict[str, float]
    solvability: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "overall_score": self.overall_score,
            "topic_match": self.topic_match,
            "difficulty_score": self.difficulty_score,
            "clarity": self.clarity,
            "solvability_score": self.solvability_score,
            "novelty_combined": self.novelty_combined,
            "measured_difficulty": self.measured_difficulty,
            "detected_topic": self.detected_topic,
            "novelty": self.novelty,
            "solvability": self.solvability,
        }


class QuestionQualityEvaluator:
    """Evaluate generated question quality for curriculum reward shaping."""

    def __init__(
        self,
        reference_questions: Optional[List[str]] = None,
        classifier: Optional[QuestionClassifier] = None,
        novelty_window_size: int = 500,   # raised from 100: 5 SP/iter → fills in ~100 iters
    ):
        self.reference_questions = reference_questions or []
        self.classifier = classifier or QuestionClassifier()
        self.novelty_window_size = novelty_window_size
        self.recent_questions: List[str] = []
        # Pre-compute and cache reference n-gram sets once at init.
        self._reference_ngrams = [self._extract_ngrams(q.lower()) for q in self.reference_questions]
        # Rolling cache of n-gram sets for recent questions (avoids recomputing every call).
        self._recent_ngrams: List[set] = []

    def evaluate(
        self,
        question: str,
        solution: str,
        consensus_result: Optional[Dict[str, object]],
        target_topic: str,
        target_difficulty: float,
    ) -> Dict[str, object]:
        detected_topic = self.classifier.classify_topic(question=question, solution=solution)
        topic_match = self._topic_match_score(detected_topic, target_topic)

        measured_difficulty = self.classifier.estimate_difficulty(
            question=question,
            solution=solution,
            consensus_result=consensus_result,
        )
        difficulty_score = max(0.0, 1.0 - 2.0 * abs(measured_difficulty - target_difficulty))

        clarity = self.classifier.check_clarity(question)
        novelty = self.compute_novelty_score(question)
        solvability = self.assess_solvability(question, solution, consensus_result)

        overall = (
            0.25 * topic_match
            + 0.15 * difficulty_score
            + 0.20 * clarity
            + 0.20 * float(solvability["score"])
            + 0.20 * novelty["combined"]   # raised 0.10→0.20; taken from difficulty_score
        )

        return QuestionEvalResult(
            overall_score=max(0.0, min(1.0, overall)),
            topic_match=topic_match,
            difficulty_score=difficulty_score,
            clarity=clarity,
            solvability_score=float(solvability["score"]),
            novelty_combined=novelty["combined"],
            measured_difficulty=measured_difficulty,
            detected_topic=detected_topic,
            novelty=novelty,
            solvability=solvability,
        ).to_dict()

    def compute_novelty_score(self, question: str) -> Dict[str, float]:
        dataset_novelty = self._novelty_against_reference(question, self._reference_ngrams)
        # Use cached recent n-gram sets instead of recomputing from strings each call (O(n²)→O(n)).
        session_novelty = self._novelty_against_reference(question, self._recent_ngrams)
        # Weight dataset novelty higher (60%) — comparing against 8k GSM8K questions
        # is a stable, meaningful signal. Session novelty (40%) guards against
        # the model looping the same question template within a run.
        combined = max(0.0, min(1.0, 0.60 * dataset_novelty + 0.40 * session_novelty))

        self.recent_questions.append(question)
        self.recent_questions = self.recent_questions[-self.novelty_window_size:]
        # Keep n-gram cache in sync with the question window.
        self._recent_ngrams.append(self._extract_ngrams(question.lower()))
        self._recent_ngrams = self._recent_ngrams[-self.novelty_window_size:]

        return {
            "combined": combined,
            "dataset_novelty": dataset_novelty,
            "session_novelty": session_novelty,
        }

    def assess_solvability(
        self,
        question: str,
        solution: str,
        consensus_result: Optional[Dict[str, object]],
    ) -> Dict[str, object]:
        q_lower = (question or "").lower()
        has_numbers = bool(re.search(r"\d", q_lower))
        has_question = ("?" in q_lower) or bool(re.search(
            r"\b(find|calculate|how many|what is|determine|compute|evaluate|express|simplify|solve)\b",
            q_lower,
        ))
        length_ok = 8 <= len(q_lower.split()) <= 120
        if not (has_numbers and has_question and length_ok):
            return {"solvable": False, "reason": "syntactic_failure", "score": 0.0}

        has_contradiction = bool(re.search(r"\b(impossible|cannot|undefined)\b", q_lower))
        if has_contradiction:
            return {"solvable": False, "reason": "semantic_failure", "score": 0.3}

        # PRM-based arithmetic quality check (replaces SymPy step verification).
        # consensus_strength = prm_mean: average PRM score across all reasoning steps.
        # A low PRM mean means the model produced inconsistent or incorrect reasoning,
        # which strongly signals the question is ambiguous, contradictory, or unsolvable.
        # PRM understands full mathematical semantics — it catches errors that SymPy
        # misses (e.g., wrong logic, incorrect setups) while not failing on valid prose.
        if consensus_result:
            confidence = float(consensus_result.get("consensus_strength", 0.5))
            if confidence < 0.30:
                # PRM rejects most steps → solution is invalid → question is likely unsolvable
                return {"solvable": False, "reason": "low_prm_confidence", "score": 0.5}
            if not bool(consensus_result.get("has_majority", False)):
                # PRM is borderline (0.30–0.49) → uncertain solvability
                return {"solvable": False, "reason": "no_consensus", "score": 0.6}
        else:
            confidence = 0.5

        return {
            "solvable": True,
            "reason": "fully_solvable",
            "score": 1.0,
            "confidence": confidence,
        }

    @staticmethod
    def _extract_ngrams(text: str, n: int = 3) -> set[str]:
        normalized = re.sub(r"\s+", " ", (text or "").strip())
        if len(normalized) < n:
            return {normalized} if normalized else set()
        return {normalized[i : i + n] for i in range(len(normalized) - n + 1)}

    @staticmethod
    def _jaccard_similarity(set1: set[str], set2: set[str]) -> float:
        if not set1 or not set2:
            return 0.0
        union = set1 | set2
        if not union:
            return 0.0
        return len(set1 & set2) / len(union)

    def _novelty_against_reference(self, question: str, reference_sets: List[set[str]]) -> float:
        if not reference_sets:
            return 1.0
        current = self._extract_ngrams((question or "").lower())
        max_similarity = 0.0
        for ref_set in reference_sets:
            max_similarity = max(max_similarity, self._jaccard_similarity(current, ref_set))
        return max(0.0, 1.0 - max_similarity)

    @staticmethod
    def _topic_match_score(detected_topic: Dict[str, object], target_topic: str) -> float:
        primary = str(detected_topic.get("primary_topic", ""))
        secondary = [str(x) for x in detected_topic.get("secondary_topics", [])]
        confidence = float(detected_topic.get("confidence", 0.0))
        if primary == target_topic:
            return max(0.6, min(1.0, confidence))
        if target_topic in secondary:
            return max(0.4, min(0.8, confidence))
        return min(0.35, confidence)
