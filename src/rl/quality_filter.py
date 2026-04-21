"""
Quality gating and novelty checks for replay admission.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Set, Tuple

from src.rl.mdp_components import Trajectory
from src.rl.replay_buffer import StoredTrajectory


class QualityFilter:
    def __init__(self, novelty_threshold: float = 0.7) -> None:
        self.novelty_threshold = novelty_threshold

    def meets_replay_criteria(self, metadata: Dict[str, object]) -> Tuple[bool, str]:
        combined_reward = float(metadata.get("combined_reward", 0.0))
        if combined_reward < 0.7:
            return False, "reward_threshold"

        if not bool(metadata.get("sympy_verified", False)):
            return False, "sympy_failed"

        consensus = bool(metadata.get("consensus_achieved", False))
        matches_majority = bool(metadata.get("primary_matches_majority", False))
        if not (consensus and matches_majority):
            return False, "consensus_failed"

        if float(metadata.get("topic_match_score", 0.0)) < 0.6:
            return False, "topic_mismatch"

        return True, "passed"

    def compute_quality_score(self, metadata: Dict[str, object]) -> float:
        return max(
            0.0,
            min(
                1.0,
                (
                    0.4 * float(metadata.get("combined_reward", 0.0))
                    + 0.3 * (1.0 if bool(metadata.get("sympy_verified", False)) else 0.0)
                    + 0.2 * float(metadata.get("topic_match_score", 0.0))
                    + 0.1 * float(metadata.get("clarity_score", 0.0))
                ),
            ),
        )

    def check_novelty(
        self,
        trajectory: Trajectory,
        existing: Iterable[StoredTrajectory],
    ) -> float:
        if trajectory.metadata is None:
            return 0.0
        question = str(trajectory.metadata.get("generated_question", ""))
        new_ngrams = self._extract_ngrams(question.lower(), n=3)
        if not new_ngrams:
            return 0.0

        max_similarity = 0.0
        for stored in existing:
            stored_q = str(stored.metadata.get("generated_question", ""))
            existing_ngrams = self._extract_ngrams(stored_q.lower(), n=3)
            similarity = self._jaccard(new_ngrams, existing_ngrams)
            if similarity > max_similarity:
                max_similarity = similarity
        return 1.0 - max_similarity

    def is_novel_enough(self, novelty_score: float) -> bool:
        return novelty_score >= self.novelty_threshold

    @staticmethod
    def _extract_ngrams(text: str, n: int = 3) -> Set[str]:
        normalized = re.sub(r"\s+", " ", text.strip())
        if not normalized:
            return set()
        if len(normalized) < n:
            return {normalized}
        return {normalized[i : i + n] for i in range(len(normalized) - n + 1)}

    @staticmethod
    def _jaccard(left: Set[str], right: Set[str]) -> float:
        if not left or not right:
            return 0.0
        union = left | right
        if not union:
            return 0.0
        return len(left & right) / len(union)
