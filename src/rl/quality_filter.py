"""
Quality gating and novelty checks for replay admission.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Set, Tuple

from src.rl.mdp_components import Trajectory
from src.rl.replay_buffer import StoredTrajectory


class QualityFilter:
    def __init__(self, novelty_threshold: float = 0.5) -> None:
        """
        Initialize quality filter with relaxed novelty threshold.
        
        Args:
            novelty_threshold: Minimum novelty score (0.5 = moderate diversity)
        """
        self.novelty_threshold = novelty_threshold

    def meets_replay_criteria(self, metadata: Dict[str, object]) -> Tuple[bool, str]:
        """
        Three-tier quality filter for buffer admission.
        
        Tier 1 (Gold): High reward + both verification signals
        Tier 2 (Silver): Very high reward + at least one strong signal
        Tier 3 (Platinum): Near-perfect trajectories bypass filters
        
        Args:
            metadata: Trajectory metadata dict
            
        Returns:
            (is_eligible, reason_or_tier)
        """
        combined_reward = float(metadata.get("combined_reward", 0.0))
        
        # Tier 3: Platinum standard - near-perfect trajectories always get in
        if combined_reward >= 0.95:
            return True, "platinum_standard"
        
        # Tier 1: Gold standard - high quality with both verification signals
        if combined_reward >= 0.7:
            has_consensus = (
                bool(metadata.get("consensus_achieved", False)) and 
                bool(metadata.get("primary_matches_majority", False))
            )
            sympy_clean = bool(metadata.get("sympy_verified", False))
            
            if has_consensus and sympy_clean:
                if float(metadata.get("topic_match_score", 0.0)) >= 0.6:
                    return True, "gold_standard"
        
        # Tier 2: Silver standard - very high reward with at least one strong signal
        if combined_reward >= 0.75:
            # Accept if EITHER perfect SymPy OR strong consensus
            perfect_sympy = float(metadata.get("sympy_score", 0.0)) >= 0.95
            strong_consensus = (
                bool(metadata.get("consensus_achieved", False)) and
                float(metadata.get("consensus_strength", 0.0)) >= 0.8
            )
            
            if perfect_sympy or strong_consensus:
                if float(metadata.get("topic_match_score", 0.0)) >= 0.6:
                    return True, "silver_standard"
        
        # Failed all tiers
        if combined_reward < 0.7:
            return False, f"reward_too_low_{combined_reward:.2f}"
        elif combined_reward < 0.75:
            return False, "reward_below_silver_tier"
        else:
            return False, "no_strong_verification_signal"

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
