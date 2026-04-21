"""
Consensus Reward Calculator

Combines SymPy arithmetic verification with consensus-based semantic verification
to compute rewards for PPO training.

The key insight: SymPy catches arithmetic errors (2+2=5) but not semantic errors
(adding instead of subtracting). Consensus voting catches semantic errors.
Together they provide robust verification.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from src.rl.triple_verifier import TripleVerifier

logger = logging.getLogger(__name__)

# Reward weights: SymPy, Consensus, Format
CONSENSUS_WEIGHTS = [0.4, 0.4, 0.2]


class ConsensusRewardCalculator:
    """
    Calculate rewards combining SymPy verification and consensus voting.
    
    Reward formula:
        R = 0.4 × S_sympy + 0.4 × S_consensus + 0.2 × S_format
    
    where:
        S_sympy = (steps_verified_ok / steps_total) × penalty_multiplier
        penalty_multiplier = 0.5 if any steps_failed > 0, else 1.0
        
        S_consensus = consensus_strength if primary matches majority
                      0.2 if primary is outlier
                      0.1 if no consensus at all
        consensus_strength = (majority_count - 1) / (N - 1)
        
        S_format = 0.5 × has_steps + 0.5 × has_final_answer
    """
    
    def __init__(self, verifier: TripleVerifier):
        """
        Initialize consensus reward calculator.
        
        Args:
            verifier: TripleVerifier instance for generating and checking solutions
        """
        self.verifier = verifier
    
    def _compute_sympy_score(self, sympy_verification: Dict[str, Any]) -> float:
        """
        Compute SymPy arithmetic verification score.
        
        Args:
            sympy_verification: Summary dict from verify_solution_text()
        
        Returns:
            Score in [0, 1]
        """
        steps_total = sympy_verification.get("steps_total", 0)
        steps_verified_ok = sympy_verification.get("steps_verified_ok", 0)
        steps_failed = sympy_verification.get("steps_failed", 0)
        
        if steps_total == 0:
            # No steps to verify
            return 0.0
        
        # Base score: fraction of steps verified
        base_score = steps_verified_ok / steps_total
        
        # Penalty if any steps failed
        if steps_failed > 0:
            base_score *= 0.5
        
        return base_score
    
    def _compute_consensus_score(self, consensus: Dict[str, Any]) -> float:
        """
        Compute consensus voting score.
        
        Args:
            consensus: Consensus dict from TripleVerifier
        
        Returns:
            Score in [0, 1]
        """
        has_majority = consensus.get("has_majority", False)
        consensus_strength = consensus.get("consensus_strength", 0.0)
        primary_matches_majority = consensus.get("primary_matches_majority", False)
        
        if not has_majority:
            # No consensus (all 3 different) - question likely ambiguous
            return 0.1
        
        if primary_matches_majority:
            # Primary solution matches majority - good!
            # Bonus for matching majority
            score = min(1.0, consensus_strength + 0.3)
            return score
        else:
            # Primary is the outlier - bad!
            # Even if there's a majority, primary didn't match it
            return 0.2
    
    def _compute_format_score(self, sympy_verification: Dict[str, Any]) -> float:
        """
        Compute format compliance score.
        
        Args:
            sympy_verification: Summary dict from verify_solution_text()
        
        Returns:
            Score in [0, 1]
        """
        steps_total = sympy_verification.get("steps_total", 0)
        final_answer = sympy_verification.get("final_answer", "")
        
        # 0.5 if has steps
        has_steps = 0.5 if steps_total > 0 else 0.0
        
        # 0.5 if has final answer
        has_final = 0.5 if final_answer == "ok" else 0.0
        
        return has_steps + has_final
    
    def calculate_reward(
        self,
        question: str,
        solution: str,
    ) -> Dict[str, Any]:
        """
        Calculate combined reward for a question-solution pair.
        
        This is the main entry point. It:
        1. Runs triple verification (generates 2 alternatives, checks consensus)
        2. Computes SymPy score
        3. Computes consensus score
        4. Computes format score
        5. Combines with weights [0.4, 0.4, 0.2]
        
        Args:
            question: Generated question text
            solution: Generated solution text (primary)
        
        Returns:
            Dict with:
            {
                "combined_score": float [0, 1],
                "sympy_score": float,
                "consensus_score": float,
                "format_score": float,
                "verification_details": {...},
                "breakdown": {
                    "sympy": {...},
                    "consensus": {...}
                }
            }
        """
        # Run triple verification
        verification_result = self.verifier.verify_with_triple_check(
            question=question,
            primary_solution=solution,
        )
        
        # Extract components
        sympy_verification = verification_result["sympy_verification"]
        consensus = verification_result["consensus"]
        
        # Compute individual scores
        sympy_score = self._compute_sympy_score(sympy_verification)
        consensus_score = self._compute_consensus_score(consensus)
        format_score = self._compute_format_score(sympy_verification)
        
        # Combined score with weights
        w_sympy, w_consensus, w_format = CONSENSUS_WEIGHTS
        combined_score = (
            w_sympy * sympy_score +
            w_consensus * consensus_score +
            w_format * format_score
        )
        
        logger.info(
            f"Consensus reward: combined={combined_score:.3f} = "
            f"0.4×{sympy_score:.3f} + 0.4×{consensus_score:.3f} + 0.2×{format_score:.3f} | "
            f"has_majority={consensus.get('has_majority', False)}, "
            f"primary_matches={consensus.get('primary_matches_majority', False)} | "
            f"SymPy: {sympy_verification.get('steps_verified_ok', 0)}/{sympy_verification.get('steps_total', 0)} ok, "
            f"{sympy_verification.get('steps_failed', 0)} failed, "
            f"final_answer={sympy_verification.get('final_answer', 'missing')}"
        )
        
        return {
            "combined_score": combined_score,
            "sympy_score": sympy_score,
            "consensus_score": consensus_score,
            "format_score": format_score,
            "verification_details": verification_result,
            "breakdown": {
                "sympy": {
                    "steps_total": sympy_verification.get("steps_total", 0),
                    "steps_verified_ok": sympy_verification.get("steps_verified_ok", 0),
                    "steps_failed": sympy_verification.get("steps_failed", 0),
                    "final_answer": sympy_verification.get("final_answer", ""),
                },
                "consensus": {
                    "has_majority": consensus.get("has_majority", False),
                    "consensus_strength": consensus.get("consensus_strength", 0.0),
                    "primary_matches_majority": consensus.get("primary_matches_majority", False),
                    "answer_diversity": consensus.get("answer_diversity", 0),
                    "majority_answer": consensus.get("majority_answer"),
                    "primary_answer": consensus.get("primary_answer"),
                },
            },
        }
