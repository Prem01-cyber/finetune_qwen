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

        Old formula halved the score (×0.5) if *any* step failed, turning
        9/10 correct steps into 0.45 while 10/10 → 1.0 — a 0.55 cliff for
        a single arithmetic slip.  On multi-step problems that cliff made
        the signal extremely noisy.

        New formula is proportional:

            step_score = steps_ok / steps_total - 0.3 * (steps_failed / steps_total)

        then we add a small bonus for producing a parseable final-answer
        line (``final_answer == "ok"``) so that solutions which are arithmetically
        self-consistent *and* commit to a clear final answer are scored higher
        than ones that dodge the last line.

            sympy = clamp( step_score + 0.1 * final_ok,  0, 1 )

        Args:
            sympy_verification: Summary dict from verify_solution_text()

        Returns:
            Score in [0, 1]
        """
        steps_total = int(sympy_verification.get("steps_total", 0))
        steps_verified_ok = int(sympy_verification.get("steps_verified_ok", 0))
        steps_failed = int(sympy_verification.get("steps_failed", 0))
        final_answer = sympy_verification.get("final_answer", "")

        if steps_total == 0:
            return 0.0

        ok_ratio = steps_verified_ok / steps_total
        fail_ratio = steps_failed / steps_total
        step_score = ok_ratio - 0.3 * fail_ratio

        final_bonus = 0.1 if final_answer == "ok" else 0.0
        return max(0.0, min(1.0, step_score + final_bonus))
    
    def _compute_consensus_score(self, consensus: Dict[str, Any]) -> float:
        """
        Compute consensus voting score.

        With 3 verifier samples, ``consensus_strength`` is either 0.5
        (2/3 agree) or 1.0 (3/3 agree).  The old formula added a flat
        ``+0.3`` when the primary matched the majority, which ceiling-
        clipped both 0.5 and 1.0 to 1.0 — the model saw no reason to
        push from "just enough" agreement to "robust" agreement.  We
        now return raw ``consensus_strength`` so the gradient actually
        rewards unanimity.

        Scoring:
          * primary matches majority  →  consensus_strength   (0.5 or 1.0)
          * primary is the outlier    →  0.1  (small floor to avoid NaN advantages)
          * no majority at all        →  0.1

        Args:
            consensus: Consensus dict from TripleVerifier

        Returns:
            Score in [0, 1]
        """
        has_majority = consensus.get("has_majority", False)
        consensus_strength = float(consensus.get("consensus_strength", 0.0))
        primary_matches_majority = consensus.get("primary_matches_majority", False)

        if not has_majority:
            return 0.1

        if primary_matches_majority:
            return max(0.0, min(1.0, consensus_strength))

        return 0.1
    
    def _compute_format_score(self, sympy_verification: Dict[str, Any]) -> float:
        """
        Compute format compliance score.

        The old implementation was ``0.5·has_any_step + 0.5·has_final_answer``,
        which saturated at 1.0 as soon as the SFT-primed model wrote one
        ``Step N:`` line and one ``Final Answer:`` line.  That made format a
        constant 0.2 offset on the combined solution reward — a dead gradient.

        New formula (still in [0, 1]) rewards *quality* of structure:

          * equation_ratio = steps_with_equation / total_step_lines
              — penalises steps that are prose with no parseable ``LHS = RHS``
          * final_ok = 1 if final-answer line parses as a number, else 0
          * length_bonus: 0.0 / 0.5 / 1.0 for 0 / 1 / ≥2 steps.
              — discourages one-line hand-waves; rewards multi-step reasoning

          format = 0.5·equation_ratio + 0.3·final_ok + 0.2·length_bonus

        Args:
            sympy_verification: Summary dict from verify_solution_text()

        Returns:
            Score in [0, 1]
        """
        steps_total = int(sympy_verification.get("steps_total", 0))
        steps_skipped = int(sympy_verification.get("steps_skipped_no_equality", 0))
        final_answer = sympy_verification.get("final_answer", "")

        total_step_lines = steps_total + steps_skipped
        if total_step_lines == 0:
            equation_ratio = 0.0
        else:
            equation_ratio = steps_total / total_step_lines

        final_ok = 1.0 if final_answer == "ok" else 0.0

        if steps_total >= 2:
            length_bonus = 1.0
        elif steps_total == 1:
            length_bonus = 0.5
        else:
            length_bonus = 0.0

        score = 0.5 * equation_ratio + 0.3 * final_ok + 0.2 * length_bonus
        return max(0.0, min(1.0, score))
    
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
