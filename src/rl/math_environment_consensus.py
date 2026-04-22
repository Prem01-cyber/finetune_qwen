"""
Consensus-Based Math Environment

Extends MathEnvironment to use triple-consensus verification instead of
standard reward calculation.

The only difference is compute_reward() - everything else (rollout logic,
generation, etc.) is inherited from the base class.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.rl.math_environment import MathEnvironment
from src.rl.value_network import ValueHead
from src.rl.triple_verifier import TripleVerifier
from src.rl.consensus_reward_calculator import ConsensusRewardCalculator

logger = logging.getLogger(__name__)


class ConsensusMathEnvironment(MathEnvironment):
    """
    Math environment using triple consensus verification.
    
    Extends MathEnvironment with consensus-based rewards:
    - Generates 3 solutions per question
    - Uses majority voting to catch semantic errors
    - Combines with SymPy verification for arithmetic errors
    
    Note: This is ~3x slower per rollout than base MathEnvironment
    (generates 3 solutions instead of 1), but provides much better
    quality signal for RL training.
    """
    
    def __init__(
        self,
        policy_model: AutoModelForCausalLM,
        value_model: ValueHead,
        tokenizer: AutoTokenizer,
        max_question_tokens: int = 200,
        max_solution_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        consensus_temperature: float = 0.7,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize consensus math environment.
        
        Args:
            policy_model: Language model (actor)
            value_model: Value network (critic)
            tokenizer: Tokenizer
            max_question_tokens: Max tokens for question generation
            max_solution_tokens: Max tokens for solution generation
            temperature: Sampling temperature for primary generation
            top_p: Nucleus sampling parameter
            consensus_temperature: Temperature for alternative solutions
        """
        # Initialize base class without reward_calculator
        # (we'll create our own)
        super().__init__(
            policy_model=policy_model,
            value_model=value_model,
            tokenizer=tokenizer,
            reward_calculator=None,  # We'll override compute_reward
            max_question_tokens=max_question_tokens,
            max_solution_tokens=max_solution_tokens,
            temperature=temperature,
            top_p=top_p,
            device=device,
        )
        
        # Create triple verifier (reuse the resolved device from the base
        # class to avoid inferring it from possibly-sharded parameters).
        self.triple_verifier = TripleVerifier(
            model=self.policy,
            tokenizer=self.tokenizer,
            temperature=consensus_temperature,
            top_p=top_p,
            max_tokens=max_solution_tokens,
            device=self.device,
        )
        
        # Create consensus reward calculator
        self.consensus_reward_calculator = ConsensusRewardCalculator(
            verifier=self.triple_verifier,
        )
        
        logger.info("ConsensusMathEnvironment initialized with triple verification")
    
    def compute_reward(self, question: str, solution: str) -> Dict:
        """
        Compute terminal reward using consensus + SymPy verification.
        
        This is the ONLY method that differs from base MathEnvironment.
        Everything else (rollout_trajectory, generate_with_logging, etc.)
        stays the same.
        
        Args:
            question: Generated question text
            solution: Generated solution text (primary)
        
        Returns:
            Dict with:
                - combined_score: Terminal reward ∈ [0, 1]
                - question_metrics: Not used (set to default)
                - solution_metrics: Breakdown of consensus + SymPy scores
        """
        # Calculate consensus-based reward
        reward_result = self.consensus_reward_calculator.calculate_reward(
            question=question,
            solution=solution,
        )
        
        # Convert to format expected by rest of MathEnvironment
        # (matches the structure from base RewardCalculator)
        return {
            "combined_score": reward_result["combined_score"],
            "question_metrics": {
                "overall_score": 0.5,  # Not used in consensus mode
                "solvability": 0.5,
                "novelty": 0.5,
                "difficulty": 0.5,
            },
            "solution_metrics": {
                "overall_score": reward_result["combined_score"],
                "correctness": reward_result["sympy_score"],
                "format_compliance": reward_result["format_score"],
                "efficiency": reward_result["consensus_score"],  # Reuse this field
                "steps_total": reward_result["breakdown"]["sympy"]["steps_total"],
                "consensus_score": reward_result["consensus_score"],
                "sympy_score": reward_result["sympy_score"],
                "verification_details": reward_result["verification_details"],
            },
        }
