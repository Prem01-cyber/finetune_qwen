"""
Triple Consensus Verifier

Generates 3 independent solutions to a question and uses majority voting
to detect semantic errors (wrong operations) while relying on SymPy for
arithmetic verification.

Core idea: If a question is well-defined, multiple independent solution
attempts should converge to the same answer. Disagreement signals ambiguity
or errors.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Optional

import torch
from sympy import sympify
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.sft.solution_format import extract_final_answer_numeric_str
from src.sft.step_verify_sympy import verify_solution_text, VerificationReport
from src.config.prompts import create_solver_messages

logger = logging.getLogger(__name__)

# Round to 6 decimals to avoid floating point precision issues
ANSWER_PRECISION = 6


class TripleVerifier:
    """
    Generate 3 solutions to a question and perform consensus-based verification.
    
    This combines:
    - Self-consistency via majority voting (catches semantic errors)
    - SymPy arithmetic verification (catches calculation errors)
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        temperature: float = 0.5,
        top_p: float = 0.9,
        max_tokens: int = 500,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize triple verifier.
        
        Args:
            model: Language model for generation
            tokenizer: Tokenizer
            temperature: Sampling temperature (0.5 = moderate consensus, was 0.7)
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens per solution
            device: Optional explicit compute device.  Defaults to the
                device of ``model``'s first parameter.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        if device is not None:
            self.device = torch.device(device)
        else:
            try:
                self.device = next(model.parameters()).device
            except StopIteration:
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
    
    def generate_three_solutions(self, question: str) -> List[str]:
        """
        Generate 3 solutions in a single batched forward pass.
        
        This is more efficient than 3 sequential generations:
        - Single GPU call for all 3 solutions
        - temperature > 0 ensures diversity
        
        Args:
            question: Question to solve
        
        Returns:
            List of 3 solution strings
        """
        # Use centralized prompt configuration
        messages = create_solver_messages(question)
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Create batch of 3 identical prompts
        prompts = [prompt] * 3
        
        # Tokenize all prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        
        # Generate all 3 solutions in single call
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        
        # Decode each solution
        solutions = []
        prompt_length = inputs["input_ids"].shape[1]
        
        for i in range(3):
            # Only decode the generated tokens (skip the prompt)
            generated_ids = outputs[i][prompt_length:]
            solution = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            solutions.append(solution)
        
        # Log for debugging
        answers = [self.extract_numeric_answer(sol) for sol in solutions]
        logger.debug(
            f"Triple verify: Generated 3 solutions → answers: {answers}, "
            f"lengths: [{len(s) for s in solutions}]"
        )
        
        return solutions
    
    def extract_numeric_answer(self, solution: str) -> Optional[float]:
        """
        Extract final numeric answer from solution text.
        
        Uses existing extract_final_answer_numeric_str() to find the
        "Final Answer: X" line, then parses with SymPy and converts to float.
        
        Args:
            solution: Solution text
        
        Returns:
            Float answer or None if parsing fails
        """
        # Extract the "Final Answer: X" line
        answer_str = extract_final_answer_numeric_str(solution)
        
        if not answer_str:
            logger.debug("No final answer line found")
            return None
        
        try:
            # Parse with SymPy
            expr = sympify(answer_str)
            
            # Convert to float
            answer_float = float(expr.evalf())
            
            # Round to avoid floating point precision issues
            answer_rounded = round(answer_float, ANSWER_PRECISION)
            
            return answer_rounded
        
        except (ValueError, TypeError, AttributeError) as e:
            logger.debug(f"Failed to parse answer '{answer_str}': {e}")
            return None
    
    def check_consensus(self, solutions: List[str]) -> Dict[str, Any]:
        """
        Perform majority voting on 3 solutions.
        
        Returns consensus metrics including:
        - Which answer is most common
        - How many solutions agree
        - Consensus strength metric
        
        Args:
            solutions: List of 3 solution strings
        
        Returns:
            Dict with consensus information:
            {
                "answers": [60.0, 60.0, 140.0],
                "majority_answer": 60.0,
                "majority_count": 2,
                "vote_distribution": {60.0: 2, 140.0: 1},
                "has_majority": True,
                "consensus_strength": 0.5,
                "answer_diversity": 2
            }
        """
        # Extract numeric answers from all solutions
        answers = [self.extract_numeric_answer(sol) for sol in solutions]
        
        # Filter out None values (failed parses)
        valid_answers = [a for a in answers if a is not None]
        
        if not valid_answers:
            # No valid answers parsed
            return {
                "answers": answers,
                "majority_answer": None,
                "majority_count": 0,
                "vote_distribution": {},
                "has_majority": False,
                "consensus_strength": 0.0,
                "answer_diversity": 0,
            }
        
        # Count votes
        vote_counts = Counter(valid_answers)
        
        # Get majority answer (most common)
        majority_answer, majority_count = vote_counts.most_common(1)[0]
        
        # Check if we have a majority (≥2 out of 3)
        has_majority = majority_count >= 2
        
        # Consensus strength: (majority_count - 1) / (N - 1)
        # 3/3 agree: (3-1)/(3-1) = 1.0
        # 2/3 agree: (2-1)/(3-1) = 0.5
        # 1/3 agree: (1-1)/(3-1) = 0.0
        N = len(valid_answers)
        if N > 1:
            consensus_strength = (majority_count - 1) / (N - 1)
        else:
            consensus_strength = 1.0 if N == 1 else 0.0
        
        # Answer diversity (number of unique answers)
        answer_diversity = len(vote_counts)
        
        return {
            "answers": answers,
            "majority_answer": majority_answer,
            "majority_count": majority_count,
            "vote_distribution": dict(vote_counts),
            "has_majority": has_majority,
            "consensus_strength": consensus_strength,
            "answer_diversity": answer_diversity,
        }
    
    def verify_with_triple_check(
        self,
        question: str,
        primary_solution: str,
    ) -> Dict[str, Any]:
        """
        Complete verification pipeline combining consensus and SymPy.
        
        Steps:
        1. Generate 2 alternative solutions (we already have primary)
        2. Run SymPy verification on primary solution
        3. Extract answers from all 3 solutions
        4. Compute consensus
        5. Check if primary matches majority
        
        Args:
            question: Question text
            primary_solution: Primary solution (already generated)
        
        Returns:
            Comprehensive report with:
            - SymPy verification results
            - Consensus voting results
            - Whether primary matches majority
            - All 3 solutions for debugging
        """
        logger.debug(f"Running triple verification for question: {question[:50]}...")
        
        # Generate 2 alternative solutions
        # We generate 3 and use indices 1 and 2 (index 0 is similar to primary)
        alternative_solutions = self.generate_three_solutions(question)
        
        # Use last 2 as alternatives (more diverse than first which is similar to primary)
        all_solutions = [primary_solution] + alternative_solutions[1:]
        
        # Run SymPy verification on primary solution
        logger.debug(f"Verifying primary solution (len={len(primary_solution)}): {primary_solution[:200]}...")
        sympy_report = verify_solution_text(primary_solution)
        
        # Extract consensus information from all 3 solutions
        consensus_info = self.check_consensus(all_solutions)
        
        # Check if primary matches majority
        primary_answer = self.extract_numeric_answer(primary_solution)
        majority_answer = consensus_info["majority_answer"]
        
        primary_matches_majority = False
        if primary_answer is not None and majority_answer is not None:
            # Compare with tolerance
            primary_matches_majority = abs(primary_answer - majority_answer) < 1e-6
        
        return {
            "primary_solution": primary_solution,
            "alternative_solutions": alternative_solutions[1:],
            "all_solutions": all_solutions,
            "sympy_verification": sympy_report.summary,
            "consensus": {
                **consensus_info,
                "primary_answer": primary_answer,
                "primary_matches_majority": primary_matches_majority,
            },
        }
