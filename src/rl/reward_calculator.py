"""
Reward calculation for RL self-improvement loop.

This module computes rewards for both question generation and solution quality
in the self-play training loop.

Reward Components:
------------------
1. Question Quality (0-1):
   - Solvability: Can the model solve the generated question?
   - Novelty: Is it different from training set?
   - Difficulty: Appropriate challenge level (2-5 steps)

2. Solution Quality (0-1):
   - Correctness: SymPy arithmetic verification
   - Format Compliance: Proper step/answer structure
   - Efficiency: Reasonable step count

Combined Reward: 0.5 * question_score + 0.5 * solution_score
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.sft.solution_format import validate_sympy_solution_format
from src.sft.step_verify_sympy import verify_solution_text


@dataclass
class QuestionQualityMetrics:
    """Metrics for question quality assessment."""
    solvability_score: float
    novelty_score: float
    difficulty_score: float
    overall_score: float
    
    # Details
    has_solution: bool
    solution_format_ok: bool
    solution_arithmetic_ok: bool
    question_length: int
    estimated_steps: int


@dataclass
class SolutionQualityMetrics:
    """Metrics for solution quality assessment."""
    correctness_score: float
    format_score: float
    efficiency_score: float
    overall_score: float
    
    # Details
    steps_total: int
    steps_verified_ok: int
    format_valid: bool
    final_answer_ok: bool


@dataclass
class CombinedReward:
    """Combined reward for question + solution pair."""
    question_metrics: QuestionQualityMetrics
    solution_metrics: SolutionQualityMetrics
    combined_score: float
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "combined_score": self.combined_score,
            "question": {
                "overall_score": self.question_metrics.overall_score,
                "solvability": self.question_metrics.solvability_score,
                "novelty": self.question_metrics.novelty_score,
                "difficulty": self.question_metrics.difficulty_score,
            },
            "solution": {
                "overall_score": self.solution_metrics.overall_score,
                "correctness": self.solution_metrics.correctness_score,
                "format": self.solution_metrics.format_score,
                "efficiency": self.solution_metrics.efficiency_score,
            },
        }


class RewardCalculator:
    """
    Calculate rewards for question-solution pairs in self-play loop.
    
    Usage:
        calculator = RewardCalculator(
            reference_questions=gsm8k_questions,  # For novelty scoring
        )
        
        reward = calculator.calculate_reward(
            generated_question=q,
            generated_solution=s,
        )
    """
    
    def __init__(
        self,
        reference_questions: list[str] | None = None,
        novelty_threshold: float = 0.3,
        target_min_steps: int = 2,
        target_max_steps: int = 5,
        max_efficient_steps: int = 8,
    ):
        """
        Initialize reward calculator.
        
        Args:
            reference_questions: List of reference questions for novelty scoring
            novelty_threshold: Similarity threshold for novelty (Jaccard)
            target_min_steps: Minimum target step count
            target_max_steps: Maximum target step count
            max_efficient_steps: Maximum steps before efficiency penalty
        """
        self.reference_questions = reference_questions or []
        self.novelty_threshold = novelty_threshold
        self.target_min_steps = target_min_steps
        self.target_max_steps = target_max_steps
        self.max_efficient_steps = max_efficient_steps
        
        # Prepare reference questions for novelty scoring
        self._reference_ngrams = [
            self._extract_ngrams(q.lower(), n=3) for q in self.reference_questions
        ]
    
    @staticmethod
    def _extract_ngrams(text: str, n: int = 3) -> set[str]:
        """Extract character n-grams from text."""
        text = re.sub(r'\s+', ' ', text.strip())
        return {text[i:i+n] for i in range(len(text) - n + 1)}
    
    @staticmethod
    def _jaccard_similarity(set1: set, set2: set) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    def _calculate_novelty_score(self, question: str) -> float:
        """
        Calculate novelty score for a question.
        
        Returns 1.0 for highly novel, 0.0 for near-duplicate.
        """
        if not self.reference_questions:
            # No reference, assume novel
            return 1.0
        
        question_ngrams = self._extract_ngrams(question.lower(), n=3)
        
        # Find maximum similarity to any reference question
        max_similarity = 0.0
        for ref_ngrams in self._reference_ngrams:
            sim = self._jaccard_similarity(question_ngrams, ref_ngrams)
            max_similarity = max(max_similarity, sim)
        
        # Convert similarity to novelty score
        if max_similarity < self.novelty_threshold:
            # Sufficiently different
            return 1.0
        else:
            # Linearly decay from threshold to 1.0 similarity
            novelty = 1.0 - ((max_similarity - self.novelty_threshold) / (1.0 - self.novelty_threshold))
            return max(0.0, novelty)
    
    def _calculate_difficulty_score(self, step_count: int) -> float:
        """
        Calculate difficulty score based on step count.
        
        Target: 2-5 steps = 1.0
        Below or above: linear decay
        """
        if self.target_min_steps <= step_count <= self.target_max_steps:
            return 1.0
        elif step_count < self.target_min_steps:
            # Too easy
            if step_count == 0:
                return 0.0
            # Linear decay: 1 step = 0.5, 0 steps = 0.0
            return max(0.0, step_count / self.target_min_steps)
        else:
            # Too hard (decay slower to not overly penalize complex problems)
            excess = step_count - self.target_max_steps
            penalty = excess * 0.1  # 10% penalty per extra step
            return max(0.0, 1.0 - penalty)
    
    def assess_question_quality(
        self,
        question: str,
        solution: str | None = None,
    ) -> QuestionQualityMetrics:
        """
        Assess the quality of a generated question.
        
        Args:
            question: Generated question text
            solution: Optional solution generated by model for the question
        
        Returns:
            QuestionQualityMetrics with scores and details
        """
        # Solvability assessment
        solvability_score = 0.0
        has_solution = solution is not None and len(solution.strip()) > 0
        solution_format_ok = False
        solution_arithmetic_ok = False
        steps_verified = 0
        steps_total = 0
        
        if has_solution:
            # Check if solution has proper format
            format_result = validate_sympy_solution_format(solution)
            solution_format_ok = format_result.valid
            
            if solution_format_ok:
                # Check arithmetic correctness
                verify_result = verify_solution_text(solution)
                steps_total = verify_result.summary["steps_total"]
                steps_verified = verify_result.summary["steps_verified_ok"]
                solution_arithmetic_ok = (
                    steps_verified > 0 and
                    verify_result.summary["steps_failed"] == 0 and
                    verify_result.summary["final_answer"] == "ok"
                )
                
                # Solvability scoring
                if solution_arithmetic_ok:
                    solvability_score = 1.0
                elif solution_format_ok:
                    solvability_score = 0.5
                else:
                    solvability_score = 0.2  # At least produced something
            else:
                # Format invalid but has output
                solvability_score = 0.1
        else:
            # No solution provided
            solvability_score = 0.0
        
        # Novelty assessment
        novelty_score = self._calculate_novelty_score(question)
        
        # Difficulty assessment (based on estimated steps from solution)
        estimated_steps = steps_total if steps_total > 0 else 0
        difficulty_score = self._calculate_difficulty_score(estimated_steps)
        
        # Overall question quality
        # Weights: solvability=0.4, novelty=0.3, difficulty=0.3
        overall_score = (
            0.4 * solvability_score +
            0.3 * novelty_score +
            0.3 * difficulty_score
        )
        
        return QuestionQualityMetrics(
            solvability_score=solvability_score,
            novelty_score=novelty_score,
            difficulty_score=difficulty_score,
            overall_score=overall_score,
            has_solution=has_solution,
            solution_format_ok=solution_format_ok,
            solution_arithmetic_ok=solution_arithmetic_ok,
            question_length=len(question),
            estimated_steps=estimated_steps,
        )
    
    def assess_solution_quality(self, solution: str) -> SolutionQualityMetrics:
        """
        Assess the quality of a generated solution.
        
        Args:
            solution: Solution text to assess
        
        Returns:
            SolutionQualityMetrics with scores and details
        """
        # Format compliance
        format_result = validate_sympy_solution_format(solution)
        format_score = 1.0 if format_result.valid else 0.0
        
        # Correctness (arithmetic verification)
        verify_result = verify_solution_text(solution)
        steps_total = verify_result.summary["steps_total"]
        steps_verified_ok = verify_result.summary["steps_verified_ok"]
        steps_failed = verify_result.summary["steps_failed"]
        final_answer_status = verify_result.summary["final_answer"]
        
        final_answer_ok = final_answer_status == "ok"
        
        # Correctness score based on verified steps
        if steps_total > 0:
            correctness_score = steps_verified_ok / steps_total
            # Penalize if any steps failed
            if steps_failed > 0:
                correctness_score *= 0.5
            # Penalize if final answer not OK
            if not final_answer_ok:
                correctness_score *= 0.7
        else:
            correctness_score = 0.0
        
        # Efficiency score (penalize excessive steps)
        if steps_total <= self.max_efficient_steps:
            efficiency_score = 1.0
        else:
            excess = steps_total - self.max_efficient_steps
            penalty = excess * 0.1  # 10% penalty per extra step
            efficiency_score = max(0.0, 1.0 - penalty)
        
        # Overall solution quality
        # Weights: correctness=0.6, format=0.2, efficiency=0.2
        overall_score = (
            0.6 * correctness_score +
            0.2 * format_score +
            0.2 * efficiency_score
        )
        
        return SolutionQualityMetrics(
            correctness_score=correctness_score,
            format_score=format_score,
            efficiency_score=efficiency_score,
            overall_score=overall_score,
            steps_total=steps_total,
            steps_verified_ok=steps_verified_ok,
            format_valid=format_result.valid,
            final_answer_ok=final_answer_ok,
        )
    
    def calculate_reward(
        self,
        generated_question: str,
        generated_solution: str,
    ) -> CombinedReward:
        """
        Calculate combined reward for a question-solution pair.
        
        Args:
            generated_question: Question generated by model
            generated_solution: Solution generated by model for the question
        
        Returns:
            CombinedReward with detailed metrics and combined score
        """
        question_metrics = self.assess_question_quality(
            question=generated_question,
            solution=generated_solution,
        )
        
        solution_metrics = self.assess_solution_quality(
            solution=generated_solution,
        )
        
        # Combined score: equal weight to question and solution quality
        combined_score = 0.5 * question_metrics.overall_score + 0.5 * solution_metrics.overall_score
        
        return CombinedReward(
            question_metrics=question_metrics,
            solution_metrics=solution_metrics,
            combined_score=combined_score,
        )


def load_reference_questions(gsm8k_path: str) -> list[str]:
    """
    Load reference questions from GSM8K training data for novelty scoring.
    
    Args:
        gsm8k_path: Path to GSM8K JSONL file
    
    Returns:
        List of question strings
    """
    import json
    from pathlib import Path
    
    questions = []
    path = Path(gsm8k_path)
    
    if not path.exists():
        return questions
    
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                # Extract question from messages
                for msg in record.get("messages", []):
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        if "Problem:" in content:
                            q = content.split("Problem:")[-1].strip()
                            questions.append(q)
                        break
    
    return questions
