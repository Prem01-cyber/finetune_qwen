"""
Curriculum-aware math environment with dual reward signals.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.rl.consensus_reward_calculator import ConsensusRewardCalculator
from src.rl.curriculum_manager import CurriculumManager
from src.rl.math_environment_consensus import ConsensusMathEnvironment
from src.rl.mdp_components import Trajectory
from src.rl.question_quality_evaluator import QuestionQualityEvaluator
from src.rl.value_network import ValueHead

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryMetadata:
    curriculum_iteration: int
    target_topic: str
    target_difficulty: float
    instruction: str
    generated_question: str
    generated_solution: str
    question_length: int
    solution_length: int
    detected_topic: str
    detected_secondary_topics: List[str]
    topic_match_score: float
    estimated_difficulty: float
    clarity_score: float
    novelty_scores: Dict[str, float]
    consensus_achieved: bool
    consensus_strength: float
    answer_diversity: int
    majority_answer: Optional[float]
    primary_matches_majority: bool
    sympy_verified: bool
    steps_total: int
    steps_verified_ok: int
    steps_failed: int
    final_answer_ok: bool
    question_reward: float
    solution_reward: float
    combined_reward: float
    reward_breakdown: Dict[str, object]
    topics_in_sweet_spot: List[str]
    current_focus_topics: List[str]
    curriculum_state_snapshot: Dict[str, object]


class CurriculumMathEnvironment(ConsensusMathEnvironment):
    """Consensus environment extended with adaptive curriculum logic."""

    def __init__(
        self,
        policy_model: AutoModelForCausalLM,
        value_model: ValueHead,
        tokenizer: AutoTokenizer,
        reference_questions: Optional[List[str]] = None,
        curriculum_checkpoint_dir: str = "checkpoints/curriculum",
        max_question_tokens: int = 200,
        max_solution_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        consensus_temperature: float = 0.7,
    ):
        super().__init__(
            policy_model=policy_model,
            value_model=value_model,
            tokenizer=tokenizer,
            max_question_tokens=max_question_tokens,
            max_solution_tokens=max_solution_tokens,
            temperature=temperature,
            top_p=top_p,
            consensus_temperature=consensus_temperature,
        )
        self.reference_questions = reference_questions or []
        self.curriculum_manager = CurriculumManager(checkpoint_dir=curriculum_checkpoint_dir)
        self.curriculum_manager.initialize(bootstrap_questions=self.reference_questions)
        self.curriculum_manager.load_checkpoint_safe()
        self.question_evaluator = QuestionQualityEvaluator(reference_questions=self.reference_questions)
        self.consensus_reward_calculator = ConsensusRewardCalculator(verifier=self.triple_verifier)

    def sample_instruction(self) -> Tuple[str, str, float]:
        topic, difficulty = self.curriculum_manager.select_topic_and_difficulty()
        instruction = self.curriculum_manager.generate_instruction(topic=topic, target_difficulty=difficulty)
        return instruction, topic, difficulty

    def compute_reward(
        self,
        question: str,
        solution: str,
        target_topic: str,
        target_difficulty: float,
    ) -> Dict[str, object]:
        solution_result = self.consensus_reward_calculator.calculate_reward(
            question=question,
            solution=solution,
        )
        consensus_info = solution_result["verification_details"]["consensus"]
        question_result = self.question_evaluator.evaluate(
            question=question,
            solution=solution,
            consensus_result=consensus_info,
            target_topic=target_topic,
            target_difficulty=target_difficulty,
        )

        question_reward = float(question_result["overall_score"])
        solution_reward = float(solution_result["combined_score"])
        combined_score = 0.3 * question_reward + 0.7 * solution_reward

        solution_success = bool(consensus_info.get("has_majority", False)) and bool(
            consensus_info.get("primary_matches_majority", False)
        )
        self.curriculum_manager.update_from_trajectory(
            topic=target_topic,
            question_reward=question_reward,
            solution_success=solution_success,
            combined_reward=combined_score,
            measured_difficulty=float(question_result["measured_difficulty"]),
        )

        return {
            "combined_score": combined_score,
            "question_metrics": question_result,
            "solution_metrics": {
                "overall_score": solution_reward,
                "correctness": solution_result["sympy_score"],
                "format_compliance": solution_result["format_score"],
                "efficiency": solution_result["consensus_score"],
                "steps_total": solution_result["breakdown"]["sympy"]["steps_total"],
                "consensus_score": solution_result["consensus_score"],
                "sympy_score": solution_result["sympy_score"],
                "verification_details": solution_result["verification_details"],
            },
            "curriculum_metrics": {
                "target_topic": target_topic,
                "target_difficulty": target_difficulty,
                "detected_topic": question_result["detected_topic"],
                "measured_difficulty": question_result["measured_difficulty"],
            },
        }

    def rollout_trajectory(self) -> Trajectory:
        trajectory = Trajectory()

        instruction, target_topic, target_difficulty = self.sample_instruction()
        question_prompt = self.format_question_generation_prompt(instruction)
        generated_question, question_transitions = self.generate_with_logging(
            initial_prompt=question_prompt,
            max_tokens=self.max_question_tokens,
            phase="question_generation",
        )

        solution_prompt = self.format_solution_prompt(generated_question)
        generated_solution, solution_transitions = self.generate_with_logging(
            initial_prompt=solution_prompt,
            max_tokens=self.max_solution_tokens,
            phase="solution",
        )

        reward_result = self.compute_reward(
            question=generated_question,
            solution=generated_solution,
            target_topic=target_topic,
            target_difficulty=target_difficulty,
        )

        terminal_reward = float(reward_result["combined_score"])
        all_transitions = question_transitions + solution_transitions
        for idx, transition in enumerate(all_transitions):
            transition.reward = terminal_reward if idx == len(all_transitions) - 1 else 0.0
            trajectory.add(transition)

        verification = reward_result["solution_metrics"]["verification_details"]
        consensus = verification["consensus"]
        sympy = verification["sympy_verification"]
        question_metrics = reward_result["question_metrics"]

        metadata = TrajectoryMetadata(
            curriculum_iteration=self.curriculum_manager.current_iteration,
            target_topic=target_topic,
            target_difficulty=target_difficulty,
            instruction=instruction,
            generated_question=generated_question,
            generated_solution=generated_solution,
            question_length=len(question_transitions),
            solution_length=len(solution_transitions),
            detected_topic=str(question_metrics["detected_topic"]["primary_topic"]),
            detected_secondary_topics=[str(x) for x in question_metrics["detected_topic"]["secondary_topics"]],
            topic_match_score=float(question_metrics["topic_match"]),
            estimated_difficulty=float(question_metrics["measured_difficulty"]),
            clarity_score=float(question_metrics["clarity"]),
            novelty_scores=dict(question_metrics["novelty"]),
            consensus_achieved=bool(consensus["has_majority"]),
            consensus_strength=float(consensus["consensus_strength"]),
            answer_diversity=int(consensus["answer_diversity"]),
            majority_answer=consensus.get("majority_answer"),
            primary_matches_majority=bool(consensus["primary_matches_majority"]),
            sympy_verified=int(sympy.get("steps_failed", 0)) == 0,
            steps_total=int(sympy.get("steps_total", 0)),
            steps_verified_ok=int(sympy.get("steps_verified_ok", 0)),
            steps_failed=int(sympy.get("steps_failed", 0)),
            final_answer_ok=str(sympy.get("final_answer", "")) == "ok",
            question_reward=float(question_metrics["overall_score"]),
            solution_reward=float(reward_result["solution_metrics"]["overall_score"]),
            combined_reward=terminal_reward,
            reward_breakdown=reward_result,
            topics_in_sweet_spot=self.curriculum_manager.get_sweet_spot_topics(),
            current_focus_topics=self.curriculum_manager.get_current_focus(),
            curriculum_state_snapshot=self.curriculum_manager.get_curriculum_stats(),
        )
        trajectory.metadata = asdict(metadata)

        logger.info(
            "Curriculum trajectory reward: %.3f (topic=%s target_diff=%.2f measured=%.2f)",
            terminal_reward,
            target_topic,
            target_difficulty,
            question_metrics["measured_difficulty"],
        )

        return trajectory

    def collect_rollouts(self, num_trajectories: int, verbose: bool = True) -> List[Trajectory]:
        trajectories = super().collect_rollouts(num_trajectories=num_trajectories, verbose=verbose)
        self.curriculum_manager.increment_iteration()
        self.curriculum_manager.save_state(iteration=self.curriculum_manager.current_iteration, rollout=None)
        return trajectories
