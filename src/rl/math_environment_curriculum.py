"""
Curriculum-aware math environment with dual reward signals.
"""

from __future__ import annotations

import logging
import random
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.rl.consensus_reward_calculator import ConsensusRewardCalculator
from src.rl.curriculum_manager import CurriculumManager
from src.rl.expert_panel import SimulatedExpertPanel
from src.rl.math_environment_consensus import ConsensusMathEnvironment
from src.rl.mdp_components import Trajectory
from src.rl.quality_filter import QualityFilter
from src.rl.question_quality_evaluator import QuestionQualityEvaluator
from src.rl.replay_buffer import GenerationalReplayBuffer
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
    pre_expert_reward: float
    expert_reward_modifier: float
    expert_phase: str
    expert_feedback: str
    replay_candidate: bool
    replay_novelty: float
    replay_added: bool
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
        self.expert_panel = SimulatedExpertPanel()
        self.replay_buffer = GenerationalReplayBuffer(max_size=500)
        self.quality_filter = QualityFilter(novelty_threshold=0.5)  # Relaxed from 0.7
        self.last_replay_ratio: float = 0.0
        self.last_rollout_mix: Dict[str, int] = {"fresh": 0, "replay": 0}

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
        base_combined_score = 0.3 * question_reward + 0.7 * solution_reward
        expert_adjustment = self.expert_panel.apply_expert_preferences(
            base_reward=base_combined_score,
            question_metrics=question_result,
            solution_metrics={
                "correctness": solution_result["sympy_score"],
                "consensus_score": solution_result["consensus_score"],
                "format_compliance": solution_result["format_score"],
            },
            iteration=self.curriculum_manager.current_iteration,
        )
        combined_score = float(expert_adjustment["adjusted_reward"])

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
            "base_combined_score": base_combined_score,
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
            "expert_metrics": expert_adjustment,
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
            pre_expert_reward=float(reward_result["base_combined_score"]),
            expert_reward_modifier=float(reward_result["expert_metrics"]["reward_modifier"]),
            expert_phase=str(reward_result["expert_metrics"]["phase"]),
            expert_feedback=str(reward_result["expert_metrics"]["feedback"]),
            replay_candidate=False,
            replay_novelty=0.0,
            replay_added=False,
            combined_reward=terminal_reward,
            reward_breakdown=reward_result,
            topics_in_sweet_spot=self.curriculum_manager.get_sweet_spot_topics(),
            current_focus_topics=self.curriculum_manager.get_current_focus(),
            curriculum_state_snapshot=self.curriculum_manager.get_curriculum_stats(),
        )
        metadata_dict = asdict(metadata)
        # Flatten for quality filter silver tier (not a TrajectoryMetadata field)
        metadata_dict["sympy_score"] = float(
            reward_result["solution_metrics"].get("sympy_score", 0.0)
        )

        # Must attach metadata before novelty check: check_novelty reads trajectory.metadata["generated_question"].
        trajectory.metadata = metadata_dict

        # Admission gate for recursive replay memory.
        is_candidate, reason = self.quality_filter.meets_replay_criteria(metadata_dict)
        metadata_dict["replay_candidate"] = is_candidate
        novelty_score = 0.0
        
        if is_candidate:
            novelty_score = self.quality_filter.check_novelty(trajectory, self.replay_buffer.buffer)
            metadata_dict["replay_novelty"] = float(novelty_score)
            if self.quality_filter.is_novel_enough(novelty_score):
                quality_score = self.quality_filter.compute_quality_score(metadata_dict)
                self.replay_buffer.add_trajectory(
                    trajectory=trajectory,
                    metadata=metadata_dict,
                    iteration=self.curriculum_manager.current_iteration,
                    quality_score=quality_score,
                )
                metadata_dict["replay_added"] = True
                logger.info(
                    f"✓ BUFFER ADMISSION: tier={reason}, "
                    f"reward={terminal_reward:.3f}, novelty={novelty_score:.3f}, quality={quality_score:.3f}"
                )
            else:
                metadata_dict["replay_added"] = False
                logger.info(
                    f"✗ Buffer reject (novelty): tier={reason}, "
                    f"reward={terminal_reward:.3f}, novelty={novelty_score:.3f} < {self.quality_filter.novelty_threshold}"
                )
        else:
            metadata_dict["replay_novelty"] = 0.0
            metadata_dict["replay_added"] = False
            metadata_dict["replay_reject_reason"] = reason
            logger.info(
                f"✗ Buffer reject (quality): reason={reason}, "
                f"reward={terminal_reward:.3f}, "
                f"sympy={metadata_dict.get('sympy_verified')}, "
                f"consensus={metadata_dict.get('consensus_achieved')}, "
                f"primary_match={metadata_dict.get('primary_matches_majority')}, "
                f"topic_match={metadata_dict.get('topic_match_score', 0):.3f}"
            )

        # Refresh pointer in case metadata_dict was mutated above
        trajectory.metadata = metadata_dict

        logger.info(
            "Curriculum trajectory reward: %.3f (topic=%s target_diff=%.2f measured=%.2f)",
            terminal_reward,
            target_topic,
            target_difficulty,
            question_metrics["measured_difficulty"],
        )

        return trajectory

    def _get_adaptive_replay_ratio(self) -> float:
        iteration = self.curriculum_manager.current_iteration
        if iteration < 3:
            return 0.0
        if iteration < 5:
            return 0.15

        buffer_stats = self.replay_buffer.get_buffer_stats(current_iteration=iteration)
        buffer_health = float(buffer_stats.get("buffer_health", 0.0))
        if buffer_health >= 0.75:
            return 0.3
        if buffer_health >= 0.6:
            return 0.25
        return 0.2

    def collect_rollouts(self, num_trajectories: int, verbose: bool = True) -> List[Trajectory]:
        replay_ratio = self._get_adaptive_replay_ratio()
        num_replay = int(num_trajectories * replay_ratio)
        num_replay = min(num_replay, len(self.replay_buffer))
        num_fresh = max(0, num_trajectories - num_replay)

        fresh_trajectories = [
            self.rollout_trajectory()
            for _ in range(num_fresh)
        ]
        replay_trajectories = self.replay_buffer.sample_replay_batch(num_replay, diversity_sample=True)
        for trajectory in replay_trajectories:
            trajectory.metadata["rollout_source"] = "replay"

        for trajectory in fresh_trajectories:
            trajectory.metadata["rollout_source"] = "fresh"

        trajectories = fresh_trajectories + replay_trajectories
        random.shuffle(trajectories)

        self.last_replay_ratio = replay_ratio
        self.last_rollout_mix = {"fresh": len(fresh_trajectories), "replay": len(replay_trajectories)}

        if verbose:
            buffer_stats = self.replay_buffer.get_buffer_stats(
                current_iteration=self.curriculum_manager.current_iteration
            )
            logger.info(
                "Rollout mix: %d fresh + %d replay (ratio=%.2f, buffer_size=%d, health=%.3f)",
                len(fresh_trajectories),
                len(replay_trajectories),
                replay_ratio,
                len(self.replay_buffer),
                float(buffer_stats.get("buffer_health", 0.0)),
            )

        self.curriculum_manager.increment_iteration()
        self.curriculum_manager.save_state(iteration=self.curriculum_manager.current_iteration, rollout=None)
        return trajectories
