"""
Simulated expert panel with shifting preferences across curriculum phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


MAX_MODIFIER = 0.3


@dataclass(frozen=True)
class ExpertPhase:
    name: str
    start_iteration: int
    end_iteration: Optional[int]
    clarity_weight: float
    solvability_weight: float
    correctness_weight: float
    consensus_weight: float
    difficulty_weight: float
    novelty_weight: float
    format_penalty_weight: float
    description: str

    def active_for(self, iteration: int) -> bool:
        if iteration < self.start_iteration:
            return False
        if self.end_iteration is None:
            return True
        return iteration <= self.end_iteration


class SimulatedExpertPanel:
    """Applies phase-specific bounded reward shaping to simulate changing requirements."""

    def __init__(self) -> None:
        self._phases: List[ExpertPhase] = [
            ExpertPhase(
                name="pedagogy",
                start_iteration=0,
                end_iteration=3,
                clarity_weight=0.30,
                solvability_weight=0.20,
                correctness_weight=0.05,
                consensus_weight=0.05,
                difficulty_weight=-0.10,
                novelty_weight=0.00,
                format_penalty_weight=0.10,
                description="Prioritize clear, learnable, and solvable foundation tasks.",
            ),
            ExpertPhase(
                name="accuracy",
                start_iteration=4,
                end_iteration=6,
                clarity_weight=0.05,
                solvability_weight=0.15,
                correctness_weight=0.40,
                consensus_weight=0.30,
                difficulty_weight=0.00,
                novelty_weight=0.00,
                format_penalty_weight=0.20,
                description="Prioritize arithmetic correctness and agreement stability.",
            ),
            ExpertPhase(
                name="challenge",
                start_iteration=7,
                end_iteration=None,
                clarity_weight=0.05,
                solvability_weight=0.10,
                correctness_weight=0.10,
                consensus_weight=0.10,
                difficulty_weight=0.30,
                novelty_weight=0.20,
                format_penalty_weight=0.05,
                description="Prioritize challenging, novel, and diverse problems.",
            ),
        ]

    def get_current_expert(self, iteration: int) -> ExpertPhase:
        for phase in self._phases:
            if phase.active_for(iteration):
                return phase
        return self._phases[-1]

    def apply_expert_preferences(
        self,
        base_reward: float,
        question_metrics: Dict[str, object],
        solution_metrics: Dict[str, object],
        iteration: int,
    ) -> Dict[str, object]:
        phase = self.get_current_expert(iteration)

        clarity = float(question_metrics.get("clarity", 0.0))
        solvability = float(question_metrics.get("solvability_score", 0.0))
        difficulty = float(question_metrics.get("difficulty_score", 0.0))
        novelty = float(question_metrics.get("novelty_combined", 0.0))
        correctness = float(solution_metrics.get("correctness", 0.0))
        consensus = float(solution_metrics.get("consensus_score", 0.0))
        format_compliance = float(solution_metrics.get("format_compliance", 0.0))
        format_penalty = 1.0 - format_compliance

        raw_modifier = (
            phase.clarity_weight * clarity
            + phase.solvability_weight * solvability
            + phase.correctness_weight * correctness
            + phase.consensus_weight * consensus
            + phase.difficulty_weight * difficulty
            + phase.novelty_weight * novelty
            - phase.format_penalty_weight * format_penalty
        )
        modifier = max(-MAX_MODIFIER, min(MAX_MODIFIER, raw_modifier))

        adjusted_reward = max(0.0, min(1.0, float(base_reward) * (1.0 + modifier)))
        return {
            "phase": phase.name,
            "description": phase.description,
            "phase_start_iteration": phase.start_iteration,
            "phase_end_iteration": phase.end_iteration,
            "base_reward": float(base_reward),
            "adjusted_reward": adjusted_reward,
            "reward_modifier": modifier,
            "raw_modifier": raw_modifier,
            "signals": {
                "clarity": clarity,
                "solvability": solvability,
                "difficulty_score": difficulty,
                "novelty": novelty,
                "correctness": correctness,
                "consensus": consensus,
                "format_compliance": format_compliance,
            },
            "feedback": self.get_expert_feedback(
                phase_name=phase.name,
                reward_modifier=modifier,
                signals={
                    "clarity": clarity,
                    "solvability": solvability,
                    "difficulty_score": difficulty,
                    "novelty": novelty,
                    "correctness": correctness,
                    "consensus": consensus,
                    "format_compliance": format_compliance,
                },
            ),
        }

    def get_expert_feedback(
        self,
        phase_name: str,
        reward_modifier: float,
        signals: Dict[str, float],
    ) -> str:
        direction = "boosted" if reward_modifier >= 0 else "penalized"
        if phase_name == "pedagogy":
            return (
                f"Pedagogy expert {direction} reward; clarity={signals['clarity']:.2f}, "
                f"solvability={signals['solvability']:.2f}, difficulty={signals['difficulty_score']:.2f}."
            )
        if phase_name == "accuracy":
            return (
                f"Accuracy expert {direction} reward; correctness={signals['correctness']:.2f}, "
                f"consensus={signals['consensus']:.2f}, format={signals['format_compliance']:.2f}."
            )
        return (
            f"Challenge expert {direction} reward; difficulty={signals['difficulty_score']:.2f}, "
            f"novelty={signals['novelty']:.2f}, correctness={signals['correctness']:.2f}."
        )
