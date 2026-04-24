"""
Simulated expert panel with shifting preferences across curriculum phases.

Reward-shaping design notes
---------------------------
Historically this panel applied a **multiplicative** shaping step:

    adjusted = clip01( base * (1 + modifier) ),  modifier in [-0.3, +0.3]

Two problems that analysis of 20 PPO iterations made obvious:

1.  Saturation.  Any base >= 0.77 was clipped to exactly 1.0 with the
    maximum boost, and a large fraction of self-play rollouts land in
    that zone every iteration.  After the rollout buffer whitens
    advantages, a cluster of identical 1.0s flattens the policy
    gradient — that's the "policy_loss ~ -0.004 across every
    iteration" signature.  Meanwhile the rare non-saturated outlier
    produces a huge standardized advantage -> KL spikes -> early stop.

2.  PRM triple-counting.  The panel used ``correctness`` and
    ``consensus`` weights, and the caller wired both to ``PRM_mean``.
    Combined with the PRM terms inside ``sol`` itself, a single frozen
    PRM's opinion drove ~75% of the variance in ``combined``.  The
    policy can game that by finding text the PRM likes without the
    answer being correct.

The replacement here is:

*   **Additive** shaping with a tight bound (|modifier| <= 0.08 by
    default).  No multiplication, no clip-to-1.  ``base`` stays in
    [0, 1] as computed by the environment, and shaping only nudges it
    a little — GAE + advantage normalization handle scale downstream.
*   The panel no longer consumes the PRM-correlated signals
    (``correctness``, ``consensus_score``).  Those already live inside
    ``sol``.  What the panel *does* add is curriculum-phase taste:
    clarity, solvability, difficulty match, novelty, format
    compliance.
*   A harder, one-sided format penalty: badly-formatted outputs get
    penalized more than well-formatted ones get rewarded.  Solutions
    that don't even parse should not win ties over ones that do.

Nothing about the public API changes — the returned dict still has
``adjusted_reward``, ``reward_modifier``, ``raw_modifier``,
``phase``/``description``, ``signals``, and ``feedback``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


# Tight additive bound. With base in [0, 1] this keeps the final reward
# inside roughly [-0.08, 1.08]; the environment re-clips to [0, 1].
# Lower than the old 0.3 on purpose — shaping is a flavor term, not the
# main signal.
MAX_MODIFIER = 0.08


@dataclass(frozen=True)
class ExpertPhase:
    name: str
    start_iteration: int
    end_iteration: Optional[int]
    clarity_weight: float
    solvability_weight: float
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
    """Applies phase-specific bounded **additive** reward shaping.

    No more multiplication, no more clip-to-1, and crucially no more
    ``correctness``/``consensus`` knobs (which used to double-count
    PRM_mean on top of ``sol``).  The panel now only shapes question
    quality and format — the correctness signal lives in ``sol`` alone.
    """

    def __init__(self) -> None:
        self._phases: List[ExpertPhase] = [
            ExpertPhase(
                name="pedagogy",
                start_iteration=0,
                end_iteration=3,
                clarity_weight=0.30,
                solvability_weight=0.25,
                difficulty_weight=-0.10,
                novelty_weight=0.00,
                format_penalty_weight=0.40,
                description="Prioritize clear, learnable, and solvable foundation tasks.",
            ),
            ExpertPhase(
                name="accuracy",
                start_iteration=4,
                end_iteration=6,
                clarity_weight=0.10,
                solvability_weight=0.20,
                difficulty_weight=0.00,
                novelty_weight=0.00,
                format_penalty_weight=0.70,
                description="Prioritize arithmetic correctness and agreement stability.",
            ),
            ExpertPhase(
                name="challenge",
                start_iteration=7,
                end_iteration=None,
                clarity_weight=0.10,
                solvability_weight=0.10,
                difficulty_weight=0.30,
                novelty_weight=0.20,
                format_penalty_weight=0.30,
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
        format_compliance = float(solution_metrics.get("format_compliance", 0.0))
        format_penalty = 1.0 - format_compliance

        # Centered versions keep the additive shaping close to zero when
        # quality signals are average; only genuinely good (>0.5) or
        # genuinely bad (<0.5) questions move the needle.  Without this,
        # every single rollout got a +0.15 bump just for producing a
        # non-empty string.
        clarity_c = clarity - 0.5
        solvability_c = solvability - 0.5
        difficulty_c = difficulty - 0.5
        novelty_c = novelty - 0.5

        raw_modifier = (
            phase.clarity_weight * clarity_c
            + phase.solvability_weight * solvability_c
            + phase.difficulty_weight * difficulty_c
            + phase.novelty_weight * novelty_c
            - phase.format_penalty_weight * format_penalty
        )
        modifier = max(-MAX_MODIFIER, min(MAX_MODIFIER, raw_modifier))

        # Additive, no multiplication.  We leave the final [0, 1] clip to
        # the caller (math_environment_curriculum) so it can combine the
        # shaping with its own format-floor rule.
        adjusted_reward = float(base_reward) + modifier
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
                f"Accuracy expert {direction} reward; solvability={signals['solvability']:.2f}, "
                f"format={signals['format_compliance']:.2f}."
            )
        return (
            f"Challenge expert {direction} reward; difficulty={signals['difficulty_score']:.2f}, "
            f"novelty={signals['novelty']:.2f}, format={signals['format_compliance']:.2f}."
        )
