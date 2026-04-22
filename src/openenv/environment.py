"""OpenEnv-shaped wrapper around ``CurriculumMathEnvironment``.

The existing training code talks to ``CurriculumMathEnvironment`` via a
single ``collect_rollouts(n)`` method.  That method is the right tool
for training (it samples from the policy directly and returns
Trajectory objects with per-token log-probs and values), but it is the
*wrong* shape for the hackathon rubric, which wants:

    reset() -> observation
    step(action) -> observation, reward, done, info

So here we expose a single-step, agent-facing API that mirrors the
OpenEnv convention.  Concretely:

* ``reset()`` pulls one ``(instruction, topic, difficulty)`` tuple out
  of the ``CurriculumManager`` (without yet generating anything) and
  returns it as an ``Observation``.
* ``step(action)`` takes the agent's ``(question, primary_solution)``
  pair and runs exactly the same scoring pipeline the trainer uses:
  triple-verifier consensus, SymPy step verification, expert-panel
  modifier.  It then updates the curriculum state with the outcome.

The OpenEnv env therefore acts as a *pure verifier + curriculum* layer.
The agent is responsible for generation.  This split matches rubric
rule #11 ("if the task is verifiable, build the verifier first, then
plug that verifier into RL training"): the env is the verifier, the
trainer is the agent.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, Optional

from src.openenv.models import (
    Action,
    EpisodePhase,
    Observation,
    RewardBreakdown,
    StateResponse,
)
from src.rl.math_environment_curriculum import CurriculumMathEnvironment

logger = logging.getLogger(__name__)


class _NoPendingEpisodeError(RuntimeError):
    """Raised if ``step()`` is called before ``reset()`` or after ``done``."""


class SelfImprovementMathEnv:
    """Agent-facing OpenEnv environment.

    Parameters
    ----------
    curriculum_env:
        A fully constructed ``CurriculumMathEnvironment`` (policy,
        tokenizer, verifier, curriculum manager already wired).  The
        wrapper does not manage the model lifecycle; the caller decides
        whether the env lives in-process with the trainer (fast, single
        GPU) or is hosted as a separate Hugging Face Space (scales,
        slower due to HTTP).
    """

    metadata: Dict[str, Any] = {
        "render_modes": ["human"],
        "episode_model": "single_step",
        "reward_components": [
            "sympy",
            "consensus",
            "format",
            "question",
            "expert_modifier",
        ],
    }

    def __init__(self, curriculum_env: CurriculumMathEnvironment) -> None:
        self._env = curriculum_env
        self._cm = curriculum_env.curriculum_manager
        self._pending: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # OpenEnv contract
    # ------------------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        requested_topic: Optional[str] = None,
    ) -> Observation:
        """Start a new single-step episode.

        We don't generate anything here - generation is the agent's job.
        We simply pull a curriculum-selected challenge so the agent
        knows *what* to propose.  ``requested_topic`` is advisory: if
        the topic is already mastered, we defer to the curriculum.
        """
        if seed is not None:
            random.seed(seed)

        if requested_topic and requested_topic in self._cm.topics:
            status = self._cm.topics[requested_topic].status
            if status != "mastered":
                difficulty = self._cm._get_difficulty_for_topic(requested_topic)
                instruction = self._cm.generate_instruction(
                    topic=requested_topic, target_difficulty=difficulty
                )
                self._cm.current_focus_topics = [requested_topic]
                topic = requested_topic
            else:
                instruction, topic, difficulty = self._env.sample_instruction()
        else:
            instruction, topic, difficulty = self._env.sample_instruction()

        self._pending = {
            "instruction": instruction,
            "topic": topic,
            "target_difficulty": float(difficulty),
        }

        return Observation(
            instruction=instruction,
            topic=topic,
            target_difficulty=float(difficulty),
            phase=EpisodePhase.PROPOSE_AND_SOLVE,
            curriculum_iteration=self._cm.current_iteration,
            focus_topics=list(self._cm.get_current_focus()),
        )

    def step(self, action: Action) -> Dict[str, Any]:
        """Score the agent's (question, solution) and close the episode."""
        if self._pending is None:
            raise _NoPendingEpisodeError(
                "step() called without an active episode; call reset() first"
            )

        pending = self._pending
        self._pending = None  # single-step: the episode ends here

        # Delegate to the existing, battle-tested reward pipeline.  The
        # triple verifier inside compute_reward() will sample two extra
        # solutions from the reference model to run consensus.
        reward_result = self._env.compute_reward(
            question=action.question,
            solution=action.solution,
            target_topic=pending["topic"],
            target_difficulty=pending["target_difficulty"],
        )

        combined = float(reward_result["combined_score"])
        verification = reward_result["solution_metrics"]["verification_details"]
        consensus = verification["consensus"]
        sympy = verification["sympy_verification"]
        question_metrics = reward_result["question_metrics"]
        solution_metrics = reward_result["solution_metrics"]
        expert_metrics = reward_result["expert_metrics"]

        breakdown = RewardBreakdown(
            combined=combined,
            sympy_score=float(solution_metrics.get("sympy_score", 0.0)),
            consensus_score=float(solution_metrics.get("consensus_score", 0.0)),
            format_score=float(solution_metrics.get("format_score", 0.0)),
            question_score=float(question_metrics.get("overall_score", 0.0)),
            expert_modifier=float(expert_metrics.get("reward_modifier", 1.0)),
            steps_total=int(sympy.get("steps_total", 0)),
            steps_verified_ok=int(sympy.get("steps_verified_ok", 0)),
            steps_failed=int(sympy.get("steps_failed", 0)),
            final_answer_ok=str(sympy.get("final_answer", "")) == "ok",
            consensus_has_majority=bool(consensus.get("has_majority", False)),
            primary_matches_majority=bool(
                consensus.get("primary_matches_majority", False)
            ),
            consensus_strength=float(consensus.get("consensus_strength", 0.0)),
            measured_difficulty=float(
                question_metrics.get("measured_difficulty", 0.0)
            ),
        )

        # Curriculum bookkeeping: feed the outcome back so the next
        # reset() samples harder/easier tasks as appropriate.  This is
        # the Theme #4 self-improvement signal.
        solution_success = (
            breakdown.primary_matches_majority and breakdown.final_answer_ok
        )
        self._cm.update_from_trajectory(
            topic=pending["topic"],
            question_reward=breakdown.question_score,
            solution_success=solution_success,
            combined_reward=combined,
            measured_difficulty=breakdown.measured_difficulty,
        )

        # Terminal observation - phase flips to DONE so the agent
        # cannot accidentally try to step() again.
        next_obs = Observation(
            instruction=pending["instruction"],
            topic=pending["topic"],
            target_difficulty=pending["target_difficulty"],
            phase=EpisodePhase.DONE,
            curriculum_iteration=self._cm.current_iteration,
            focus_topics=list(self._cm.get_current_focus()),
        )

        return {
            "observation": next_obs,
            "reward": combined,
            "done": True,
            "reward_breakdown": breakdown,
            "info": {
                "majority_answer": consensus.get("majority_answer"),
                "answer_diversity": int(consensus.get("answer_diversity", 0)),
                "detected_topic": str(
                    question_metrics.get("detected_topic", {}).get(
                        "primary_topic", "unknown"
                    )
                ),
                "expert_phase": str(expert_metrics.get("phase", "unknown")),
                "solution_success": solution_success,
            },
        }

    def state(self) -> StateResponse:
        """Public snapshot of curriculum state (for dashboards / judges)."""
        return StateResponse(
            curriculum_iteration=self._cm.current_iteration,
            current_focus_topics=list(self._cm.get_current_focus()),
            sweet_spot_topics=list(self._cm.get_sweet_spot_topics()),
            topic_success_rates={
                name: float(ts.success_rate) for name, ts in self._cm.topics.items()
            },
            topic_difficulty_targets={
                name: float(ts.difficulty_target)
                for name, ts in self._cm.topics.items()
            },
        )

    def close(self) -> None:
        """Match the OpenEnv contract; we hold no OS-level resources."""
        self._pending = None

    # ------------------------------------------------------------------
    # Convenience: the env is iterable as contexts for quick demos.
    # ------------------------------------------------------------------
    def __enter__(self) -> "SelfImprovementMathEnv":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
