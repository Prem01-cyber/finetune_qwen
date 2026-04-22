"""Proposer <-> Solver self-play arena (Theme #4 framing).

The same LLM plays two roles per episode:

    1. PROPOSER role - given a curriculum-selected instruction, the
       model proposes a concrete math problem (``question``).
    2. SOLVER role  - the model then solves its own proposal.

The environment scores *both* roles:

    * Proposer reward  = question-quality score (clarity, novelty,
      topic-match, measured-difficulty vs target).
    * Solver reward    = SymPy step verification + consensus voting
      (triple-verify) + format compliance.

A Zone-of-Proximal-Development (ZPD) controller then nudges topic
probabilities and target difficulty so that future episodes are
*challenging but solvable*.  This is the mechanism behind "recursive
skill amplification" in the Theme #4 description.

This module purposefully stays *thin*.  The heavy lifting is already
inside ``CurriculumMathEnvironment`` - we just expose it with the
right vocabulary for judges and for plugging into TRL/GRPO.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.rl.math_environment_curriculum import CurriculumMathEnvironment

logger = logging.getLogger(__name__)


@dataclass
class SelfPlayEpisodeResult:
    """Outcome of one proposer+solver episode.

    Kept as a plain dataclass (no tensors, no device state) so it can
    be pickled, logged, serialized to JSON, or shipped over HTTP
    without friction.
    """

    topic: str
    instruction: str
    target_difficulty: float
    measured_difficulty: float
    question: str
    solution: str
    proposer_reward: float
    solver_reward: float
    combined_reward: float
    consensus_achieved: bool
    primary_matches_majority: bool
    steps_verified_ok: int
    steps_failed: int
    final_answer_ok: bool
    curriculum_iteration: int
    wall_time_seconds: float
    raw_reward_result: Dict[str, Any] = field(default_factory=dict)


class ProposerSolverArena:
    """Named entry point for self-play episodes.

    This class deliberately owns **no** new ML state.  It is a
    semantic wrapper: ``play_episode()`` calls the existing rollout
    pipeline and repackages the output as a ``SelfPlayEpisodeResult``
    with explicit proposer/solver attribution.

    Example::

        arena = ProposerSolverArena(curriculum_env)
        for _ in range(num_episodes):
            result = arena.play_episode()
            log.info("Arena episode: reward=%.3f (prop=%.3f, solve=%.3f)",
                     result.combined_reward,
                     result.proposer_reward,
                     result.solver_reward)
    """

    def __init__(self, curriculum_env: CurriculumMathEnvironment) -> None:
        self._env = curriculum_env

    # ------------------------------------------------------------------
    # Per-episode API
    # ------------------------------------------------------------------
    def play_episode(self) -> SelfPlayEpisodeResult:
        """Run one full proposer -> solver -> verify episode.

        We drive the underlying ``rollout_trajectory()`` and then
        repackage its ``TrajectoryMetadata`` payload into our Theme-#4
        shape.  Generation time (typically the dominant cost at 1.5B)
        is surfaced in ``wall_time_seconds`` so a team can see whether
        inference is their bottleneck (per rubric rule #12).
        """
        t0 = time.time()
        trajectory = self._env.rollout_trajectory()
        wall = time.time() - t0

        md = trajectory.metadata
        raw = md.get("reward_breakdown", {}) if isinstance(md, dict) else {}

        result = SelfPlayEpisodeResult(
            topic=str(md.get("target_topic", "unknown")),
            instruction=str(md.get("instruction", "")),
            target_difficulty=float(md.get("target_difficulty", 0.0)),
            measured_difficulty=float(md.get("estimated_difficulty", 0.0)),
            question=str(md.get("generated_question", "")),
            solution=str(md.get("generated_solution", "")),
            proposer_reward=float(md.get("question_reward", 0.0)),
            solver_reward=float(md.get("solution_reward", 0.0)),
            combined_reward=float(md.get("combined_reward", 0.0)),
            consensus_achieved=bool(md.get("consensus_achieved", False)),
            primary_matches_majority=bool(md.get("primary_matches_majority", False)),
            steps_verified_ok=int(md.get("steps_verified_ok", 0)),
            steps_failed=int(md.get("steps_failed", 0)),
            final_answer_ok=bool(md.get("final_answer_ok", False)),
            curriculum_iteration=int(md.get("curriculum_iteration", 0)),
            wall_time_seconds=wall,
            raw_reward_result=raw,
        )

        logger.info(
            "Arena episode [topic=%s, iter=%d]: proposer=%.3f, solver=%.3f, combined=%.3f, "
            "majority=%s, sympy=%d/%d, took=%.1fs",
            result.topic,
            result.curriculum_iteration,
            result.proposer_reward,
            result.solver_reward,
            result.combined_reward,
            result.primary_matches_majority,
            result.steps_verified_ok,
            result.steps_verified_ok + result.steps_failed,
            result.wall_time_seconds,
        )
        return result

    def play(self, n: int) -> List[SelfPlayEpisodeResult]:
        """Convenience for ``[self.play_episode() for _ in range(n)]``."""
        return [self.play_episode() for _ in range(n)]
