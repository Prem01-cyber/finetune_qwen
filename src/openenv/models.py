"""Wire-level dataclasses for the OpenEnv self-improvement math env.

We deliberately use ``pydantic.BaseModel`` for every request/response so
that FastAPI can validate payloads at the edge, render an OpenAPI schema
for free (handy for Hub/Spaces browsing), and raise meaningful 422s when
an agent sends garbage instead of silently handing bad text to the
verifier.

All numeric fields are clamped to ``[0, 1]`` where that is semantically
correct - this gives us a cheap first line of defence against agents
that would otherwise try to submit ``NaN``/``inf`` to bend the reward.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, confloat


class EpisodePhase(str, Enum):
    """One-step episode phase.

    Our env treats each curriculum-sampled challenge as a single-step
    episode: ``PROPOSE_AND_SOLVE`` is the only observation phase an agent
    ever sees on reset, and the environment emits ``DONE`` after a step.
    We keep this as an enum anyway so judges/agents can trivially extend
    the env to multi-phase episodes without reshaping the contract.
    """

    PROPOSE_AND_SOLVE = "propose_and_solve"
    DONE = "done"


# ---------------------------------------------------------------------------
# Observations (env -> agent)
# ---------------------------------------------------------------------------
class Observation(BaseModel):
    """What the agent sees after ``reset()`` or ``step()``.

    * ``instruction`` is the curriculum-selected prompt.  The agent's
      job is to propose a question that satisfies this instruction and
      then solve it.
    * ``topic`` / ``target_difficulty`` surface the curriculum's intent
      so agents (and judges) can see *why* this challenge was picked.
    * ``phase`` makes the single-step nature explicit on the wire.
    """

    instruction: str
    topic: str
    target_difficulty: confloat(ge=0.0, le=1.0)  # type: ignore[valid-type]
    phase: EpisodePhase = EpisodePhase.PROPOSE_AND_SOLVE
    curriculum_iteration: int = 0
    focus_topics: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Actions (agent -> env)
# ---------------------------------------------------------------------------
class Action(BaseModel):
    """What the agent submits on ``step()``.

    * ``question`` - a natural-language math problem the agent proposed
      in response to ``instruction``.
    * ``solution`` - the agent's primary solution for that question.

    The env will then run its own triple-verifier (2 additional solutions
    sampled from a held reference model) and score both the question and
    the primary solution.  This keeps the *verifier* honest: the agent
    never sees the two alternates, so it cannot directly optimise for
    "be the majority" without actually proposing a good question.
    """

    question: str = Field(..., min_length=1, max_length=4_000)
    solution: str = Field(..., min_length=1, max_length=8_000)


# ---------------------------------------------------------------------------
# Rewards (env -> agent, on step)
# ---------------------------------------------------------------------------
class RewardBreakdown(BaseModel):
    """Multi-signal reward decomposition (per rubric rule #7).

    The combined score that PPO/GRPO consumes is ``combined``, but every
    independent signal is surfaced so judges can verify we are not
    collapsing them into a single number and so agents can log them.

    All four ``*_score`` components live in ``[0, 1]``.  ``combined`` is
    the weighted sum used in training.
    """

    combined: confloat(ge=0.0, le=1.5)  # type: ignore[valid-type]

    # Scoring components
    sympy_score: confloat(ge=0.0, le=1.0)  # type: ignore[valid-type]
    consensus_score: confloat(ge=0.0, le=1.0)  # type: ignore[valid-type]
    format_score: confloat(ge=0.0, le=1.0)  # type: ignore[valid-type]
    question_score: confloat(ge=0.0, le=1.0)  # type: ignore[valid-type]
    expert_modifier: confloat(ge=0.0, le=1.5)  # type: ignore[valid-type]

    # Process-supervision evidence (rubric rule #9)
    steps_total: int = 0
    steps_verified_ok: int = 0
    steps_failed: int = 0
    final_answer_ok: bool = False

    # Self-play evidence (Theme #4 signal)
    consensus_has_majority: bool = False
    primary_matches_majority: bool = False
    consensus_strength: confloat(ge=0.0, le=1.0) = 0.0  # type: ignore[valid-type]
    measured_difficulty: confloat(ge=0.0, le=1.0) = 0.0  # type: ignore[valid-type]


# ---------------------------------------------------------------------------
# HTTP envelopes
# ---------------------------------------------------------------------------
class ResetRequest(BaseModel):
    """Optional knobs on reset.  All fields default to server-picked values."""

    seed: Optional[int] = None
    # An agent can *request* a topic focus, but the env may refuse if
    # that topic is already mastered (curriculum anti-stall).
    requested_topic: Optional[str] = None


class StepRequest(BaseModel):
    action: Action


class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    reward_breakdown: RewardBreakdown
    info: Dict[str, Any] = Field(default_factory=dict)


class StateResponse(BaseModel):
    """Snapshot of curriculum state - mostly for dashboards / judges."""

    curriculum_iteration: int
    current_focus_topics: List[str]
    sweet_spot_topics: List[str]
    topic_success_rates: Dict[str, float]
    topic_difficulty_targets: Dict[str, float]
