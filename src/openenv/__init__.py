"""OpenEnv-compliant wrapper around the curriculum-guided math self-play env.

This package is the public surface the hackathon rubric cares about:

* ``models`` - ``Action`` / ``Observation`` / ``RewardBreakdown`` dataclasses
  that form the wire contract between agents and the environment.
* ``environment`` - ``SelfImprovementMathEnv``: an OpenEnv-shaped
  ``reset()`` / ``step()`` / ``state()`` wrapper that delegates the heavy
  lifting to the existing ``CurriculumMathEnvironment``.
* ``server`` - tiny FastAPI app that exposes the env over HTTP for
  OpenEnv's Hub + Spaces deployment story.
* ``client`` - blocking HTTP client so agents (TRL GRPOTrainer, our own
  PPO trainer, ad-hoc demos) can drive a remote instance.

Nothing in this package touches the PPO training loop - it is strictly a
compliance + deployment layer on top of the existing, working code.
"""

from src.openenv.environment import SelfImprovementMathEnv
from src.openenv.models import (
    Action,
    EpisodePhase,
    Observation,
    RewardBreakdown,
    ResetRequest,
    StepRequest,
    StepResponse,
)

__all__ = [
    "Action",
    "EpisodePhase",
    "Observation",
    "RewardBreakdown",
    "ResetRequest",
    "StepRequest",
    "StepResponse",
    "SelfImprovementMathEnv",
]
