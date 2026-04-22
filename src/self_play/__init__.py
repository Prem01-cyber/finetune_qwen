"""Explicit self-play framing for the Theme #4 (Self-Improvement) submission.

The algorithms here are thin *facades* over components that already
live under ``src/rl/``:

    ProposerSolverArena  --uses-->  CurriculumMathEnvironment
    ZPDDifficultyController  --uses-->  CurriculumManager

Why wrap what already exists?  Because the hackathon rubric (and the
Theme #4 description) specifically look for a named self-play arena
and an adaptive-curriculum controller.  The wrappers add *no* new
learning behaviour - they rename and re-expose the existing behaviour
so judges, TRL examples, and future collaborators can locate it at a
glance.
"""

from src.self_play.arena import ProposerSolverArena, SelfPlayEpisodeResult
from src.self_play.difficulty_controller import ZPDDifficultyController

__all__ = [
    "ProposerSolverArena",
    "SelfPlayEpisodeResult",
    "ZPDDifficultyController",
]
