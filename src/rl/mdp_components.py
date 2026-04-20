"""
MDP data structures for PPO self-improvement loop.

Defines the core components of the Markov Decision Process:
  - State   : text sequence at time t
  - Action  : token sampled from π_θ(·|s_t)
  - Transition : (s_t, a_t, r_t, s_{t+1}, V(s_t), done)
  - Trajectory : full episode τ = (s_0, a_0, r_0, …, s_T)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List

import torch


@dataclass
class State:
    """
    Represents s_t = context token sequence at generation step t.

    Attributes:
        text          : Decoded string (includes prompt).
        input_ids     : 1-D token-id tensor [seq_len].
        attention_mask: 1-D mask tensor [seq_len].
        phase         : "question_generation" | "solution".
    """

    text: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    phase: str


@dataclass
class Action:
    """
    Represents a_t = single token selected at step t.

    Attributes:
        token_id : Vocabulary index of the chosen token.
        log_prob : log π_θ(a_t | s_t)  (used for importance ratio).
        entropy  : H(π(·|s_t))          (used for entropy bonus).
    """

    token_id: int
    log_prob: float
    entropy: float


@dataclass
class Transition:
    """
    Single step in the MDP: (s_t, a_t, r_t, s_{t+1}, V(s_t), done).

    Attributes:
        state   : s_t
        action  : a_t
        reward  : r_t  (0.0 for non-terminal; sparse reward at episode end)
        next_state: s_{t+1}
        value   : V(s_t) from critic at step t
        done    : Whether this is the terminal transition
    """

    state: State
    action: Action
    reward: float
    next_state: State
    value: float
    done: bool


class Trajectory:
    """
    Complete episode τ = (s_0, a_0, r_0, …, s_T).

    Provides helpers for reward summation and iteration.
    """

    def __init__(self) -> None:
        self.transitions: List[Transition] = []
        self.metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, transition: Transition) -> None:
        """Append a transition to the episode."""
        self.transitions.append(transition)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_reward(self) -> float:
        """Sum of all rewards in the episode R(τ) = Σ r_t."""
        return sum(t.reward for t in self.transitions)

    @property
    def rewards(self) -> List[float]:
        return [t.reward for t in self.transitions]

    @property
    def values(self) -> List[float]:
        return [t.value for t in self.transitions]

    @property
    def log_probs(self) -> List[float]:
        return [t.action.log_prob for t in self.transitions]

    @property
    def entropies(self) -> List[float]:
        return [t.action.entropy for t in self.transitions]

    @property
    def dones(self) -> List[bool]:
        return [t.done for t in self.transitions]

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.transitions)

    def __iter__(self) -> Iterator[Transition]:
        return iter(self.transitions)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Trajectory(len={len(self)}, "
            f"total_reward={self.total_reward:.3f})"
        )
