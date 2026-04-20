"""
Rollout Buffer and Generalised Advantage Estimation (GAE).

Stores completed trajectories, computes GAE advantages, and exposes
mini-batches for PPO updates.

Math
----
GAE advantage (Schulman et al., 2016):

    δ_t   = r_t + γ · V(s_{t+1}) · (1 - done_t) - V(s_t)
    Â_t   = Σ_{l=0}^{T-t} (γλ)^l · δ_{t+l}

Returns (targets for value-function loss):

    G_t   = Â_t + V(s_t)
"""

from __future__ import annotations

import logging
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch

from src.rl.mdp_components import Trajectory, Transition

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GAE Computer
# ---------------------------------------------------------------------------


class GAEComputer:
    """
    Computes GAE advantages and discounted returns for a trajectory.

    Args:
        gamma      : Discount factor γ (1.0 for episodic tasks).
        gae_lambda : GAE trace-decay λ (0.95 typical).
    """

    def __init__(self, gamma: float = 1.0, gae_lambda: float = 0.95) -> None:
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def compute_advantages_and_returns(
        self,
        trajectory: Trajectory,
    ) -> Tuple[List[float], List[float]]:
        """
        Run backwards GAE pass over a trajectory.

        Returns:
            advantages : List[float] length T
            returns    : List[float] length T  (used as value targets)
        """
        rewards = trajectory.rewards
        values = trajectory.values
        dones = trajectory.dones
        T = len(rewards)

        advantages = [0.0] * T
        returns = [0.0] * T

        gae = 0.0
        next_value = 0.0  # V(s_{T+1}) = 0 for terminal state

        for t in reversed(range(T)):
            mask = 0.0 if dones[t] else 1.0
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
            next_value = values[t]

        return advantages, returns


# ---------------------------------------------------------------------------
# Rollout Buffer
# ---------------------------------------------------------------------------


class RolloutBuffer:
    """
    Stores trajectories collected during rollout and prepares PPO batches.

    Usage::

        buffer = RolloutBuffer(gamma=1.0, gae_lambda=0.95)
        for traj in trajectories:
            buffer.add_trajectory(traj)

        for batch in buffer.get_batches(batch_size=32):
            ...

        stats = buffer.get_stats()
        buffer.clear()

    Args:
        gamma      : Discount factor.
        gae_lambda : GAE λ.
    """

    def __init__(self, gamma: float = 1.0, gae_lambda: float = 0.95) -> None:
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self._gae = GAEComputer(gamma, gae_lambda)

        # Flat storage (populated by add_trajectory)
        self._transitions: List[Transition] = []
        self._advantages: List[float] = []
        self._returns: List[float] = []

        # Episode-level bookkeeping
        self._episode_rewards: List[float] = []
        self._episode_lengths: List[int] = []

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    def add_trajectory(self, trajectory: Trajectory) -> None:
        """
        Compute GAE for *trajectory* and append its transitions to the buffer.
        """
        advantages, returns = self._gae.compute_advantages_and_returns(trajectory)

        self._transitions.extend(trajectory.transitions)
        self._advantages.extend(advantages)
        self._returns.extend(returns)

        self._episode_rewards.append(trajectory.total_reward)
        self._episode_lengths.append(len(trajectory))

    def clear(self) -> None:
        """Empty the buffer (call after each PPO update)."""
        self._transitions.clear()
        self._advantages.clear()
        self._returns.clear()
        self._episode_rewards.clear()
        self._episode_lengths.clear()

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, float]:
        """Return summary statistics for logging."""
        if not self._episode_rewards:
            return {
                "num_trajectories": 0,
                "total_steps": 0,
                "mean_episode_reward": 0.0,
                "std_episode_reward": 0.0,
                "mean_episode_length": 0.0,
                "mean_advantage": 0.0,
            }

        rewards = np.array(self._episode_rewards)
        lengths = np.array(self._episode_lengths)
        advs = np.array(self._advantages)

        return {
            "num_trajectories": len(self._episode_rewards),
            "total_steps": int(lengths.sum()),
            "mean_episode_reward": float(rewards.mean()),
            "std_episode_reward": float(rewards.std()),
            "mean_episode_length": float(lengths.mean()),
            "mean_advantage": float(advs.mean()),
        }

    # ------------------------------------------------------------------
    # Mini-batch iteration
    # ------------------------------------------------------------------

    def get_batches(
        self, batch_size: int, shuffle: bool = True
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Yield mini-batches of (log_probs, values, advantages, returns, entropies).

        Advantages are normalised (zero mean, unit variance) within the buffer.
        """
        n = len(self._transitions)
        if n == 0:
            return

        # Normalise advantages
        adv_arr = np.array(self._advantages, dtype=np.float32)
        adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)

        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]

            log_probs = torch.tensor(
                [self._transitions[i].action.log_prob for i in batch_idx],
                dtype=torch.float32,
            )
            values = torch.tensor(
                [self._transitions[i].value for i in batch_idx],
                dtype=torch.float32,
            )
            advantages = torch.tensor(
                adv_arr[batch_idx], dtype=torch.float32
            )
            returns = torch.tensor(
                [self._returns[i] for i in batch_idx], dtype=torch.float32
            )
            entropies = torch.tensor(
                [self._transitions[i].action.entropy for i in batch_idx],
                dtype=torch.float32,
            )

            yield {
                "log_probs": log_probs,
                "values": values,
                "advantages": advantages,
                "returns": returns,
                "entropies": entropies,
            }

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._transitions)
