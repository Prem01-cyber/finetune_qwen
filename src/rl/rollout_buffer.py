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

from src.rl.mdp_components import State, Trajectory, Transition

logger = logging.getLogger(__name__)


def _pad_2d(
    id_list: List[torch.Tensor],
    mask_list: List[torch.Tensor],
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad token ids and masks to a common length (batch, max_len)."""
    max_len = max(t.size(0) for t in id_list)
    batch = len(id_list)
    input_ids = torch.full((batch, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch, max_len), dtype=torch.long)
    for i, (ids, m) in enumerate(zip(id_list, mask_list)):
        L = ids.size(0)
        input_ids[i, :L] = ids
        attention_mask[i, :L] = m
    return input_ids, attention_mask


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

    def __init__(
        self,
        gamma: float = 1.0,
        gae_lambda: float = 0.95,
        pad_token_id: int = 0,
    ) -> None:
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.pad_token_id = pad_token_id
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
        
        Clones tensors to detach from inference mode (fixes torch.compile CUDA graph issues).
        """
        advantages, returns = self._gae.compute_advantages_and_returns(trajectory)

        # Clone transitions to detach from inference mode
        cloned_transitions = []
        for trans in trajectory.transitions:
            cloned_trans = Transition(
                state=State(
                    text=trans.state.text,
                    input_ids=trans.state.input_ids.clone(),
                    attention_mask=trans.state.attention_mask.clone(),
                    phase=trans.state.phase,
                ),
                action=trans.action,  # No tensors in Action
                reward=trans.reward,
                next_state=State(
                    text=trans.next_state.text,
                    input_ids=trans.next_state.input_ids.clone(),
                    attention_mask=trans.next_state.attention_mask.clone(),
                    phase=trans.next_state.phase,
                ),
                value=trans.value,
                done=trans.done,
            )
            cloned_transitions.append(cloned_trans)
        
        self._transitions.extend(cloned_transitions)
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

        # Normalise advantages then clip to prevent a single outlier batch
        # from spiking approx_kl and tripping the early-stop before the rest
        # of the gradient budget is used (observed as 51/52 batches KL≈0.0003,
        # one batch KL=0.085, early-stop at 5% budget).
        adv_arr = np.array(self._advantages, dtype=np.float32)
        adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)
        adv_arr = np.clip(adv_arr, -5.0, 5.0)

        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, n, batch_size):
            batch_idx = indices[start : start + batch_size]

            # Padded state tensors for re-forward through policy / value (variable length)
            id_list = [
                self._transitions[i].state.input_ids.long().cpu()
                for i in batch_idx
            ]
            mask_list = [
                self._transitions[i].state.attention_mask.long().cpu()
                for i in batch_idx
            ]
            input_ids, attention_mask = _pad_2d(
                id_list, mask_list, pad_id=self.pad_token_id
            )

            old_log_probs = torch.tensor(
                [self._transitions[i].action.log_prob for i in batch_idx],
                dtype=torch.float32,
            )
            old_values = torch.tensor(
                [self._transitions[i].value for i in batch_idx],
                dtype=torch.float32,
            )
            action_token_ids = torch.tensor(
                [self._transitions[i].action.token_id for i in batch_idx],
                dtype=torch.long,
            )
            advantages = torch.tensor(adv_arr[batch_idx], dtype=torch.float32)
            returns = torch.tensor(
                [self._returns[i] for i in batch_idx], dtype=torch.float32
            )

            yield {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "action_token_ids": action_token_ids,
                "old_log_probs": old_log_probs,
                "old_values": old_values,
                "advantages": advantages,
                "returns": returns,
            }

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._transitions)
