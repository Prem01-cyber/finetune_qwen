"""
Generational replay buffer for recursive self-improvement.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from src.rl.mdp_components import Trajectory


@dataclass
class StoredTrajectory:
    trajectory: Trajectory
    metadata: Dict[str, object]
    generation_iteration: int
    reward: float
    quality_score: float
    topic: str


class GenerationalReplayBuffer:
    """Stores high-quality trajectories and samples diverse replays."""

    def __init__(self, max_size: int = 500) -> None:
        self.max_size = max_size
        self.buffer: List[StoredTrajectory] = []
        self.replayed_count = 0
        self.total_sampled = 0
        self.additions_since_prune = 0

    def __len__(self) -> int:
        return len(self.buffer)

    def add_trajectory(
        self,
        trajectory: Trajectory,
        metadata: Dict[str, object],
        iteration: int,
        quality_score: float,
    ) -> bool:
        stored = StoredTrajectory(
            trajectory=trajectory,
            metadata=metadata,
            generation_iteration=iteration,
            reward=float(metadata.get("combined_reward", trajectory.total_reward)),
            quality_score=float(quality_score),
            topic=str(metadata.get("target_topic", "unknown")),
        )
        self.buffer.append(stored)
        self.additions_since_prune += 1

        if len(self.buffer) > self.max_size:
            self._prune_by_topic_capacity(per_topic_keep=50)
        return True

    def sample_replay_batch(self, n: int, diversity_sample: bool = True) -> List[Trajectory]:
        if n <= 0 or not self.buffer:
            return []
        n = min(n, len(self.buffer))
        self.total_sampled += n

        if not diversity_sample:
            chosen = random.sample(self.buffer, n)
            self.replayed_count += len(chosen)
            return [item.trajectory for item in chosen]

        by_topic = self._group_by_topic()
        topic_names = list(by_topic.keys())
        topic_sizes = np.array([len(by_topic[t]) for t in topic_names], dtype=np.float64)
        topic_probs = topic_sizes / topic_sizes.sum()

        selected: List[StoredTrajectory] = []
        used_ids: set[int] = set()
        attempts = 0
        max_attempts = n * 8
        while len(selected) < n and attempts < max_attempts:
            attempts += 1
            topic_idx = int(np.random.choice(len(topic_names), p=topic_probs))
            topic = topic_names[topic_idx]
            candidates = by_topic[topic]

            # Higher quality samples are preferred within a topic.
            quality = np.array([max(1e-6, c.quality_score) for c in candidates], dtype=np.float64)
            q_probs = quality / quality.sum()
            candidate_idx = int(np.random.choice(len(candidates), p=q_probs))
            candidate = candidates[candidate_idx]
            candidate_key = id(candidate)
            if candidate_key in used_ids:
                continue
            used_ids.add(candidate_key)
            selected.append(candidate)

        if len(selected) < n:
            remainder = [x for x in self.buffer if id(x) not in used_ids]
            random.shuffle(remainder)
            selected.extend(remainder[: n - len(selected)])

        self.replayed_count += len(selected)
        return [item.trajectory for item in selected]

    def get_buffer_stats(self, current_iteration: int | None = None) -> Dict[str, float]:
        if not self.buffer:
            return {
                "buffer_size": 0.0,
                "avg_quality": 0.0,
                "quality_variance": 0.0,
                "staleness": 0.0,
                "topic_entropy": 0.0,
                "replay_success_rate": 0.0,
                "buffer_turnover_rate": 0.0,
                "topics_in_buffer": 0.0,
                "buffer_health": 0.0,
            }

        qualities = np.array([x.quality_score for x in self.buffer], dtype=np.float64)
        max_iter = (
            current_iteration
            if current_iteration is not None
            else max(x.generation_iteration for x in self.buffer)
        )
        staleness = np.array([max_iter - x.generation_iteration for x in self.buffer], dtype=np.float64)
        topic_entropy = self._compute_topic_entropy()
        replay_success = self.replayed_count / max(1, self.total_sampled)
        turnover = self.additions_since_prune / max(1, len(self.buffer))

        stats = {
            "buffer_size": float(len(self.buffer)),
            "avg_quality": float(qualities.mean()),
            "quality_variance": float(qualities.var()),
            "staleness": float(staleness.mean()),
            "topic_entropy": float(topic_entropy),
            "replay_success_rate": float(replay_success),
            "buffer_turnover_rate": float(turnover),
            "topics_in_buffer": float(len(self._group_by_topic())),
        }
        stats["buffer_health"] = float(self.compute_buffer_health(stats))
        return stats

    def compute_buffer_health(self, stats: Dict[str, float] | None = None) -> float:
        if not self.buffer:
            return 0.0
        base = stats or self.get_buffer_stats()
        avg_quality = base["avg_quality"]
        # Use max_size as coarse normalization ceiling for topic entropy.
        entropy_norm = max(1.0, math.log(max(2, len(self._group_by_topic()))))
        topic_diversity = min(1.0, base["topic_entropy"] / entropy_norm)
        staleness_penalty = max(0.0, 1.0 - min(1.0, base["staleness"] / 10.0))
        health = 0.5 * avg_quality + 0.3 * topic_diversity + 0.2 * staleness_penalty
        return float(max(0.0, min(1.0, health)))

    def _group_by_topic(self) -> Dict[str, List[StoredTrajectory]]:
        grouped: Dict[str, List[StoredTrajectory]] = {}
        for item in self.buffer:
            grouped.setdefault(item.topic, []).append(item)
        return grouped

    def _prune_by_topic_capacity(self, per_topic_keep: int) -> None:
        grouped = self._group_by_topic()
        pruned: List[StoredTrajectory] = []
        for _, items in grouped.items():
            items_sorted = sorted(items, key=lambda x: x.quality_score, reverse=True)
            pruned.extend(items_sorted[:per_topic_keep])

        if len(pruned) > self.max_size:
            pruned = sorted(pruned, key=lambda x: x.quality_score, reverse=True)[: self.max_size]
        self.buffer = pruned
        self.additions_since_prune = 0

    def _compute_topic_entropy(self) -> float:
        grouped = self._group_by_topic()
        if not grouped:
            return 0.0
        counts = np.array([len(v) for v in grouped.values()], dtype=np.float64)
        probs = counts / counts.sum()
        return float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum())
