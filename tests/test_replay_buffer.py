"""
Tests for generational replay buffer and quality filter.
"""

import unittest

import torch

from src.rl.mdp_components import Action, State, Trajectory, Transition
from src.rl.quality_filter import QualityFilter
from src.rl.replay_buffer import GenerationalReplayBuffer


def _build_dummy_trajectory(question: str, reward: float, topic: str) -> Trajectory:
    traj = Trajectory()
    state = State(
        text="prompt",
        input_ids=torch.tensor([1, 2]),
        attention_mask=torch.tensor([1, 1]),
        phase="question_generation",
    )
    action = Action(token_id=1, log_prob=-0.1, entropy=0.5)
    transition = Transition(
        state=state,
        action=action,
        reward=reward,
        next_state=state,
        value=0.0,
        done=True,
    )
    traj.add(transition)
    traj.metadata = {
        "generated_question": question,
        "combined_reward": reward,
        "target_topic": topic,
    }
    return traj


class TestReplayBuffer(unittest.TestCase):
    def setUp(self) -> None:
        self.buffer = GenerationalReplayBuffer(max_size=5)
        self.filter = QualityFilter(novelty_threshold=0.7)

    def test_add_and_sample(self):
        for idx in range(3):
            t = _build_dummy_trajectory(
                question=f"A shop has {10+idx} apples and sells 2. How many left?",
                reward=0.8,
                topic="basic_arithmetic",
            )
            self.buffer.add_trajectory(
                trajectory=t,
                metadata=t.metadata,
                iteration=idx,
                quality_score=0.8,
            )
        sampled = self.buffer.sample_replay_batch(2, diversity_sample=True)
        self.assertEqual(len(sampled), 2)

    def test_stats_and_health(self):
        t = _build_dummy_trajectory(
            question="A class has 20 students and 5 leave. How many remain?",
            reward=0.9,
            topic="single_step_word_problems",
        )
        self.buffer.add_trajectory(t, t.metadata, iteration=0, quality_score=0.9)
        stats = self.buffer.get_buffer_stats(current_iteration=3)
        self.assertGreater(stats["buffer_size"], 0)
        self.assertGreaterEqual(stats["buffer_health"], 0.0)
        self.assertLessEqual(stats["buffer_health"], 1.0)

    def test_quality_filter_and_novelty(self):
        metadata = {
            "combined_reward": 0.85,
            "sympy_verified": True,
            "consensus_achieved": True,
            "primary_matches_majority": True,
            "topic_match_score": 0.8,
            "clarity_score": 0.7,
        }
        ok, reason = self.filter.meets_replay_criteria(metadata)
        self.assertTrue(ok, reason)
        self.assertGreater(self.filter.compute_quality_score(metadata), 0.0)

        t1 = _build_dummy_trajectory(
            question="A baker has 12 muffins and sells 3. How many left?",
            reward=0.9,
            topic="basic_arithmetic",
        )
        self.buffer.add_trajectory(t1, t1.metadata, iteration=1, quality_score=0.9)

        t2 = _build_dummy_trajectory(
            question="A baker has 12 muffins and sells 3. How many left?",
            reward=0.9,
            topic="basic_arithmetic",
        )
        novelty = self.filter.check_novelty(t2, self.buffer.buffer)
        self.assertLess(novelty, 0.7)


if __name__ == "__main__":
    unittest.main()
