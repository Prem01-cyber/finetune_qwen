"""
Tests for the simulated expert panel reward shaping.
"""

import unittest

from src.rl.expert_panel import MAX_MODIFIER, SimulatedExpertPanel


class TestSimulatedExpertPanel(unittest.TestCase):
    def setUp(self) -> None:
        self.panel = SimulatedExpertPanel()
        self.question_metrics = {
            "clarity": 0.8,
            "solvability_score": 0.9,
            "difficulty_score": 0.6,
            "novelty_combined": 0.7,
        }
        self.solution_metrics = {
            "correctness": 0.9,
            "consensus_score": 0.8,
            "format_compliance": 1.0,
        }

    def test_phase_transitions(self):
        self.assertEqual(self.panel.get_current_expert(0).name, "pedagogy")
        self.assertEqual(self.panel.get_current_expert(3).name, "pedagogy")
        self.assertEqual(self.panel.get_current_expert(4).name, "accuracy")
        self.assertEqual(self.panel.get_current_expert(6).name, "accuracy")
        self.assertEqual(self.panel.get_current_expert(7).name, "challenge")

    def test_reward_modifier_is_bounded(self):
        result = self.panel.apply_expert_preferences(
            base_reward=0.95,
            question_metrics=self.question_metrics,
            solution_metrics=self.solution_metrics,
            iteration=4,
        )
        self.assertGreaterEqual(result["reward_modifier"], -MAX_MODIFIER)
        self.assertLessEqual(result["reward_modifier"], MAX_MODIFIER)
        self.assertGreaterEqual(result["adjusted_reward"], 0.0)
        self.assertLessEqual(result["adjusted_reward"], 1.0)

    def test_feedback_includes_phase_context(self):
        result = self.panel.apply_expert_preferences(
            base_reward=0.6,
            question_metrics=self.question_metrics,
            solution_metrics=self.solution_metrics,
            iteration=7,
        )
        self.assertIn("Challenge expert", result["feedback"])


if __name__ == "__main__":
    unittest.main()
