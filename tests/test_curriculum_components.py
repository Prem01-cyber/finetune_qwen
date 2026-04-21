"""
Tests for curriculum-driven dual-task components.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import torch

from src.rl.curriculum_manager import CurriculumManager
from src.rl.question_classifier import QuestionClassifier
from src.rl.question_quality_evaluator import QuestionQualityEvaluator

try:
    from src.rl.math_environment_curriculum import CurriculumMathEnvironment
except ModuleNotFoundError:  # pragma: no cover
    CurriculumMathEnvironment = None


class TestQuestionClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = QuestionClassifier()

    def test_detect_fractions_topic(self):
        result = self.classifier.classify_topic(
            "Sarah has 3/4 of a pizza and eats 1/3 of it. How much remains?"
        )
        self.assertEqual(result["primary_topic"], "fractions")
        self.assertGreaterEqual(result["confidence"], 0.2)

    def test_detect_percentages_topic(self):
        result = self.classifier.classify_topic(
            "A shirt costs $50 with a 20% discount. What is the final price?"
        )
        self.assertEqual(result["primary_topic"], "percentages")

    def test_solution_signal_preferred(self):
        result = self.classifier.classify_topic(
            question="A mixed arithmetic question with no clear keywords.",
            solution="Step 1: Solve equation 2*x + 5 = 11\nStep 2: x = 3\nFinal Answer: 3",
        )
        self.assertEqual(result["primary_topic"], "algebra")
        self.assertIn("solution_ops", result["signals_used"])

    def test_difficulty_in_range(self):
        difficulty = self.classifier.estimate_difficulty(
            question="Compute a result from fractions and percentages.",
            solution=(
                "Step 1: 3/4 * 120 = 90\n"
                "Step 2: 20% of 90 = 18\n"
                "Step 3: 90 - 18 = 72\n"
                "Final Answer: 72"
            ),
            consensus_result={"consensus_strength": 0.4},
        )
        self.assertGreaterEqual(difficulty, 0.0)
        self.assertLessEqual(difficulty, 1.0)

    def test_clarity_penalizes_vague_question(self):
        good = self.classifier.check_clarity("A store has 20 apples and sells 5. How many are left?")
        bad = self.classifier.check_clarity("Calculate something.")
        self.assertGreater(good, bad)


class TestCurriculumManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.manager = CurriculumManager(checkpoint_dir=self.temp_dir.name)
        self.manager.initialize()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_select_topic_returns_valid_tuple(self):
        topic, difficulty = self.manager.select_topic_and_difficulty()
        self.assertIn(topic, self.manager.topics)
        self.assertGreaterEqual(difficulty, 0.1)
        self.assertLessEqual(difficulty, 0.95)

    def test_update_increases_attempts(self):
        topic = "fractions"
        before = self.manager.topics[topic].total_attempts
        self.manager.update_from_trajectory(topic, question_reward=0.8, solution_success=True, combined_reward=0.85)
        after = self.manager.topics[topic].total_attempts
        self.assertEqual(after, before + 1)

    def test_persistent_failure_regresses_difficulty(self):
        topic = "algebra"
        state = self.manager.topics[topic]
        state.difficulty_target = 0.8
        for _ in range(3):
            self.manager.update_from_trajectory(topic, question_reward=0.2, solution_success=False, combined_reward=0.15)
        self.assertLess(state.difficulty_target, 0.8)

    def test_checkpoint_save_and_load(self):
        topic = "fractions"
        self.manager.update_from_trajectory(topic, question_reward=0.9, solution_success=True, combined_reward=0.9)
        self.manager.save_state(iteration=1, rollout=None)

        new_manager = CurriculumManager(checkpoint_dir=self.temp_dir.name)
        loaded = new_manager.load_checkpoint_safe()
        self.assertTrue(loaded)
        self.assertGreaterEqual(new_manager.topics[topic].total_attempts, 1)

    def test_generate_instruction_contains_topic_semantics(self):
        instruction = self.manager.generate_instruction("fractions", target_difficulty=0.5)
        self.assertTrue("fraction" in instruction.lower() or "fractions" in instruction.lower())


class TestQuestionQualityEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = QuestionQualityEvaluator(
            reference_questions=["Tom has 10 apples and gives away 2. How many left?"],
            classifier=QuestionClassifier(),
        )

    def test_evaluate_returns_expected_keys(self):
        result = self.evaluator.evaluate(
            question="A shirt costs $50 with 20% discount. What is final price?",
            solution="Step 1: 20/100 * 50 = 10\nStep 2: 50 - 10 = 40\nFinal Answer: 40",
            consensus_result={"has_majority": True, "consensus_strength": 1.0},
            target_topic="percentages",
            target_difficulty=0.5,
        )
        for key in [
            "overall_score",
            "topic_match",
            "difficulty_score",
            "clarity",
            "solvability_score",
            "novelty_combined",
            "measured_difficulty",
            "detected_topic",
        ]:
            self.assertIn(key, result)

    def test_novelty_drops_for_repetition(self):
        first = self.evaluator.compute_novelty_score("A baker has 12 muffins and sells 3. How many left?")
        second = self.evaluator.compute_novelty_score("A baker has 12 muffins and sells 3. How many left?")
        self.assertGreaterEqual(first["combined"], second["combined"])

    def test_solvability_detects_syntactic_failure(self):
        solv = self.evaluator.assess_solvability(
            question="Compute",
            solution="Final Answer: 1",
            consensus_result={"has_majority": True, "consensus_strength": 1.0},
        )
        self.assertFalse(solv["solvable"])
        self.assertEqual(solv["reason"], "syntactic_failure")


@unittest.skipIf(CurriculumMathEnvironment is None, "transformers dependency not installed")
class TestCurriculumMathEnvironment(unittest.TestCase):
    @patch("src.rl.math_environment_curriculum.CurriculumMathEnvironment.generate_with_logging")
    @patch("src.rl.triple_verifier.TripleVerifier.generate_three_solutions")
    def test_rollout_metadata_schema(self, mock_three, mock_generate):
        mock_model = Mock()
        mock_model.parameters.return_value = [torch.tensor([1.0])]
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_value = Mock()

        from src.rl.mdp_components import Action, State, Transition

        fake_state = State(
            text="test",
            input_ids=torch.tensor([1, 2, 3]),
            attention_mask=torch.tensor([1, 1, 1]),
            phase="question_generation",
        )
        fake_action = Action(token_id=1, log_prob=-0.1, entropy=0.5)
        fake_transition = Transition(
            state=fake_state,
            action=fake_action,
            reward=0.0,
            next_state=fake_state,
            value=0.0,
            done=True,
        )
        mock_generate.side_effect = [
            ("A store has 100 apples and sells 20. How many left?", [fake_transition]),
            ("Step 1: 100 - 20 = 80\nFinal Answer: 80", [fake_transition]),
        ]
        mock_three.return_value = [
            "Step 1: 100 - 20 = 80\nFinal Answer: 80",
            "Step 1: 100 - 20 = 80\nFinal Answer: 80",
            "Step 1: 100 - 20 = 80\nFinal Answer: 80",
        ]

        with TemporaryDirectory() as temp_dir:
            env = CurriculumMathEnvironment(
                policy_model=mock_model,
                value_model=mock_value,
                tokenizer=mock_tokenizer,
                reference_questions=[],
                curriculum_checkpoint_dir=temp_dir,
            )
            trajectory = env.rollout_trajectory()
            meta = trajectory.metadata
            for key in [
                "target_topic",
                "target_difficulty",
                "detected_topic",
                "question_reward",
                "solution_reward",
                "combined_reward",
                "reward_breakdown",
            ]:
                self.assertIn(key, meta)
            self.assertGreaterEqual(meta["combined_reward"], 0.0)
            self.assertLessEqual(meta["combined_reward"], 1.0)


if __name__ == "__main__":
    unittest.main()
