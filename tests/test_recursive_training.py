"""
Integration tests for curriculum environment recursive replay behavior.
"""

import unittest
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import torch

from src.rl.mdp_components import Action, State, Transition

try:
    from src.rl.math_environment_curriculum import CurriculumMathEnvironment
except ModuleNotFoundError:  # pragma: no cover
    CurriculumMathEnvironment = None


@unittest.skipIf(CurriculumMathEnvironment is None, "transformers dependency not installed")
class TestRecursiveTraining(unittest.TestCase):
    @patch("src.rl.math_environment_curriculum.CurriculumMathEnvironment.generate_with_logging")
    @patch("src.rl.triple_verifier.TripleVerifier.generate_three_solutions")
    def test_collect_rollouts_mixes_sources(self, mock_three, mock_generate):
        mock_model = Mock()
        mock_model.parameters.return_value = [torch.tensor([1.0])]
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_value = Mock()

        fake_state = State(
            text="x",
            input_ids=torch.tensor([1, 2]),
            attention_mask=torch.tensor([1, 1]),
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

        # Enough generations for two collect calls.
        mock_generate.side_effect = [
            ("A store has 10 apples and sells 2. How many left?", [fake_transition]),
            ("Step 1: 10 - 2 = 8\nFinal Answer: 8", [fake_transition]),
            ("A shop has 12 apples and sells 3. How many left?", [fake_transition]),
            ("Step 1: 12 - 3 = 9\nFinal Answer: 9", [fake_transition]),
            ("A class has 14 books and gives 4 away. How many left?", [fake_transition]),
            ("Step 1: 14 - 4 = 10\nFinal Answer: 10", [fake_transition]),
            ("A team has 15 balls and loses 5. How many left?", [fake_transition]),
            ("Step 1: 15 - 5 = 10\nFinal Answer: 10", [fake_transition]),
        ]
        mock_three.return_value = [
            "Step 1: 10 - 2 = 8\nFinal Answer: 8",
            "Step 1: 10 - 2 = 8\nFinal Answer: 8",
            "Step 1: 10 - 2 = 8\nFinal Answer: 8",
        ]

        with TemporaryDirectory() as temp_dir:
            env = CurriculumMathEnvironment(
                policy_model=mock_model,
                value_model=mock_value,
                tokenizer=mock_tokenizer,
                reference_questions=[],
                curriculum_checkpoint_dir=temp_dir,
            )

            # Warmup iterations fill buffer and increment curriculum iteration.
            env.collect_rollouts(num_trajectories=2, verbose=False)
            env.collect_rollouts(num_trajectories=2, verbose=False)
            env.collect_rollouts(num_trajectories=2, verbose=False)

            self.assertGreaterEqual(len(env.replay_buffer), 1)
            self.assertGreaterEqual(env.last_rollout_mix["fresh"], 0)
            self.assertGreaterEqual(env.last_rollout_mix["replay"], 0)


if __name__ == "__main__":
    unittest.main()
