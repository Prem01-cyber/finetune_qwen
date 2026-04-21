"""
Integration tests for ConsensusMathEnvironment

Tests full rollout with consensus verification.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import torch

from src.rl.math_environment_consensus import ConsensusMathEnvironment
from src.rl.consensus_reward_calculator import ConsensusRewardCalculator


class TestConsensusRewardQuality(unittest.TestCase):
    """Test that consensus catches semantic errors."""
    
    def setUp(self):
        """Create mock environment components."""
        # Mock model
        self.mock_model = Mock()
        self.mock_model.parameters.return_value = [torch.tensor([1.0])]
        
        # Mock tokenizer
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.pad_token_id = 0
        self.mock_tokenizer.eos_token_id = 1
        
        # Mock value network
        self.mock_value = Mock()
    
    @patch('src.rl.triple_verifier.TripleVerifier.generate_three_solutions')
    def test_semantic_error_detection(self, mock_generate):
        """
        Test that wrong operations get low consensus scores.
        
        Example: Question "100 cupcakes, sell 40, how many left?"
        - Bad solution: "100 + 40 = 140" (wrong operation)
        - Good alternatives: "100 - 40 = 60" (correct)
        
        Expected: Low consensus score (primary doesn't match majority)
        """
        # Setup: primary has wrong operation (addition instead of subtraction)
        bad_primary = """
Step 1: Add the cupcakes
100 + 40 = 140

Final Answer: 140
"""
        
        # Alternatives have correct operation
        good_alt_1 = """
Step 1: Subtract sold cupcakes
100 - 40 = 60

Final Answer: 60
"""
        
        good_alt_2 = """
Step 1: Remaining cupcakes
100 - 40 = 60

Final Answer: 60
"""
        
        # Mock generate to return correct alternatives
        mock_generate.return_value = [good_alt_1, good_alt_2, good_alt_1]
        
        # Create environment
        env = ConsensusMathEnvironment(
            policy_model=self.mock_model,
            value_model=self.mock_value,
            tokenizer=self.mock_tokenizer,
        )
        
        question = "A baker has 100 cupcakes and sells 40. How many are left?"
        
        # Compute reward
        reward = env.compute_reward(question, bad_primary)
        
        # Verify consensus score is low (primary doesn't match majority)
        consensus_score = reward["solution_metrics"]["consensus_score"]
        self.assertLess(consensus_score, 0.5, 
                       "Consensus score should be low when primary is outlier")
        
        # Overall combined score should also be impacted
        combined = reward["combined_score"]
        self.assertLess(combined, 0.6,
                       "Combined score should be low due to semantic error")
    
    @patch('src.rl.triple_verifier.TripleVerifier.generate_three_solutions')
    def test_correct_solution_high_reward(self, mock_generate):
        """
        Test that correct solutions get high consensus scores.
        
        When primary matches majority and all have correct arithmetic,
        should get high reward.
        """
        # All solutions agree
        correct_solution = """
Step 1: Subtract sold cupcakes
100 - 40 = 60

Final Answer: 60
"""
        
        # Mock generate to return similar solutions
        mock_generate.return_value = [correct_solution, correct_solution, correct_solution]
        
        # Create environment
        env = ConsensusMathEnvironment(
            policy_model=self.mock_model,
            value_model=self.mock_value,
            tokenizer=self.mock_tokenizer,
        )
        
        question = "A baker has 100 cupcakes and sells 40. How many are left?"
        
        # Compute reward
        reward = env.compute_reward(question, correct_solution)
        
        # Verify high consensus score
        consensus_score = reward["solution_metrics"]["consensus_score"]
        self.assertGreater(consensus_score, 0.7,
                          "Consensus score should be high when all agree")
        
        # SymPy score should also be high
        sympy_score = reward["solution_metrics"]["sympy_score"]
        self.assertGreater(sympy_score, 0.8,
                          "SymPy score should be high for correct arithmetic")
        
        # Combined score should be high
        combined = reward["combined_score"]
        self.assertGreater(combined, 0.7,
                          "Combined score should be high for fully correct solution")
    
    @patch('src.rl.triple_verifier.TripleVerifier.generate_three_solutions')
    def test_arithmetic_error_detection(self, mock_generate):
        """
        Test that arithmetic errors are caught by SymPy.
        
        Even if consensus is good, arithmetic errors should lower SymPy score.
        """
        # Primary has arithmetic error
        arithmetic_error = """
Step 1: Subtract sold cupcakes
100 - 40 = 70

Final Answer: 70
"""
        
        # Alternatives also have same error (consensus will be high)
        mock_generate.return_value = [arithmetic_error, arithmetic_error, arithmetic_error]
        
        # Create environment
        env = ConsensusMathEnvironment(
            policy_model=self.mock_model,
            value_model=self.mock_value,
            tokenizer=self.mock_tokenizer,
        )
        
        question = "A baker has 100 cupcakes and sells 40. How many are left?"
        
        # Compute reward
        reward = env.compute_reward(question, arithmetic_error)
        
        # Consensus should be high (all agree)
        consensus_score = reward["solution_metrics"]["consensus_score"]
        self.assertGreater(consensus_score, 0.7,
                          "Consensus should be high when all agree")
        
        # SymPy score should be low (arithmetic is wrong)
        sympy_score = reward["solution_metrics"]["sympy_score"]
        self.assertLess(sympy_score, 0.8,
                       "SymPy score should be lower for arithmetic error")


class TestFullRollout(unittest.TestCase):
    """Test complete rollout with consensus metadata."""
    
    @patch('src.rl.math_environment_consensus.ConsensusMathEnvironment.generate_with_logging')
    @patch('src.rl.triple_verifier.TripleVerifier.generate_three_solutions')
    def test_trajectory_has_consensus_metadata(self, mock_generate, mock_gen_logging):
        """
        Test that trajectory includes consensus verification details.
        """
        # Mock model and components
        mock_model = Mock()
        mock_model.parameters.return_value = [torch.tensor([1.0])]
        
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        
        mock_value = Mock()
        
        # Mock generation
        from src.rl.mdp_components import Transition, State, Action
        
        mock_state = State(
            text="test",
            input_ids=torch.tensor([1, 2, 3]),
            attention_mask=torch.tensor([1, 1, 1]),
            phase="question_generation",
        )
        
        mock_action = Action(
            token_id=1,
            log_prob=-1.0,
            entropy=0.5,
        )
        
        mock_transition = Transition(
            state=mock_state,
            action=mock_action,
            reward=0.0,
            next_state=mock_state,
            value=0.5,
            done=True,
        )
        
        # Mock generate_with_logging to return simple transitions
        mock_gen_logging.side_effect = [
            ("Test question?", [mock_transition]),  # Question generation
            ("Step 1: test\nFinal Answer: 60", [mock_transition]),  # Solution
        ]
        
        # Mock alternative solutions
        mock_generate.return_value = [
            "Step 1: test\nFinal Answer: 60",
            "Step 1: test\nFinal Answer: 60",
            "Step 1: test\nFinal Answer: 60",
        ]
        
        # Create environment
        env = ConsensusMathEnvironment(
            policy_model=mock_model,
            value_model=mock_value,
            tokenizer=mock_tokenizer,
        )
        
        # Run rollout
        trajectory = env.rollout_trajectory()
        
        # Verify trajectory has metadata
        self.assertIn("reward_breakdown", trajectory.metadata)
        
        # Verify solution_metrics includes consensus info
        solution_metrics = trajectory.metadata["reward_breakdown"]["solution_metrics"]
        self.assertIn("consensus_score", solution_metrics)
        self.assertIn("sympy_score", solution_metrics)
        self.assertIn("verification_details", solution_metrics)
        
        # Verify verification details has consensus info
        verification = solution_metrics["verification_details"]
        self.assertIn("consensus", verification)
        
        consensus = verification["consensus"]
        self.assertIn("has_majority", consensus)
        self.assertIn("consensus_strength", consensus)
        self.assertIn("primary_matches_majority", consensus)


class TestErrorHandling(unittest.TestCase):
    """Test error handling for edge cases."""
    
    def setUp(self):
        """Create mock environment components."""
        self.mock_model = Mock()
        self.mock_model.parameters.return_value = [torch.tensor([1.0])]
        
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.pad_token_id = 0
        
        self.mock_value = Mock()
    
    @patch('src.rl.triple_verifier.TripleVerifier.generate_three_solutions')
    def test_no_consensus_low_reward(self, mock_generate):
        """
        Test that no consensus (all 3 different) gives low reward.
        """
        # All solutions different
        mock_generate.return_value = [
            "Final Answer: 60",
            "Final Answer: 140",
            "Final Answer: 200",
        ]
        
        env = ConsensusMathEnvironment(
            policy_model=self.mock_model,
            value_model=self.mock_value,
            tokenizer=self.mock_tokenizer,
        )
        
        question = "Ambiguous question?"
        solution = "Final Answer: 60"
        
        reward = env.compute_reward(question, solution)
        
        # No consensus should give very low score
        consensus_score = reward["solution_metrics"]["consensus_score"]
        self.assertLess(consensus_score, 0.2,
                       "No consensus should give very low score")
    
    @patch('src.rl.triple_verifier.TripleVerifier.generate_three_solutions')
    def test_generation_failure_safe_default(self, mock_generate):
        """
        Test that generation failures return safe default reward.
        """
        # Mock generation to return empty/invalid solutions
        mock_generate.return_value = ["", "", ""]
        
        env = ConsensusMathEnvironment(
            policy_model=self.mock_model,
            value_model=self.mock_value,
            tokenizer=self.mock_tokenizer,
        )
        
        question = "Test question?"
        solution = "No final answer"
        
        # Should not crash
        reward = env.compute_reward(question, solution)
        
        # Should return low but valid reward
        self.assertIsInstance(reward["combined_score"], float)
        self.assertGreaterEqual(reward["combined_score"], 0.0)
        self.assertLessEqual(reward["combined_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
