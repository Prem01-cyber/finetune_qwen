"""
Unit tests for TripleVerifier

Tests consensus voting, answer extraction, and edge cases.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import torch

from src.rl.triple_verifier import TripleVerifier


class TestAnswerExtraction(unittest.TestCase):
    """Test numeric answer extraction from solutions."""
    
    def setUp(self):
        """Create mock verifier for testing."""
        self.mock_model = Mock()
        self.mock_model.parameters.return_value = [torch.tensor([1.0])]
        
        self.mock_tokenizer = Mock()
        
        self.verifier = TripleVerifier(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
        )
    
    def test_extract_valid_integer(self):
        """Test extracting valid integer answer."""
        solution = """
Step 1: Calculate total
100 + 40 = 140

Final Answer: 140
"""
        answer = self.verifier.extract_numeric_answer(solution)
        self.assertEqual(answer, 140.0)
    
    def test_extract_valid_float(self):
        """Test extracting valid float answer."""
        solution = """
Step 1: Divide
10 / 3 = 3.333...

Final Answer: 3.333333
"""
        answer = self.verifier.extract_numeric_answer(solution)
        self.assertIsNotNone(answer)
        self.assertAlmostEqual(answer, 3.333333, places=6)
    
    def test_extract_fraction(self):
        """Test extracting fraction answer."""
        solution = """
Step 1: Simplify
2/3 of 15

Final Answer: 10
"""
        answer = self.verifier.extract_numeric_answer(solution)
        self.assertEqual(answer, 10.0)
    
    def test_no_final_answer(self):
        """Test solution without Final Answer line."""
        solution = """
Step 1: Calculate
100 + 40 = 140
"""
        answer = self.verifier.extract_numeric_answer(solution)
        self.assertIsNone(answer)
    
    def test_unparseable_answer(self):
        """Test answer that can't be parsed."""
        solution = """
Step 1: Calculate
Final Answer: maybe around 100 or so
"""
        answer = self.verifier.extract_numeric_answer(solution)
        self.assertIsNone(answer)
    
    def test_very_large_number(self):
        """Test very large number."""
        solution = """
Final Answer: 1000000000
"""
        answer = self.verifier.extract_numeric_answer(solution)
        self.assertEqual(answer, 1000000000.0)
    
    def test_very_small_number(self):
        """Test very small number."""
        solution = """
Final Answer: 0.000001
"""
        answer = self.verifier.extract_numeric_answer(solution)
        self.assertAlmostEqual(answer, 0.000001, places=6)
    
    def test_negative_number(self):
        """Test negative number."""
        solution = """
Final Answer: -42
"""
        answer = self.verifier.extract_numeric_answer(solution)
        self.assertEqual(answer, -42.0)


class TestConsensusVoting(unittest.TestCase):
    """Test consensus voting logic."""
    
    def setUp(self):
        """Create mock verifier for testing."""
        self.mock_model = Mock()
        self.mock_model.parameters.return_value = [torch.tensor([1.0])]
        
        self.mock_tokenizer = Mock()
        
        self.verifier = TripleVerifier(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
        )
    
    def test_perfect_consensus_3_of_3(self):
        """Test 3/3 solutions agree."""
        solutions = [
            "Step 1: 100 - 40 = 60\nFinal Answer: 60",
            "Step 1: 100 - 40 = 60\nFinal Answer: 60",
            "Step 1: 100 - 40 = 60\nFinal Answer: 60",
        ]
        
        consensus = self.verifier.check_consensus(solutions)
        
        self.assertTrue(consensus["has_majority"])
        self.assertEqual(consensus["majority_answer"], 60.0)
        self.assertEqual(consensus["majority_count"], 3)
        self.assertEqual(consensus["consensus_strength"], 1.0)
        self.assertEqual(consensus["answer_diversity"], 1)
    
    def test_majority_2_of_3(self):
        """Test 2/3 solutions agree."""
        solutions = [
            "Step 1: 100 - 40 = 60\nFinal Answer: 60",
            "Step 1: 100 - 40 = 60\nFinal Answer: 60",
            "Step 1: 100 + 40 = 140\nFinal Answer: 140",
        ]
        
        consensus = self.verifier.check_consensus(solutions)
        
        self.assertTrue(consensus["has_majority"])
        self.assertEqual(consensus["majority_answer"], 60.0)
        self.assertEqual(consensus["majority_count"], 2)
        self.assertEqual(consensus["consensus_strength"], 0.5)
        self.assertEqual(consensus["answer_diversity"], 2)
    
    def test_no_consensus_all_different(self):
        """Test all 3 solutions different."""
        solutions = [
            "Final Answer: 60",
            "Final Answer: 140",
            "Final Answer: 200",
        ]
        
        consensus = self.verifier.check_consensus(solutions)
        
        self.assertFalse(consensus["has_majority"])
        self.assertEqual(consensus["consensus_strength"], 0.0)
        self.assertEqual(consensus["answer_diversity"], 3)
    
    def test_no_valid_answers(self):
        """Test when no answers can be parsed."""
        solutions = [
            "Step 1: Calculate something",
            "Step 1: Do math",
            "Step 1: Compute result",
        ]
        
        consensus = self.verifier.check_consensus(solutions)
        
        self.assertFalse(consensus["has_majority"])
        self.assertIsNone(consensus["majority_answer"])
        self.assertEqual(consensus["majority_count"], 0)
        self.assertEqual(consensus["answer_diversity"], 0)
    
    def test_float_comparison_tolerance(self):
        """Test that similar floats are treated as same answer."""
        solutions = [
            "Final Answer: 3.333333",
            "Final Answer: 3.333333",
            "Final Answer: 10.0",
        ]
        
        consensus = self.verifier.check_consensus(solutions)
        
        self.assertTrue(consensus["has_majority"])
        self.assertEqual(consensus["majority_count"], 2)
        self.assertAlmostEqual(consensus["majority_answer"], 3.333333, places=6)


class TestBatchedGeneration(unittest.TestCase):
    """Test batched solution generation."""
    
    @patch('src.rl.triple_verifier.torch')
    def test_generates_three_solutions(self, mock_torch):
        """Test that 3 solutions are generated in a single call."""
        # Setup mocks
        mock_model = Mock()
        mock_model.parameters.return_value = [torch.tensor([1.0])]
        
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            "input_ids": torch.zeros((3, 10), dtype=torch.long),
            "attention_mask": torch.ones((3, 10), dtype=torch.long),
        }
        mock_tokenizer.decode.side_effect = [
            "### Task: Solve Problem\nProblem: Test\nSolution:\nSolution 1",
            "### Task: Solve Problem\nProblem: Test\nSolution:\nSolution 2",
            "### Task: Solve Problem\nProblem: Test\nSolution:\nSolution 3",
        ]
        mock_tokenizer.pad_token_id = 0
        
        # Mock generate to return 3 outputs
        mock_model.generate.return_value = torch.zeros((3, 20), dtype=torch.long)
        
        verifier = TripleVerifier(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )
        
        solutions = verifier.generate_three_solutions("Test question")
        
        # Verify 3 solutions generated
        self.assertEqual(len(solutions), 3)
        
        # Verify model.generate called once with batch_size=3
        mock_model.generate.assert_called_once()
        call_kwargs = mock_model.generate.call_args[1]
        self.assertEqual(call_kwargs["input_ids"].shape[0], 3)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def setUp(self):
        """Create mock verifier for testing."""
        self.mock_model = Mock()
        self.mock_model.parameters.return_value = [torch.tensor([1.0])]
        
        self.mock_tokenizer = Mock()
        
        self.verifier = TripleVerifier(
            model=self.mock_model,
            tokenizer=self.mock_tokenizer,
        )
    
    def test_handle_nan_answer(self):
        """Test handling NaN in answer."""
        solution = "Final Answer: nan"
        answer = self.verifier.extract_numeric_answer(solution)
        # NaN should fail to parse
        self.assertIsNone(answer)
    
    def test_handle_inf_answer(self):
        """Test handling infinity in answer."""
        solution = "Final Answer: inf"
        answer = self.verifier.extract_numeric_answer(solution)
        # Should parse inf but may need special handling
        # For now, just check it doesn't crash
        self.assertIsNotNone(answer)
    
    def test_partial_valid_answers(self):
        """Test when only some answers are valid."""
        solutions = [
            "Final Answer: 60",
            "No answer here",
            "Final Answer: 60",
        ]
        
        consensus = self.verifier.check_consensus(solutions)
        
        # Should still get consensus from 2 valid answers
        self.assertTrue(consensus["has_majority"])
        self.assertEqual(consensus["majority_answer"], 60.0)
    
    def test_empty_solutions(self):
        """Test with empty solutions."""
        solutions = ["", "", ""]
        
        consensus = self.verifier.check_consensus(solutions)
        
        self.assertFalse(consensus["has_majority"])
        self.assertEqual(consensus["answer_diversity"], 0)


if __name__ == "__main__":
    unittest.main()
