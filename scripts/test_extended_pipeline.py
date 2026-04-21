"""
Smoke test for expert panel + recursive replay integration.
"""

from __future__ import annotations

import json
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import torch

from src.rl.mdp_components import Action, State, Transition


def run_smoke_test() -> dict:
    from src.rl.math_environment_curriculum import CurriculumMathEnvironment

    with patch(
        "src.rl.math_environment_curriculum.CurriculumMathEnvironment.generate_with_logging"
    ) as mock_generate, patch(
        "src.rl.triple_verifier.TripleVerifier.generate_three_solutions"
    ) as mock_three:
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

        generation_side_effect = []
        for idx in range(20):
            generation_side_effect.extend(
                [
                    (f"A store has {20+idx} apples and sells {idx%5+1}. How many left?", [fake_transition]),
                    (f"Step 1: {20+idx} - {idx%5+1} = {19+idx-(idx%5)}\nFinal Answer: {19+idx-(idx%5)}", [fake_transition]),
                ]
            )
        mock_generate.side_effect = generation_side_effect
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
            for _ in range(5):
                env.collect_rollouts(num_trajectories=4, verbose=False)

            replay_stats = env.replay_buffer.get_buffer_stats(
                current_iteration=env.curriculum_manager.current_iteration
            )
            return {
                "iterations_completed": env.curriculum_manager.current_iteration,
                "replay_buffer_size": int(len(env.replay_buffer)),
                "last_replay_ratio": float(env.last_replay_ratio),
                "last_rollout_mix": dict(env.last_rollout_mix),
                "buffer_health": float(replay_stats.get("buffer_health", 0.0)),
            }


def main() -> None:
    result = run_smoke_test()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
