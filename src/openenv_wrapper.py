"""
OpenEnv Framework Integration

Wraps our MathEnvironment to conform to OpenEnv API.
Required for hackathon submission.

OpenEnv expects:
- reset() → initial observation
- step(action) → (observation, reward, done, info)
- Agent interface for training
"""

import logging
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

try:
    import gym
    from gym import spaces
except ImportError:  # pragma: no cover
    gym = None  # type: ignore
    spaces = None  # type: ignore

from src.rl.math_environment import MathEnvironment

logger = logging.getLogger(__name__)


class OpenEnvMathReasoning:
    """
    OpenEnv-compatible wrapper for mathematical reasoning environment.

    This allows integration with standard RL frameworks and the
    OpenEnv hackathon evaluation system.

    Action space: Discrete(vocab_size) — next token selection
    Observation space: Text (variable length string)
    """

    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(
        self,
        math_env: MathEnvironment,
        vocab_size: int = 50000,
    ) -> None:
        """
        Args:
            math_env  : Our MathEnvironment instance
            vocab_size: Size of token vocabulary
        """
        self.math_env = math_env

        if gym is not None:
            self.action_space = spaces.Discrete(vocab_size)
            self.observation_space = spaces.Dict(
                {
                    "text": spaces.Text(max_length=1000),
                    "phase": spaces.Discrete(2),   # 0=question, 1=solution
                    "step": spaces.Discrete(1000),  # current step in episode
                }
            )

        # Episode state
        self.current_instruction = None
        self.current_question = None
        self.current_phase = "question_generation"
        self.current_step = 0
        self.episode_transitions = []

    def reset(self) -> Dict[str, Any]:
        """
        Reset environment to initial state s_0 ~ μ_0.

        Returns:
            Initial observation
        """
        self.current_instruction = self.math_env.sample_instruction()
        self.current_phase = "question_generation"
        self.current_step = 0
        self.episode_transitions = []
        self.current_question = None

        prompt = self.math_env.format_question_generation_prompt(
            self.current_instruction
        )

        obs = {
            "text": prompt,
            "phase": 0,
            "step": 0,
        }

        logger.debug(f"Environment reset with instruction: {self.current_instruction}")

        return obs

    def step(
        self, action: int
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one environment step: s_{t+1} = P(s_t, a_t).

        Args:
            action: Token ID to append

        Returns:
            observation: Next state s_{t+1}
            reward: r_t (0 for non-terminal, R_terminal for terminal)
            done: Whether episode ended
            info: Additional debugging information

        Note:
            The main training loop uses rollout_trajectory() directly.
            This interface is provided for OpenEnv framework compatibility.
        """
        raise NotImplementedError(
            "Use MathEnvironment.rollout_trajectory() for PPO training. "
            "This step() interface is for OpenEnv compatibility only."
        )

    def render(self, mode: str = "human"):
        """Render current state for debugging."""
        if mode == "human":
            print("\n" + "=" * 80)
            print(f"PHASE: {self.current_phase}")
            print(f"STEP:  {self.current_step}")
            print(f"INSTRUCTION: {self.current_instruction}")
            if self.current_question:
                print(f"QUESTION: {self.current_question}")
            print("=" * 80 + "\n")
        elif mode == "ansi":
            return f"Phase: {self.current_phase}, Step: {self.current_step}"

    def close(self) -> None:
        """Clean up resources."""
        pass


class OpenEnvAgent:
    """
    OpenEnv-compatible agent wrapper.

    Wraps our PPO policy for use with OpenEnv evaluation.
    """

    def __init__(self, ppo_trainer) -> None:
        """
        Args:
            ppo_trainer: Our PPOTrainer instance
        """
        self.ppo_trainer = ppo_trainer
        self.policy = ppo_trainer.policy
        self.tokenizer = ppo_trainer.tokenizer

    def act(self, observation: Dict[str, Any]) -> int:
        """
        Select action given observation.

        Args:
            observation: Current state from environment

        Returns:
            action: Token ID to take
        """
        text = observation["text"]
        inputs = self.tokenizer(text, return_tensors="pt").to(self.policy.device)

        with torch.no_grad():
            outputs = self.policy(**inputs)
            logits = outputs.logits[0, -1, :]

            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, num_samples=1).item()

        return action

    def reset(self) -> None:
        """Reset agent state between episodes."""
        pass
