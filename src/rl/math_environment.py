"""
Mathematical Reasoning Environment for PPO

This is the "environment" in the RL sense:
- Defines how the agent interacts with the task
- Handles question generation and solution phases
- Computes rewards using SymPy verification
- Integrates with OpenEnv framework
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple, Optional
import random
import logging

from src.rl.mdp_components import State, Action, Transition, Trajectory
from src.rl.value_network import ValueHead
from src.rl.reward_calculator import RewardCalculator
from src.sft.step_verify_sympy import verify_solution_text

logger = logging.getLogger(__name__)


# Question generation prompts (curriculum of increasing difficulty)
QUESTION_PROMPTS = [
    # Level 1: 2-step problems
    "Generate a grade-school math problem about money and fractions that requires exactly 2 steps to solve.",
    "Create a simple word problem about time and distance with 2 calculation steps.",
    "Generate a problem about percentages and discounts requiring 2 steps.",

    # Level 2: 3-step problems
    "Generate a math problem about shopping with multiple items requiring 3 steps.",
    "Create a problem about ratios and proportions with 3 calculation steps.",
    "Generate a word problem combining fractions and multiplication in 3 steps.",

    # Level 3: 4-5 step problems
    "Generate a complex problem about compound percentages requiring 4-5 steps.",
    "Create a multi-stage problem about resource allocation with 4-5 steps.",
    "Generate a challenging problem combining multiple operations in 4-5 steps.",
]


class MathEnvironment:
    """
    Environment for mathematical reasoning self-play.

    This implements the "environment" side of the agent-environment loop.
    The agent (LM policy) interacts by:
    1. Generating a question given an instruction
    2. Solving the generated question
    3. Receiving reward based on verification

    Math:
        Environment provides: P(s'|s,a), R(s,a,s')
        Agent provides: π(a|s)
    """

    def __init__(
        self,
        policy_model: AutoModelForCausalLM,
        value_model: ValueHead,
        tokenizer: AutoTokenizer,
        reward_calculator: RewardCalculator,
        max_question_tokens: int = 200,
        max_solution_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            policy_model: Language model (actor)
            value_model: Value network (critic)
            tokenizer: Tokenizer
            reward_calculator: Computes rewards from Q, S pairs
            max_question_tokens: Max tokens for question generation
            max_solution_tokens: Max tokens for solution generation
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            device: Optional explicit compute device.  This is needed when
                using DeepSpeed ZeRO-3 with CPU offload, where
                ``next(policy_model.parameters()).device`` would return
                ``cpu`` (params are offloaded) but we actually want to run
                on the local GPU.
        """
        self.policy = policy_model
        self.value = value_model
        self.tokenizer = tokenizer
        self.reward_calculator = reward_calculator

        self.max_question_tokens = max_question_tokens
        self.max_solution_tokens = max_solution_tokens
        self.temperature = temperature
        self.top_p = top_p

        if device is not None:
            self.device = torch.device(device)
        else:
            # Fall back to the first parameter's device.  This is only
            # meaningful when the model is *not* sharded / offloaded.
            try:
                self.device = next(policy_model.parameters()).device
            except StopIteration:
                self.device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )

    def sample_instruction(self) -> str:
        """
        Sample initial state s_0 ~ μ_0.

        Returns curriculum-based prompt for question generation.
        """
        return random.choice(QUESTION_PROMPTS)

    def format_question_generation_prompt(self, instruction: str) -> str:
        """
        Format prompt for question generation phase.

        Returns:
            s_0^gen = "### Task: Generate Question\n<instruction>"
        """
        return f"### Task: Generate Question\n{instruction}"

    def format_solution_prompt(self, question: str) -> str:
        """
        Format prompt for solution generation phase.
        
        Uses chat template with system prompt to ensure proper formatting.
        This matches the format used by TripleVerifier for consensus solutions.

        Returns:
            Formatted prompt with chat template
        """
        system_prompt = (
            "You are a step-by-step math solver. "
            "Solve the given problem one step at a time. "
            "Each step must be on its own line, starting with 'Step N:'. "
            "End with a line starting with 'Final Answer:'. "
            "Write every mathematical expression in Python/SymPy syntax."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"### Task: Solve Problem\nProblem: {question}\nSolution:"},
        ]
        
        # Apply chat template (matches training format)
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        return prompt

    def generate_with_logging(
        self,
        initial_prompt: str,
        max_tokens: int,
        phase: str,
    ) -> Tuple[str, List[Transition]]:
        """
        Generate text while logging all transitions for PPO.

        This is the core interaction loop:
        For t = 0 to T:
            1. Observe state s_t
            2. Compute V(s_t) from critic
            3. Sample action a_t ~ π(·|s_t)
            4. Compute log π(a_t|s_t)
            5. Transition to s_{t+1} = s_t ⊕ a_t
            6. Store transition (s_t, a_t, V(s_t))

        Args:
            initial_prompt: Starting state s_0
            max_tokens: Maximum generation length T
            phase: "question_generation" or "solution"

        Returns:
            generated_text: Complete generated text
            transitions: List of (s_t, a_t, r_t=0, s_{t+1}, V(s_t))
        """
        transitions = []

        # Tokenize initial state
        current_text = initial_prompt
        current_ids = self.tokenizer.encode(
            current_text, return_tensors="pt"
        ).to(self.device)
        
        # Track prompt length to strip it later
        prompt_length = current_ids.shape[1]

        # Create initial state
        current_state = State(
            text=current_text,
            input_ids=current_ids[0],
            attention_mask=torch.ones_like(current_ids[0]),
            phase=phase,
        )

        # Generation loop
        for step in range(max_tokens):
            # Compute value V(s_t)
            with torch.no_grad():
                value_estimate = self.value(
                    input_ids=current_ids,
                    attention_mask=torch.ones_like(current_ids),
                ).item()

            # Forward through policy to get action distribution
            with torch.no_grad():
                outputs = self.policy(
                    input_ids=current_ids,
                    attention_mask=torch.ones_like(current_ids),
                    return_dict=True,
                )

            # Get logits for next token
            next_token_logits = outputs.logits[0, -1, :]  # [vocab_size]

            # Apply temperature
            next_token_logits = next_token_logits / self.temperature

            # Apply top-p (nucleus) sampling
            sorted_logits, sorted_indices = torch.sort(
                next_token_logits, descending=True
            )
            cumulative_probs = torch.cumsum(
                F.softmax(sorted_logits, dim=-1), dim=-1
            )

            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[0] = False  # Keep at least one token

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = float("-inf")

            # Sample action a_t ~ π(·|s_t)
            probs = F.softmax(next_token_logits, dim=-1)
            action_token = torch.multinomial(probs, num_samples=1).item()

            # Compute log probability log π(a_t|s_t)
            log_prob = F.log_softmax(next_token_logits, dim=-1)[action_token].item()

            # Compute entropy H(π(·|s_t))
            entropy = -(probs * torch.log(probs + 1e-10)).sum().item()

            # Create action
            action = Action(
                token_id=action_token,
                log_prob=log_prob,
                entropy=entropy,
            )

            # Transition to next state s_{t+1} = s_t ⊕ a_t
            next_ids = torch.cat(
                [current_ids, torch.tensor([[action_token]], device=self.device)],
                dim=1,
            )

            next_text = self.tokenizer.decode(
                next_ids[0], skip_special_tokens=False
            )

            next_state = State(
                text=next_text,
                input_ids=next_ids[0],
                attention_mask=torch.ones_like(next_ids[0]),
                phase=phase,
            )

            # Store transition (reward will be set later — sparse)
            transition = Transition(
                state=current_state,
                action=action,
                reward=0.0,
                next_state=next_state,
                value=value_estimate,
                done=False,
            )

            transitions.append(transition)

            # Update current state
            current_state = next_state
            current_ids = next_ids
            current_text = next_text

            # Check for EOS token
            if action_token == self.tokenizer.eos_token_id:
                transitions[-1].done = True
                break

        # Decode only the newly generated tokens (skip the prompt)
        generated_ids = current_ids[0][prompt_length:]
        generated_text = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()
        
        logger.debug(f"Generated text for phase {phase} (len={len(generated_text)}): {generated_text[:150]}...")

        return generated_text, transitions

    def compute_reward(self, question: str, solution: str) -> Dict:
        """
        Compute terminal reward R(τ) for complete trajectory.

        Uses SymPy verification as ground truth.

        Math:
            R_terminal = 0.5·R_question + 0.5·R_solution

        where:
            R_question = f(solvability, novelty, difficulty)
            R_solution = f(correctness, format, efficiency)

        Args:
            question: Generated question text
            solution: Generated solution text

        Returns:
            Dict with:
                - combined_score: Terminal reward ∈ [0, 1]
                - question_metrics: Breakdown
                - solution_metrics: Breakdown
        """
        # Call with correct parameter names
        reward_result = self.reward_calculator.calculate_reward(
            generated_question=question,
            generated_solution=solution,
        )

        # Convert CombinedReward to dict format expected by the rest of the code
        return {
            "combined_score": reward_result.combined_score,
            "question_metrics": {
                "overall_score": reward_result.question_metrics.overall_score,
                "solvability": reward_result.question_metrics.solvability_score,
                "novelty": reward_result.question_metrics.novelty_score,
                "difficulty": reward_result.question_metrics.difficulty_score,
            },
            "solution_metrics": {
                "overall_score": reward_result.solution_metrics.overall_score,
                "correctness": reward_result.solution_metrics.correctness_score,
                "format_compliance": reward_result.solution_metrics.format_score,
                "efficiency": reward_result.solution_metrics.efficiency_score,
                "steps_total": reward_result.solution_metrics.steps_total,
            },
        }

    def rollout_trajectory(self) -> Trajectory:
        """
        Execute complete episode: generate question → solve → compute reward.

        This implements one full MDP episode τ = (s_0, a_0, ..., s_T).

        Returns:
            Complete trajectory with rewards assigned.
        """
        trajectory = Trajectory()

        # Sample initial instruction s_0 ~ μ_0
        instruction = self.sample_instruction()

        # ===== PHASE 1: QUESTION GENERATION =====
        question_prompt = self.format_question_generation_prompt(instruction)

        logger.debug(f"Generating question with prompt: {instruction}")

        generated_question, question_transitions = self.generate_with_logging(
            initial_prompt=question_prompt,
            max_tokens=self.max_question_tokens,
            phase="question_generation",
        )

        logger.debug(f"Generated question: {generated_question[:100]}...")

        # ===== PHASE 2: SOLUTION GENERATION =====
        solution_prompt = self.format_solution_prompt(generated_question)

        logger.debug("Generating solution...")

        generated_solution, solution_transitions = self.generate_with_logging(
            initial_prompt=solution_prompt,
            max_tokens=self.max_solution_tokens,
            phase="solution",
        )

        logger.debug(f"Generated solution: {generated_solution[:100]}...")

        # ===== PHASE 3: REWARD COMPUTATION =====
        reward_result = self.compute_reward(
            question=generated_question,
            solution=generated_solution,
        )

        terminal_reward = reward_result["combined_score"]

        # Log with consensus breakdown if available
        sol_metrics = reward_result['solution_metrics']
        if 'consensus_score' in sol_metrics and 'sympy_score' in sol_metrics:
            # Consensus mode: show SymPy, Consensus, Format breakdown
            logger.info(
                f"Trajectory reward: {terminal_reward:.3f} "
                f"(SymPy: {sol_metrics.get('sympy_score', 0):.3f}, "
                f"Consensus: {sol_metrics.get('consensus_score', 0):.3f}, "
                f"Format: {sol_metrics.get('format_compliance', 0):.3f})"
            )
        else:
            # Standard mode: show Q, S breakdown
            logger.info(
                f"Trajectory reward: {terminal_reward:.3f} "
                f"(Q: {reward_result['question_metrics']['overall_score']:.3f}, "
                f"S: {reward_result['solution_metrics']['overall_score']:.3f})"
            )

        # ===== PHASE 4: ASSIGN REWARDS =====
        # Sparse rewards: only terminal state gets reward
        all_transitions = question_transitions + solution_transitions

        for i, transition in enumerate(all_transitions):
            if i == len(all_transitions) - 1:
                transition.reward = terminal_reward
            else:
                transition.reward = 0.0

            trajectory.add(transition)

        trajectory.metadata = {
            "instruction": instruction,
            "generated_question": generated_question,
            "generated_solution": generated_solution,
            "reward_breakdown": reward_result,
            "question_length": len(question_transitions),
            "solution_length": len(solution_transitions),
            "total_length": len(all_transitions),
        }

        return trajectory

    def collect_rollouts(
        self,
        num_trajectories: int,
        verbose: bool = True,
    ) -> List[Trajectory]:
        """
        Collect batch of trajectories for PPO update.

        Args:
            num_trajectories: Number of episodes to collect
            verbose: Print progress

        Returns:
            List of complete trajectories
        """
        trajectories = []

        for i in range(num_trajectories):
            if verbose and (i + 1) % 10 == 0:
                logger.info(f"Collected {i + 1}/{num_trajectories} trajectories")

            trajectory = self.rollout_trajectory()
            trajectories.append(trajectory)

        mean_reward = sum(t.total_reward for t in trajectories) / len(trajectories)
        mean_length = sum(len(t) for t in trajectories) / len(trajectories)

        logger.info(
            f"Rollout complete: {num_trajectories} trajectories, "
            f"mean reward: {mean_reward:.3f}, mean length: {mean_length:.1f}"
        )

        return trajectories
