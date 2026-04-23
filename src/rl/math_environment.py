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
import time

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
            device: Optional explicit compute device.  Defaults to the
                device of ``policy_model``'s first parameter.
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
        Generate text with PPO-grade per-step logging — fast path.

        Design
        ------
        The old loop ran one FULL model forward over the entire growing
        sequence at every step (re-doing all previous tokens' attention
        from scratch), plus a second full ValueHead forward.  For T=500
        that was ~500*500/2 ≈ 125 000 token-forwards — the single
        biggest bottleneck in the training pipeline.

        This rewrite does the same math in O(T) instead of O(T^2):

        1. One call to HuggingFace ``generate(use_cache=True)``.  KV-cache
           keeps attention cost constant per step, and ``output_logits=True``
           returns the RAW pre-processor logit at each step — which is
           exactly the distribution ``PPOTrainer._policy_logits_at_state``
           re-computes during the update.  That keeps the PPO importance
           ratio ``exp(new - old)`` mathematically valid.

        2. One call to ``value.values_at_positions(...)`` that runs the
           value backbone once over the full trajectory and plucks the
           ``T`` hidden states we need, instead of T separate forwards.

        3. A plain Python loop builds the ``Transition`` objects.  This
           is cheap — just pointer moves and one log_prob gather — it
           does not re-enter the model.

        PPO correctness notes
        ---------------------
        * ``old_log_prob`` and ``entropy`` are computed from RAW logits
          (no temperature, no top-p).  Sampling is done inside
          ``generate()`` using temperature + top-p.  These are two
          different things: we WANT sampling to explore but we WANT the
          stored log-prob to match the un-tempered policy that the PPO
          re-forward sees.  The old loop made exactly the same split
          manually — we are just moving the manual loop into ``generate``.

        * HF's ``output_logits`` (added in transformers 4.38) returns
          the pre-LogitsProcessor logits, not the post-processor scores.
          That is the critical distinction — ``output_scores`` would
          have been wrong because those have been divided by
          temperature and have top-p ``-inf`` masks baked in.

        Args:
            initial_prompt: Starting state s_0 (plain text).
            max_tokens: Generation budget in new tokens.
            phase: "question_generation" or "solution" (carried through
                into every ``State`` for reward routing).

        Returns:
            generated_text: Decoded newly-generated tokens (no prompt,
                special tokens stripped).
            transitions: ``[Transition]`` of length ``<= max_tokens``,
                ending at EOS if the model emitted one.
        """
        # Tokenize prompt.  encode() returns [1, P] already on self.device.
        prompt_ids = self.tokenizer.encode(
            initial_prompt, return_tensors="pt"
        ).to(self.device)
        prompt_length = prompt_ids.shape[1]
        prompt_attn = torch.ones_like(prompt_ids)

        # HF generate requires temperature strictly > 0 when do_sample=True.
        # We still want to expose temperature=0 externally as "greedy",
        # so map very low temperatures onto do_sample=False.
        temperature = float(self.temperature)
        do_sample = temperature > 1e-4
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = eos_id

        gen_kwargs = dict(
            input_ids=prompt_ids,
            attention_mask=prompt_attn,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            use_cache=True,
            output_logits=True,
            return_dict_in_generate=True,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = max(temperature, 1e-6)
            gen_kwargs["top_p"] = float(self.top_p)

        with torch.no_grad():
            try:
                gen_out = self.policy.generate(**gen_kwargs)
            except TypeError as e:
                # Older transformers (<4.38) don't accept output_logits.
                # Fall back to output_scores — that is technically the
                # processor-modified distribution (temperature+top-p
                # already applied), which breaks PPO importance ratios.
                # We make this a loud error rather than a silent fallback
                # so the operator knows to upgrade transformers.
                if "output_logits" in str(e):
                    raise RuntimeError(
                        "transformers<4.38 does not support output_logits; "
                        "upgrade transformers to >=4.38 — PPO correctness "
                        "requires RAW (pre-temperature, pre-top-p) logits "
                        "from generate()."
                    ) from e
                raise

        full_ids = gen_out.sequences  # [1, prompt_length + T_gen]
        T_gen = int(full_ids.shape[1] - prompt_length)
        if T_gen <= 0:
            logger.debug(
                "Phase %s produced zero tokens (prompt_len=%d)",
                phase, prompt_length,
            )
            return "", []

        # Raw per-step logits.  gen_out.logits is a tuple of length T_gen,
        # each element [1, V].  Stack → [T_gen, V].  Cast to fp32 for
        # numerically stable log_softmax.
        raw_logits_per_step = torch.stack(
            [lg[0] for lg in gen_out.logits], dim=0
        ).float()  # [T_gen, V]

        sampled_tokens = full_ids[0, prompt_length:]  # [T_gen]

        raw_log_probs = F.log_softmax(raw_logits_per_step, dim=-1)  # [T_gen, V]
        chosen_log_probs = raw_log_probs.gather(
            1, sampled_tokens.unsqueeze(1)
        ).squeeze(1)  # [T_gen]
        # Entropy of the RAW policy (logging only; PPO computes its own).
        entropies = -(raw_log_probs.exp() * raw_log_probs).sum(dim=-1)  # [T_gen]

        # Batched V(s_t) — ONE backbone forward over the full trajectory.
        # State s_t's "last token" is at index prompt_length + t - 1 in
        # full_ids, for t = 0..T_gen-1, so positions = [P-1, P, ..., P+T-2].
        positions = torch.arange(
            prompt_length - 1,
            prompt_length + T_gen - 1,
            device=self.device,
        )
        full_attn = torch.ones_like(full_ids)
        values = self.value.values_at_positions(
            input_ids=full_ids,
            positions=positions,
            attention_mask=full_attn,
        )  # [T_gen] float32

        # Incremental text fragments (O(T) decode, not O(T^2)).
        #
        # Why: downstream ``State.text`` is used for logging and for
        # debug inspection only; the PPO update uses ``input_ids``.  We
        # accumulate single-token decodes to build per-state text in
        # linear time rather than re-decoding the growing sequence T
        # times (which tokenizers do NOT amortize internally).
        piece_by_piece: List[str] = self.tokenizer.batch_decode(
            [[tok.item()] for tok in sampled_tokens],
            skip_special_tokens=False,
        )

        transitions: List[Transition] = []
        running_text = initial_prompt
        for t in range(T_gen):
            state_input_ids = full_ids[0, : prompt_length + t]
            current_state = State(
                text=running_text,
                input_ids=state_input_ids,
                attention_mask=torch.ones_like(state_input_ids),
                phase=phase,
            )

            action_token = int(sampled_tokens[t].item())
            action = Action(
                token_id=action_token,
                log_prob=float(chosen_log_probs[t].item()),
                entropy=float(entropies[t].item()),
            )

            next_text = running_text + piece_by_piece[t]
            next_input_ids = full_ids[0, : prompt_length + t + 1]
            next_state = State(
                text=next_text,
                input_ids=next_input_ids,
                attention_mask=torch.ones_like(next_input_ids),
                phase=phase,
            )

            is_done = eos_id is not None and action_token == eos_id
            transitions.append(
                Transition(
                    state=current_state,
                    action=action,
                    reward=0.0,
                    next_state=next_state,
                    value=float(values[t].item()),
                    done=is_done,
                )
            )

            running_text = next_text
            if is_done:
                break

        # Decoded generated text (specials stripped, trimmed).  Only the
        # tokens up to (and including) the stop reason count — if we
        # broke on EOS partway through, truncate accordingly.
        generated_ids = full_ids[0, prompt_length : prompt_length + len(transitions)]
        generated_text = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()

        logger.debug(
            "Generated text for phase %s (len=%d, tokens=%d): %s...",
            phase, len(generated_text), len(transitions), generated_text[:150],
        )

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
        trajectories: List[Trajectory] = []
        total_tokens = 0
        start = time.perf_counter()

        for i in range(num_trajectories):
            traj_start = time.perf_counter()
            trajectory = self.rollout_trajectory()
            traj_seconds = time.perf_counter() - traj_start
            traj_tokens = len(trajectory)
            total_tokens += traj_tokens
            trajectories.append(trajectory)

            if verbose:
                logger.info(
                    "  Trajectory %d/%d: %d tokens in %.1fs (%.1f tok/s, reward=%.3f)",
                    i + 1, num_trajectories, traj_tokens, traj_seconds,
                    traj_tokens / max(traj_seconds, 1e-6), trajectory.total_reward,
                )

        if not trajectories:
            return trajectories

        elapsed = time.perf_counter() - start
        mean_reward = sum(t.total_reward for t in trajectories) / len(trajectories)
        mean_length = sum(len(t) for t in trajectories) / len(trajectories)
        tokens_per_sec = total_tokens / max(elapsed, 1e-6)

        logger.info(
            "Rollout complete: %d trajectories, %d total tokens, %.1fs wall, "
            "%.1f tok/s mean | reward=%.3f len=%.1f",
            num_trajectories, total_tokens, elapsed,
            tokens_per_sec, mean_reward, mean_length,
        )

        return trajectories
