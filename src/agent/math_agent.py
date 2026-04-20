"""
MathAgent: Dual-mode LLM agent for problem generation and solving.

The same Qwen2.5-Math model with QLoRA adapters serves two roles:
  - Generator Mode: creates math problems targeting a specific skill
  - Solver Mode:    solves problems step-by-step with verifiable outputs

A lightweight critic (value) head is attached on top of the frozen
transformer for PPO advantage estimation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "o_proj", "gate_proj"]
    )
    bias: str = "none"


@dataclass
class AgentConfig:
    model_name: str = "Qwen/Qwen2.5-Math-7B-Instruct"
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"     # compute in bf16, store in 4-bit NF4
    bnb_4bit_quant_type: str = "nf4"
    max_new_tokens_generator: int = 512
    max_new_tokens_solver: int = 1024
    max_solution_steps: int = 10
    temperature: float = 0.8
    top_p: float = 0.95
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    device: str = "auto"


# ---------------------------------------------------------------------------
# Critic head
# ---------------------------------------------------------------------------

class CriticHead(nn.Module):
    """Single linear layer that maps the last hidden state → scalar value."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.linear.bias)
        nn.init.orthogonal_(self.linear.weight, gain=0.01)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (batch, seq_len, hidden_size)
        # pool over last valid token → (batch, hidden_size)
        pooled = hidden_states[:, -1, :]
        return self.linear(pooled).squeeze(-1)   # (batch,)


# ---------------------------------------------------------------------------
# SolutionTrace  (returned by solve())
# ---------------------------------------------------------------------------

@dataclass
class SolutionStep:
    text: str
    token_ids: torch.Tensor           # shape (seq_len,)
    log_probs: torch.Tensor           # shape (seq_len,)
    value_estimate: float


@dataclass
class SolutionTrace:
    steps: List[SolutionStep]
    final_answer: str
    success: bool = False             # filled in by the verifier
    total_steps: int = 0

    def __post_init__(self):
        self.total_steps = len(self.steps)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

GENERATOR_SYSTEM_PROMPT = (
    "You are a mathematics problem creator. "
    "Given a skill name and required techniques, generate ONE clear, "
    "self-contained math problem at the requested difficulty. "
    "Output ONLY the problem statement. Do NOT include a solution."
)

SOLVER_SYSTEM_PROMPT = (
    "You are a step-by-step math solver. "
    "Solve the given problem one step at a time. "
    "Each step must be on its own line, starting with 'Step N:'. "
    "End with a line starting with 'Final Answer:'. "
    "Write every mathematical expression in Python/SymPy syntax "
    "so it can be verified programmatically."
)

SOLVER_STEP_SYSTEM_PROMPT = (
    "You are continuing to solve a math problem step by step. "
    "Given the problem and all steps so far, produce ONLY the next single step "
    "in the format 'Step N: <expression or explanation>'. "
    "When finished, write 'Final Answer: <answer>'."
)


# ---------------------------------------------------------------------------
# MathAgent
# ---------------------------------------------------------------------------

class MathAgent:
    """
    Dual-mode agent wrapping Qwen2.5-Math-7B with QLoRA + Critic head.

    Usage
    -----
    agent = MathAgent(config)
    problem = agent.generate_problem(skill_name="Power rule", techniques=["power_rule"], difficulty=0.5)
    trace   = agent.solve(problem)
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self._build_model_and_tokenizer()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_model_and_tokenizer(self):
        cfg = self.config
        compute_dtype = getattr(torch, cfg.bnb_4bit_compute_dtype)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=cfg.load_in_4bit,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=True,
        ) if cfg.load_in_4bit else None

        print(f"[MathAgent] Loading base model: {cfg.model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            quantization_config=bnb_config,
            device_map=cfg.device,
            trust_remote_code=True,
            dtype=compute_dtype if not cfg.load_in_4bit else None,
        )

        lora_cfg = cfg.lora
        peft_config = LoraConfig(
            r=lora_cfg.rank,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=lora_cfg.dropout,
            target_modules=lora_cfg.target_modules,
            bias=lora_cfg.bias,
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(base_model, peft_config)
        self.model.print_trainable_parameters()

        hidden_size = self.model.config.hidden_size
        self.critic = CriticHead(hidden_size)
        # Move critic to same device as model
        device = next(self.model.parameters()).device
        self.critic = self.critic.to(device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        print("[MathAgent] Model and tokenizer ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_problem(
        self,
        skill_name: str,
        techniques: List[str],
        difficulty: float,
        exploration_mode: bool = False,
    ) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """
        Generator mode: produce a math problem for the given skill.

        Returns
        -------
        problem_text : str
        token_ids    : Tensor  shape (seq_len,)
        log_probs    : Tensor  shape (seq_len,)
        """
        difficulty_label = self._difficulty_to_label(difficulty, exploration_mode)
        user_content = (
            f"Skill: {skill_name}\n"
            f"Techniques required: {', '.join(techniques)}\n"
            f"Difficulty: {difficulty_label}\n\n"
            "Generate ONE math problem:"
        )
        prompt = self._build_chat_prompt(GENERATOR_SYSTEM_PROMPT, user_content)
        inputs = self._tokenize(prompt)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens_generator,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_ids = outputs.sequences[0, inputs["input_ids"].shape[1]:]
        log_probs = self._scores_to_log_probs(outputs.scores, generated_ids)
        problem_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        return problem_text, generated_ids, log_probs

    def solve(
        self,
        problem: str,
        max_steps: Optional[int] = None,
    ) -> SolutionTrace:
        """
        Solver mode: produce a step-by-step solution.

        Each step is generated auto-regressively; after each step the
        partial context is extended so the model maintains full history.

        Returns
        -------
        SolutionTrace with individual steps and final answer.
        """
        max_steps = max_steps or self.config.max_solution_steps
        steps: List[SolutionStep] = []
        context = problem
        final_answer = ""

        for step_idx in range(1, max_steps + 1):
            user_content = (
                f"Problem:\n{problem}\n\n"
                f"Steps so far:\n{context if context != problem else '(none)'}\n\n"
                f"Produce Step {step_idx} or 'Final Answer:' if done:"
            )
            prompt = self._build_chat_prompt(SOLVER_STEP_SYSTEM_PROMPT, user_content)
            inputs = self._tokenize(prompt)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    output_hidden_states=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            generated_ids = outputs.sequences[0, inputs["input_ids"].shape[1]:]
            log_probs = self._scores_to_log_probs(outputs.scores, generated_ids)
            step_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # Extract value estimate from last hidden state of generated tokens
            value_estimate = self._estimate_value(outputs, inputs["input_ids"].shape[1])

            step = SolutionStep(
                text=step_text,
                token_ids=generated_ids,
                log_probs=log_probs,
                value_estimate=value_estimate,
            )
            steps.append(step)

            # Check if model is done
            if step_text.startswith("Final Answer:"):
                final_answer = step_text.replace("Final Answer:", "").strip()
                break

            # Extend context for next step
            context = context + "\n" + step_text

        if not final_answer and steps:
            # Extract from last step if model ran out of steps
            last = steps[-1].text
            if "Final Answer:" in last:
                final_answer = last.split("Final Answer:")[-1].strip()

        return SolutionTrace(steps=steps, final_answer=final_answer)

    def get_action_log_probs(
        self,
        context_ids: torch.Tensor,
        action_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log-probabilities of `action_ids` given `context_ids`.
        Used by PPO to compute the importance sampling ratio.

        Parameters
        ----------
        context_ids : Tensor  shape (context_len,)
        action_ids  : Tensor  shape (action_len,)

        Returns
        -------
        log_probs : Tensor  shape (action_len,)
        """
        full_ids = torch.cat([context_ids, action_ids], dim=0).unsqueeze(0)
        full_ids = full_ids.to(self.model.device)

        with torch.no_grad():
            logits = self.model(full_ids).logits  # (1, seq_len, vocab_size)

        # Shift: predict token at position i using logits at position i-1
        context_len = context_ids.shape[0]
        action_logits = logits[0, context_len - 1 : context_len - 1 + action_ids.shape[0]]
        log_probs = torch.log_softmax(action_logits, dim=-1)

        action_ids_device = action_ids.to(self.model.device)
        return log_probs.gather(1, action_ids_device.unsqueeze(1)).squeeze(1)

    def get_value(self, context_ids: torch.Tensor) -> torch.Tensor:
        """
        Estimate state value V(s) using the critic head.

        Parameters
        ----------
        context_ids : Tensor  shape (seq_len,) or (batch, seq_len)

        Returns
        -------
        value : Tensor  scalar or (batch,)
        """
        if context_ids.dim() == 1:
            context_ids = context_ids.unsqueeze(0)

        context_ids = context_ids.to(self.model.device)

        with torch.no_grad():
            hidden = self.model(
                context_ids,
                output_hidden_states=True,
            ).hidden_states[-1]   # (batch, seq_len, hidden_size)

        return self.critic(hidden)   # (batch,)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save LoRA adapters + critic head."""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pt"))
        print(f"[MathAgent] Saved to {path}")

    @classmethod
    def load(cls, path: str, config: AgentConfig) -> "MathAgent":
        """Load from a saved checkpoint directory."""
        agent = cls.__new__(cls)
        agent.config = config

        compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config.load_in_4bit,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=True,
        ) if config.load_in_4bit else None

        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            device_map=config.device,
            trust_remote_code=True,
        )
        agent.model = PeftModel.from_pretrained(base_model, path)

        hidden_size = agent.model.config.hidden_size
        agent.critic = CriticHead(hidden_size)
        critic_path = os.path.join(path, "critic.pt")
        if os.path.exists(critic_path):
            agent.critic.load_state_dict(torch.load(critic_path, map_location="cpu"))
        device = next(agent.model.parameters()).device
        agent.critic = agent.critic.to(device)

        agent.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        if agent.tokenizer.pad_token is None:
            agent.tokenizer.pad_token = agent.tokenizer.eos_token

        print(f"[MathAgent] Loaded from {path}")
        return agent

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_chat_prompt(self, system: str, user: str) -> str:
        """Format a chat prompt using the Qwen chat template."""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _tokenize(self, prompt: str) -> dict:
        return self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
        ).to(self.model.device)

    @staticmethod
    def _scores_to_log_probs(
        scores: Tuple[torch.Tensor, ...],
        generated_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert HuggingFace generation scores (un-normalized logits per step)
        to per-token log-probabilities.
        """
        log_probs = []
        for i, score in enumerate(scores):
            lp = torch.log_softmax(score[0], dim=-1)
            token_id = generated_ids[i]
            log_probs.append(lp[token_id].item())
        return torch.tensor(log_probs)

    def _estimate_value(self, outputs, prompt_len: int) -> float:
        """Extract critic value from the last hidden state of generated tokens."""
        if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
            return 0.0
        # outputs.hidden_states: tuple of tuples; last element is the last generation step
        last_hidden = outputs.hidden_states[-1][-1]  # (batch, seq_len, hidden)
        last_token_hidden = last_hidden[:, -1:, :]   # (1, 1, hidden)
        with torch.no_grad():
            value = self.critic(last_token_hidden).item()
        return value

    @staticmethod
    def _difficulty_to_label(difficulty: float, exploration_mode: bool) -> str:
        if exploration_mode:
            return "easy (exploration mode)"
        if difficulty < 0.3:
            return "easy"
        elif difficulty < 0.6:
            return "medium"
        elif difficulty < 0.8:
            return "hard"
        else:
            return "very hard"
