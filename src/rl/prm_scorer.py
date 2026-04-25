"""
Process Reward Model (PRM) scorer for step-level correctness.

Uses Qwen/Qwen2.5-Math-PRM-7B — a purpose-built process reward model that
assigns each reasoning step a probability of being correct.  This replaces
the "consensus voting across three samples from the same policy" signal,
which was groupthink (three samples agree because they share the same
failure mode) and therefore uncorrelated with GSM8K accuracy.

How PRM scoring works
---------------------
* The input is ``question`` + an assistant response where each reasoning
  step is separated by the special token ``<extra_0>`` (also appended
  after the final step).
* The model runs a single forward pass and emits a classification logit
  (``[negative, positive]``) at every ``<extra_0>`` position.
* ``softmax`` → the positive-class probability is the per-step reward in
  ``[0, 1]``.

Training integration
--------------------
Loaded once at startup alongside the policy.  Scored during rollout
``compute_reward`` calls (no gradient flow).  Quantise to 4-bit via
``bitsandbytes`` to keep VRAM under ~5 GB so there is ample headroom for
policy training on a single 80 GB A100.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from src.sft.solution_format import _step_bodies, extract_final_answer_numeric_str
from src.utils.attn_backend import select_attn_implementation

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
# Qwen PRM's step separator token.  Hard-coded by the model; do not change.
STEP_SEP_TOKEN = "<extra_0>"


def extract_prm_steps(solution: str) -> List[str]:
    """
    Split a Qwen-style ``Step N:`` solution into the text fragments the PRM
    expects — one element per reasoning step, with the final-answer line
    appended as a closing step so it gets its own correctness score.

    The ``Step N:`` prefix is stripped so we feed plain reasoning text
    (matches PRM's training distribution, which was Qwen-Math-Instruct
    paragraph-style outputs).
    """
    bodies = _step_bodies(solution)
    steps: List[str] = [b.strip() for b in bodies if b.strip()]
    final_raw = extract_final_answer_numeric_str(solution)
    if final_raw:
        steps.append(f"The answer is \\boxed{{{final_raw.strip()}}}")
    return steps


class ProcessRewardScorer:
    """
    Qwen2.5-Math-PRM-7B scorer.  Memory-efficient: the model is held in
    inference mode on the training device and runs in ``torch.no_grad``.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Math-PRM-7B",
        device: Optional[torch.device] = None,
        load_in_4bit: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        max_input_tokens: int = 4096,
    ):
        self.model_name = model_name
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.max_input_tokens = max_input_tokens

        logger.info(
            "Loading PRM %s (4-bit=%s, dtype=%s) on %s …",
            model_name, load_in_4bit, dtype, self.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
            # PRM forward is eval-only but sequences can be 1-2k tokens
            # when the policy writes a lot of steps; flash-attn 2 cuts the
            # scoring forward by ~2x at those lengths.  Falls back to SDPA.
            "attn_implementation": select_attn_implementation(),
        }
        if load_in_4bit and torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                load_kwargs["device_map"] = {"": self.device}
            except ImportError:
                logger.warning(
                    "bitsandbytes not available; falling back to bf16 PRM load"
                )
                load_in_4bit = False
        if not load_in_4bit:
            load_kwargs["device_map"] = {"": self.device}

        self.model = AutoModel.from_pretrained(model_name, **load_kwargs).eval()

        # Cache separator token id so we don't re-tokenize it every call.
        # encode() returns a list — PRM's step_sep is a single token.
        sep_ids = self.tokenizer.encode(STEP_SEP_TOKEN, add_special_tokens=False)
        if len(sep_ids) != 1:
            raise RuntimeError(
                f"PRM step separator {STEP_SEP_TOKEN!r} tokenized to "
                f"{sep_ids} (expected a single id).  Tokenizer mismatch."
            )
        self.step_sep_id = int(sep_ids[0])

        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
            logger.info(
                "PRM ready.  GPU memory allocated: %.2f GB  step_sep_id=%d",
                mem_alloc, self.step_sep_id,
            )

    @torch.no_grad()
    def score_solution(
        self,
        question: str,
        solution: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> Dict[str, Any]:
        """
        Return per-step correctness probabilities for ``solution``.

        Returns dict with:
            step_scores : List[float]  — per-step prob in [0, 1]
            num_steps   : int
            mean_score  : float        — avg across steps
            min_score   : float        — weakest step (error locator)
            final_score : float        — score on the answer-line step
            degraded    : bool         — True if we returned a zero-length
                                         score list (empty solution, etc.)
        """
        steps = extract_prm_steps(solution)
        if not steps:
            return {
                "step_scores": [],
                "num_steps": 0,
                "mean_score": 0.0,
                "min_score": 0.0,
                "final_score": 0.0,
                "degraded": True,
                "degraded_reason": "no extractable steps",
            }

        assistant_body = STEP_SEP_TOKEN.join(steps) + STEP_SEP_TOKEN
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question.strip()},
            {"role": "assistant", "content": assistant_body},
        ]
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception as exc:
            logger.warning("PRM chat template failed: %s", exc)
            return {
                "step_scores": [],
                "num_steps": len(steps),
                "mean_score": 0.0,
                "min_score": 0.0,
                "final_score": 0.0,
                "degraded": True,
                "degraded_reason": f"chat template error: {exc}",
            }

        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        try:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        except Exception as exc:
            logger.warning("PRM forward pass failed: %s", exc)
            return {
                "step_scores": [],
                "num_steps": len(steps),
                "mean_score": 0.0,
                "min_score": 0.0,
                "final_score": 0.0,
                "degraded": True,
                "degraded_reason": f"forward error: {exc}",
            }

        logits = outputs[0]  # [1, seq_len, 2]
        token_mask = (input_ids == self.step_sep_id)  # [1, seq_len] bool

        # Follow the reference make_step_rewards routine.  We softmax the
        # logits, zero out non-separator positions, then read the positive
        # class (index 1) at each separator.
        probs = F.softmax(logits, dim=-1)  # [1, seq_len, 2]
        probs = probs * token_mask.unsqueeze(-1)
        sample = probs[0]  # [seq_len, 2]
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1]
        step_scores: List[float] = positive_probs.float().cpu().tolist()

        # Truncation may have dropped trailing separators.  Align lengths
        # conservatively by padding missing positions with the mean of what
        # we did see.  Log a warning so callers know the scores are partial.
        if len(step_scores) < len(steps) and step_scores:
            pad_val = float(sum(step_scores) / len(step_scores))
            n_padded = len(steps) - len(step_scores)
            step_scores = step_scores + [pad_val] * n_padded
            logger.warning(
                "PRM: %d/%d steps scored; %d tail step(s) padded with mean=%.3f "
                "(sequence likely truncated at %d tokens).",
                len(step_scores) - n_padded, len(steps), n_padded, pad_val,
                self.max_input_tokens,
            )
        elif len(step_scores) > len(steps):
            step_scores = step_scores[: len(steps)]

        if not step_scores:
            return {
                "step_scores": [],
                "num_steps": len(steps),
                "mean_score": 0.0,
                "min_score": 0.0,
                "final_score": 0.0,
                "degraded": True,
                "degraded_reason": "no separator token in output (truncated?)",
            }

        mean_score = float(sum(step_scores) / len(step_scores))
        min_score = float(min(step_scores))
        final_score = float(step_scores[-1])

        return {
            "step_scores": [float(s) for s in step_scores],
            "num_steps": len(step_scores),
            "mean_score": mean_score,
            "min_score": min_score,
            "final_score": final_score,
            "degraded": False,
            "padded_steps": len(step_scores) < len(steps),  # True if tail was padded
        }

    @torch.no_grad()
    def score_batch(
        self,
        items: List[Dict[str, str]],
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> List[Dict[str, Any]]:
        """Score a list of ``{"question", "solution"}`` items sequentially.

        A proper padded batch path would be ~2-3× faster but needs care to
        handle variable separator counts.  Sequential is simple, correct,
        and a single PRM forward takes ~100-300 ms on an A100 — acceptable
        overhead given self-play generation dominates rollout wall-time.
        """
        return [
            self.score_solution(it["question"], it["solution"], system_prompt)
            for it in items
        ]
