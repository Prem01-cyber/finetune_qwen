"""
Utilities for vLLM-powered rollout text generation.
"""

from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    from vllm import LLM, SamplingParams
except ImportError:  # pragma: no cover - depends on optional dependency
    LLM = None  # type: ignore[assignment]
    SamplingParams = None  # type: ignore[assignment]


class VLLMQuestionGenerator:
    """Fast batched question generation using vLLM."""

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 4096,
    ) -> None:
        if LLM is None or SamplingParams is None:
            raise ImportError(
                "vLLM is not installed. Install it with `pip install vllm`."
            )

        self.model_path = model_path
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
        )
        logger.info("Initialized vLLM question generator for model: %s", model_path)

    def generate_questions_batch(
        self,
        prompts: List[str],
        max_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate a batch of questions in parallel."""
        if not prompts:
            return []

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )
        outputs = self.llm.generate(prompts, sampling_params)
        return [item.outputs[0].text for item in outputs]
