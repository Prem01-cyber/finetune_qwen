"""
Quick DeepSpeed ZeRO-3 smoke test.

Run:
    deepspeed --num_gpus=2 test_deepspeed_setup.py
"""

from __future__ import annotations

import deepspeed
import torch
from transformers import AutoModelForCausalLM


def main() -> None:
    print(f"GPUs available: {torch.cuda.device_count()}")
    print(f"DeepSpeed version: {deepspeed.__version__}")

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Math-1.5B-Instruct",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    ds_config = {
        "train_batch_size": 32,
        "train_micro_batch_size_per_gpu": 16,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "cpu"},
            "offload_param": {"device": "cpu"},
        },
        "bf16": {"enabled": True},
    }

    model_engine, _, _, _ = deepspeed.initialize(model=model, config=ds_config)
    _ = model_engine
    print(f"DeepSpeed initialized successfully on {torch.cuda.device_count()} GPUs")
    print("Model parameters are sharded across GPUs and CPU")


if __name__ == "__main__":
    main()
