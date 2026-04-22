"""
Ray worker that generates PPO rollouts on a single GPU.
"""

from __future__ import annotations

import io
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.rl.math_environment_curriculum import CurriculumMathEnvironment
from src.rl.mdp_components import Trajectory
from src.rl.value_network import ValueHead

logger = logging.getLogger(__name__)


def _load_policy_and_value(base_model_path: str) -> tuple[Any, ValueHead, Any]:
    model_path = Path(base_model_path)
    is_adapter = (model_path / "adapter_config.json").exists()

    if is_adapter:
        meta_file = model_path / "pipeline_meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            base_model_name = meta.get("base_model", "Qwen/Qwen2.5-Math-1.5B-Instruct")
        else:
            base_model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        policy = PeftModel.from_pretrained(base_model, base_model_path).merge_and_unload()
        value = ValueHead(base_model_name).to(policy.device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        policy = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        value = ValueHead(base_model_path).to(policy.device)

    return policy, value, tokenizer


def _deserialize_bytes(payload: bytes) -> Dict[str, torch.Tensor]:
    buffer = io.BytesIO(payload)
    return torch.load(buffer, map_location="cpu")


def _apply_named_parameters(module: torch.nn.Module, params: Dict[str, torch.Tensor]) -> int:
    own_params = dict(module.named_parameters())
    updated = 0
    for name, tensor in params.items():
        if name not in own_params:
            continue
        own_params[name].data.copy_(tensor.to(device=own_params[name].device, dtype=own_params[name].dtype))
        updated += 1
    return updated


def _actor_options(gpu: bool = True) -> Dict[str, Any]:
    options: Dict[str, Any] = {"max_restarts": -1, "max_task_retries": 2}
    if gpu:
        options["num_gpus"] = 1
    return options


def create_rollout_worker_class(gpu: bool = True):
    import ray

    @ray.remote(**_actor_options(gpu=gpu))
    class DistributedRolloutWorker:
        """Single-GPU rollout worker with optional vLLM question generation."""

        def __init__(
            self,
            worker_id: int,
            base_model: str,
            reference_questions: Optional[List[str]] = None,
            curriculum_checkpoint_dir: str = "checkpoints/ppo_training_curriculum/curriculum",
            max_question_tokens: int = 200,
            max_solution_tokens: int = 500,
            temperature: float = 0.7,
            top_p: float = 0.9,
            consensus_temperature: float = 0.5,
            use_vllm: bool = True,
            vllm_batch_size: int = 8,
            vllm_tensor_parallel_size: int = 1,
        ) -> None:
            self.worker_id = worker_id
            self.base_model = base_model
            self.policy, self.value, self.tokenizer = _load_policy_and_value(base_model)
            self.env = CurriculumMathEnvironment(
                policy_model=self.policy,
                value_model=self.value,
                tokenizer=self.tokenizer,
                reference_questions=reference_questions or [],
                curriculum_checkpoint_dir=curriculum_checkpoint_dir,
                max_question_tokens=max_question_tokens,
                max_solution_tokens=max_solution_tokens,
                temperature=temperature,
                top_p=top_p,
                consensus_temperature=consensus_temperature,
                use_vllm=use_vllm,
                vllm_batch_size=vllm_batch_size,
                vllm_tensor_parallel_size=vllm_tensor_parallel_size,
            )
            self.last_rollout_seconds = 0.0

        def ping(self) -> Dict[str, Any]:
            return {
                "worker_id": self.worker_id,
                "device": str(next(self.policy.parameters()).device),
                "use_vllm": self.env.vllm_generator is not None,
            }

        def generate_rollouts(
            self,
            num_rollouts: int,
            batch_size: Optional[int] = None,
            verbose: bool = False,
        ) -> Dict[str, Any]:
            start = time.perf_counter()
            if self.env.vllm_generator is not None:
                trajectories = self.env.collect_rollouts_batched(
                    num_trajectories=num_rollouts,
                    batch_size=batch_size,
                    verbose=verbose,
                )
            else:
                trajectories = self.env.collect_rollouts(
                    num_trajectories=num_rollouts,
                    verbose=verbose,
                )
            elapsed = time.perf_counter() - start
            self.last_rollout_seconds = elapsed

            for trajectory in trajectories:
                trajectory.metadata["worker_id"] = self.worker_id
                trajectory.metadata["worker_rollout_seconds"] = elapsed

            return {
                "worker_id": self.worker_id,
                "elapsed_seconds": elapsed,
                "num_rollouts": len(trajectories),
                "trajectories": trajectories,
            }

        def update_trainable_weights(
            self,
            policy_trainable_bytes: bytes,
            value_head_bytes: bytes,
        ) -> Dict[str, Any]:
            policy_trainable = _deserialize_bytes(policy_trainable_bytes)
            value_head_state = _deserialize_bytes(value_head_bytes)
            updated_policy = _apply_named_parameters(self.policy, policy_trainable)
            self.value.value_head.load_state_dict(value_head_state, strict=False)
            return {
                "worker_id": self.worker_id,
                "updated_policy_params": updated_policy,
                "updated_value_head_params": len(value_head_state),
            }

        def get_curriculum_state(self) -> Dict[str, Any]:
            return self.env.curriculum_manager.get_curriculum_stats()

        def shutdown(self) -> bool:
            del self.env
            del self.policy
            del self.value
            torch.cuda.empty_cache()
            return True

    return DistributedRolloutWorker


DistributedRolloutWorker = create_rollout_worker_class(gpu=True)
DistributedRolloutWorkerCPU = create_rollout_worker_class(gpu=False)

