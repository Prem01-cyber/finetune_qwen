"""
Multiprocessing rollout worker for single-GPU trajectory generation.
"""

from __future__ import annotations

import io
import json
import logging
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.rl.value_network import ValueHead

logger = logging.getLogger(__name__)


@dataclass
class WorkerRuntimeConfig:
    base_model: str
    reference_questions: List[str]
    curriculum_checkpoint_dir: str
    max_question_tokens: int
    max_solution_tokens: int
    temperature: float
    top_p: float
    consensus_temperature: float
    use_vllm: bool
    vllm_batch_size: int
    vllm_tensor_parallel_size: int
    vllm_gpu_memory_utilization: float


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
        own_params[name].data.copy_(
            tensor.to(device=own_params[name].device, dtype=own_params[name].dtype)
        )
        updated += 1
    return updated


class MultiGPURolloutWorker:
    """Single-GPU rollout worker running inside one process."""

    def __init__(
        self,
        worker_id: int,
        gpu_id: int,
        config: WorkerRuntimeConfig,
    ) -> None:
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.config = config
        self.policy, self.value, self.tokenizer = _load_policy_and_value(config.base_model)
        from src.rl.math_environment_curriculum import CurriculumMathEnvironment

        self.env = CurriculumMathEnvironment(
            policy_model=self.policy,
            value_model=self.value,
            tokenizer=self.tokenizer,
            reference_questions=config.reference_questions,
            curriculum_checkpoint_dir=config.curriculum_checkpoint_dir,
            max_question_tokens=config.max_question_tokens,
            max_solution_tokens=config.max_solution_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            consensus_temperature=config.consensus_temperature,
            use_vllm=config.use_vllm,
            vllm_batch_size=config.vllm_batch_size,
            vllm_tensor_parallel_size=config.vllm_tensor_parallel_size,
            vllm_gpu_memory_utilization=config.vllm_gpu_memory_utilization,
            base_model_path=config.base_model,
        )

    def ping(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "gpu_id": self.gpu_id,
            "device": str(next(self.policy.parameters()).device),
            "use_vllm": self.env.vllm_generator is not None,
        }

    def generate_rollouts(
        self,
        num_rollouts: int,
        batch_size: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        effective_batch_size = max(1, int(batch_size or self.config.vllm_batch_size))
        trajectories = self.env.collect_fresh_rollouts_batched(
            num_trajectories=max(0, int(num_rollouts)),
            batch_size=effective_batch_size,
            verbose=verbose,
        )
        for trajectory in trajectories:
            trajectory.metadata["worker_id"] = self.worker_id
            trajectory.metadata["worker_gpu_id"] = self.gpu_id
        return {
            "worker_id": self.worker_id,
            "gpu_id": self.gpu_id,
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

    def shutdown(self) -> bool:
        del self.env
        del self.policy
        del self.value
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True


def rollout_worker_entrypoint(
    worker_id: int,
    gpu_id: int,
    config: WorkerRuntimeConfig,
    task_queue: Any,
    result_queue: Any,
) -> None:
    # Isolate this process to one physical GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    worker = MultiGPURolloutWorker(worker_id=worker_id, gpu_id=gpu_id, config=config)
    while True:
        message = task_queue.get()
        request_id = message.get("request_id")
        command = str(message.get("cmd", ""))
        try:
            if command == "ping":
                payload = worker.ping()
            elif command == "generate":
                payload = worker.generate_rollouts(
                    num_rollouts=int(message.get("num_rollouts", 0)),
                    batch_size=message.get("batch_size"),
                    verbose=bool(message.get("verbose", False)),
                )
            elif command == "sync":
                payload = worker.update_trainable_weights(
                    policy_trainable_bytes=message["policy_trainable_bytes"],
                    value_head_bytes=message["value_head_bytes"],
                )
            elif command == "shutdown":
                payload = worker.shutdown()
                result_queue.put({"request_id": request_id, "ok": True, "result": payload})
                break
            else:
                raise ValueError(f"Unknown worker command: {command}")

            result_queue.put({"request_id": request_id, "ok": True, "result": payload})
        except Exception as exc:
            result_queue.put(
                {
                    "request_id": request_id,
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
