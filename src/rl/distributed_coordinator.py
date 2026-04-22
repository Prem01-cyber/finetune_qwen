"""
Coordinator for distributed rollout generation across Ray workers.
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from src.rl.distributed_rollout_worker import (
    DistributedRolloutWorker,
    DistributedRolloutWorkerCPU,
)
from src.rl.mdp_components import Trajectory

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


def _serialize_state(payload: Dict[str, torch.Tensor]) -> bytes:
    buf = io.BytesIO()
    torch.save(payload, buf)
    return buf.getvalue()


def _extract_trainable_state(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu()
        for name, param in module.named_parameters()
        if param.requires_grad
    }


class DistributedRolloutCoordinator:
    """Orchestrates rollout generation and model sync over Ray actors."""

    def __init__(
        self,
        ray_address: str,
        num_workers: int,
        worker_config: WorkerRuntimeConfig,
        use_gpu_workers: bool = True,
    ) -> None:
        import ray

        self.ray = ray
        if not ray.is_initialized():
            ray.init(address=ray_address, ignore_reinit_error=True, log_to_driver=True)

        self.num_workers = num_workers
        self.worker_config = worker_config
        self.use_gpu_workers = use_gpu_workers
        self._worker_class = DistributedRolloutWorker if use_gpu_workers else DistributedRolloutWorkerCPU
        self.workers = [self._create_worker(i) for i in range(num_workers)]

    def _create_worker(self, worker_id: int):
        config = self.worker_config
        return self._worker_class.remote(
            worker_id=worker_id,
            base_model=config.base_model,
            reference_questions=config.reference_questions,
            curriculum_checkpoint_dir=f"{config.curriculum_checkpoint_dir}/worker_{worker_id}",
            max_question_tokens=config.max_question_tokens,
            max_solution_tokens=config.max_solution_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            consensus_temperature=config.consensus_temperature,
            use_vllm=config.use_vllm,
            vllm_batch_size=config.vllm_batch_size,
            vllm_tensor_parallel_size=config.vllm_tensor_parallel_size,
        )

    def health_check(self) -> List[Dict[str, Any]]:
        futures = [worker.ping.remote() for worker in self.workers]
        return list(self.ray.get(futures))

    def _split_rollouts(self, total_rollouts: int) -> List[int]:
        base = total_rollouts // self.num_workers
        remainder = total_rollouts % self.num_workers
        per_worker = [base] * self.num_workers
        for i in range(remainder):
            per_worker[i] += 1
        return per_worker

    def _restart_failed_worker(self, worker_idx: int) -> None:
        logger.warning("Restarting failed worker %d", worker_idx)
        self.workers[worker_idx] = self._create_worker(worker_idx)

    def collect_rollouts_parallel(
        self,
        total_rollouts: int,
        batch_size: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        per_worker = self._split_rollouts(total_rollouts)
        tasks = {}
        for idx, count in enumerate(per_worker):
            if count <= 0:
                continue
            task = self.workers[idx].generate_rollouts.remote(
                num_rollouts=count,
                batch_size=batch_size,
                verbose=verbose,
            )
            tasks[task] = idx

        collected: List[Trajectory] = []
        per_worker_timings: Dict[str, float] = {}
        start = time.perf_counter()

        while tasks:
            ready_refs, _ = self.ray.wait(list(tasks.keys()), num_returns=1, timeout=600.0)
            if not ready_refs:
                raise TimeoutError("Timed out waiting for rollout workers.")

            ref = ready_refs[0]
            worker_idx = tasks.pop(ref)
            try:
                result = self.ray.get(ref)
                collected.extend(result["trajectories"])
                per_worker_timings[f"worker_{worker_idx}"] = float(result["elapsed_seconds"])
            except Exception as exc:
                logger.error("Worker %d failed during rollout generation: %s", worker_idx, exc)
                self._restart_failed_worker(worker_idx)
                retry_ref = self.workers[worker_idx].generate_rollouts.remote(
                    num_rollouts=per_worker[worker_idx],
                    batch_size=batch_size,
                    verbose=verbose,
                )
                tasks[retry_ref] = worker_idx

        elapsed = time.perf_counter() - start
        return {
            "trajectories": collected,
            "elapsed_seconds": elapsed,
            "per_worker_timings": per_worker_timings,
        }

    def sync_weights(
        self,
        policy_model: torch.nn.Module,
        value_model: torch.nn.Module,
    ) -> Dict[str, Any]:
        policy_state = _extract_trainable_state(policy_model)
        value_head_state = {
            name: tensor.detach().cpu()
            for name, tensor in value_model.value_head.state_dict().items()
        }
        policy_bytes = _serialize_state(policy_state)
        value_bytes = _serialize_state(value_head_state)

        futures = [
            worker.update_trainable_weights.remote(policy_bytes, value_bytes)
            for worker in self.workers
        ]
        results = self.ray.get(futures)
        return {
            "workers_synced": len(results),
            "policy_trainable_tensors": len(policy_state),
            "value_tensors": len(value_head_state),
        }

    def shutdown(self) -> None:
        for worker in self.workers:
            try:
                self.ray.get(worker.shutdown.remote())
            except Exception:
                pass

