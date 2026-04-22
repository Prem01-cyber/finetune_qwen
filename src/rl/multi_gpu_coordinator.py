"""
Coordinator for multiprocessing multi-GPU rollout workers.
"""

from __future__ import annotations

import io
import logging
import multiprocessing as mp
import queue
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import torch

from src.rl.mdp_components import Trajectory
from src.rl.multi_gpu_rollout_worker import WorkerRuntimeConfig, rollout_worker_entrypoint

logger = logging.getLogger(__name__)


@dataclass
class WorkerHandle:
    worker_id: int
    gpu_id: int
    process: Any
    task_queue: Any
    result_queue: Any


def _serialize_state(payload: Dict[str, torch.Tensor]) -> bytes:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def _extract_trainable_state(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu()
        for name, param in module.named_parameters()
        if param.requires_grad
    }


class MultiGPUCoordinator:
    """Coordinates rollout generation over N local GPU workers."""

    def __init__(
        self,
        worker_config: WorkerRuntimeConfig,
        num_gpus: Optional[int] = None,
        worker_timeout_seconds: float = 900.0,
    ) -> None:
        self.worker_config = worker_config
        self.worker_timeout_seconds = float(worker_timeout_seconds)
        self.ctx = mp.get_context("spawn")
        self._request_counter = 0

        available_gpus = int(torch.cuda.device_count())
        if available_gpus <= 0:
            raise RuntimeError("No CUDA GPUs detected for multi-GPU rollout generation.")

        desired_gpus = int(num_gpus) if num_gpus is not None else available_gpus
        self.num_gpus = max(1, min(desired_gpus, available_gpus))
        self.gpu_ids = list(range(self.num_gpus))
        if desired_gpus > available_gpus:
            logger.warning(
                "Requested %d rollout GPUs but only %d available. Using %d.",
                desired_gpus,
                available_gpus,
                self.num_gpus,
            )

        self.workers: List[WorkerHandle] = []
        for worker_id, gpu_id in enumerate(self.gpu_ids):
            self.workers.append(self._spawn_worker(worker_id=worker_id, gpu_id=gpu_id))

    def _next_request_id(self) -> int:
        self._request_counter += 1
        return self._request_counter

    def _spawn_worker(self, worker_id: int, gpu_id: int) -> WorkerHandle:
        task_queue = self.ctx.Queue()
        result_queue = self.ctx.Queue()
        process = self.ctx.Process(
            target=rollout_worker_entrypoint,
            args=(worker_id, gpu_id, self.worker_config, task_queue, result_queue),
            daemon=True,
        )
        process.start()
        return WorkerHandle(
            worker_id=worker_id,
            gpu_id=gpu_id,
            process=process,
            task_queue=task_queue,
            result_queue=result_queue,
        )

    def _restart_worker(self, worker_id: int) -> None:
        handle = self.workers[worker_id]
        try:
            if handle.process.is_alive():
                handle.process.terminate()
            handle.process.join(timeout=2.0)
        except Exception:
            pass
        self.workers[worker_id] = self._spawn_worker(worker_id=worker_id, gpu_id=handle.gpu_id)
        logger.warning("Restarted rollout worker %d on GPU %d", worker_id, handle.gpu_id)

    def _round_robin_split(self, total_rollouts: int) -> List[int]:
        base = total_rollouts // self.num_gpus
        remainder = total_rollouts % self.num_gpus
        split = [base] * self.num_gpus
        for idx in range(remainder):
            split[idx] += 1
        return split

    def _request_worker(
        self,
        worker_id: int,
        payload: Dict[str, Any],
        timeout_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        handle = self.workers[worker_id]
        request_id = self._next_request_id()
        handle.task_queue.put({**payload, "request_id": request_id})
        timeout = float(timeout_seconds if timeout_seconds is not None else self.worker_timeout_seconds)
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                response = handle.result_queue.get(timeout=0.5)
            except queue.Empty:
                if not handle.process.is_alive():
                    raise RuntimeError(f"Worker {worker_id} process exited unexpectedly.")
                continue
            if int(response.get("request_id", -1)) != request_id:
                continue
            if not bool(response.get("ok", False)):
                raise RuntimeError(
                    f"Worker {worker_id} error: {response.get('error')}\n{response.get('traceback', '')}"
                )
            return response["result"]
        raise TimeoutError(f"Timed out waiting for worker {worker_id} response.")

    def health_check(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for worker_id in range(self.num_gpus):
            try:
                result = self._request_worker(worker_id=worker_id, payload={"cmd": "ping"}, timeout_seconds=30.0)
            except Exception as exc:
                result = {"worker_id": worker_id, "error": str(exc)}
            results.append(result)
        return results

    def collect_rollouts_parallel(
        self,
        total_rollouts: int,
        batch_size: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        target_rollouts = max(0, int(total_rollouts))
        if target_rollouts == 0:
            return {
                "trajectories": [],
                "elapsed_seconds": 0.0,
                "per_worker_timings": {},
                "per_worker_counts": {},
            }

        split = self._round_robin_split(target_rollouts)
        start = time.perf_counter()
        per_worker_timings: Dict[str, float] = {}
        per_worker_counts: Dict[str, int] = {}
        collected: List[Trajectory] = []

        for worker_id, count in enumerate(split):
            if count <= 0:
                continue
            worker_start = time.perf_counter()
            try:
                result = self._request_worker(
                    worker_id=worker_id,
                    payload={
                        "cmd": "generate",
                        "num_rollouts": count,
                        "batch_size": batch_size,
                        "verbose": verbose,
                    },
                )
            except Exception as exc:
                logger.error("Worker %d failed rollout generation: %s", worker_id, exc)
                self._restart_worker(worker_id=worker_id)
                result = self._request_worker(
                    worker_id=worker_id,
                    payload={
                        "cmd": "generate",
                        "num_rollouts": count,
                        "batch_size": batch_size,
                        "verbose": verbose,
                    },
                )
            per_worker_timings[f"worker_{worker_id}"] = float(time.perf_counter() - worker_start)
            per_worker_counts[f"worker_{worker_id}"] = int(result.get("num_rollouts", 0))
            collected.extend(result.get("trajectories", []))

        return {
            "trajectories": collected,
            "elapsed_seconds": float(time.perf_counter() - start),
            "per_worker_timings": per_worker_timings,
            "per_worker_counts": per_worker_counts,
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

        synced = 0
        for worker_id in range(self.num_gpus):
            try:
                self._request_worker(
                    worker_id=worker_id,
                    payload={
                        "cmd": "sync",
                        "policy_trainable_bytes": policy_bytes,
                        "value_head_bytes": value_bytes,
                    },
                    timeout_seconds=max(120.0, self.worker_timeout_seconds),
                )
                synced += 1
            except Exception as exc:
                logger.error("Failed syncing worker %d: %s", worker_id, exc)
                self._restart_worker(worker_id=worker_id)
        return {
            "workers_synced": synced,
            "policy_trainable_tensors": len(policy_state),
            "value_tensors": len(value_head_state),
        }

    def reduce_batch_size(self, current_batch_size: int) -> int:
        return max(1, int(current_batch_size * 0.75))

    def shutdown(self) -> None:
        for worker_id, handle in enumerate(self.workers):
            try:
                self._request_worker(
                    worker_id=worker_id,
                    payload={"cmd": "shutdown"},
                    timeout_seconds=30.0,
                )
            except Exception:
                pass
            try:
                if handle.process.is_alive():
                    handle.process.terminate()
                handle.process.join(timeout=2.0)
            except Exception:
                pass

