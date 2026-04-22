"""
Asynchronous rollout prefetch queue for non-blocking PPO training.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.rl.distributed_coordinator import DistributedRolloutCoordinator
from src.rl.mdp_components import Trajectory

logger = logging.getLogger(__name__)


@dataclass
class AsyncBufferStats:
    queue_size: int
    max_size: int
    total_generated: int
    total_consumed: int
    worker_failures: int
    last_fill_seconds: float


class AsyncRolloutBuffer:
    """Background rollout prefetcher backed by a thread-safe queue."""

    def __init__(
        self,
        coordinator: DistributedRolloutCoordinator,
        queue_max_size: int = 400,
        refill_chunk_size: int = 100,
        worker_batch_size: Optional[int] = None,
    ) -> None:
        self.coordinator = coordinator
        self.queue_max_size = max(1, int(queue_max_size))
        self.refill_chunk_size = max(1, int(refill_chunk_size))
        self.worker_batch_size = worker_batch_size

        self._queue: queue.Queue[Trajectory] = queue.Queue(maxsize=self.queue_max_size)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        self._total_generated = 0
        self._total_consumed = 0
        self._worker_failures = 0
        self._last_fill_seconds = 0.0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._fill_loop, daemon=True, name="async-rollout-fill")
        self._thread.start()
        logger.info("Started async rollout fill thread")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        logger.info("Stopped async rollout fill thread")

    def _fill_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                free_slots = self.queue_max_size - self._queue.qsize()
                if free_slots < self.refill_chunk_size:
                    time.sleep(0.5)
                    continue

                request_count = min(self.refill_chunk_size, free_slots)
                started = time.perf_counter()
                result = self.coordinator.collect_rollouts_parallel(
                    total_rollouts=request_count,
                    batch_size=self.worker_batch_size,
                    verbose=False,
                )
                elapsed = time.perf_counter() - started
                trajectories: List[Trajectory] = result["trajectories"]

                for trajectory in trajectories:
                    if self._stop_event.is_set():
                        break
                    self._queue.put(trajectory, timeout=1.0)

                with self._lock:
                    self._total_generated += len(trajectories)
                    self._last_fill_seconds = elapsed
            except Exception as exc:
                logger.exception("Async rollout fill failed: %s", exc)
                with self._lock:
                    self._worker_failures += 1
                time.sleep(1.0)

    def pop_batch(self, size: int, timeout_s: float = 120.0) -> List[Trajectory]:
        size = max(1, int(size))
        batch: List[Trajectory] = []
        deadline = time.time() + timeout_s

        while len(batch) < size:
            remaining = max(0.1, deadline - time.time())
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for rollout batch of size {size}")
            trajectory = self._queue.get(timeout=remaining)
            batch.append(trajectory)

        with self._lock:
            self._total_consumed += len(batch)
        return batch

    def warmup(self, minimum_ready: int, timeout_s: float = 600.0) -> None:
        minimum_ready = max(1, int(minimum_ready))
        deadline = time.time() + timeout_s
        while self._queue.qsize() < minimum_ready:
            if time.time() > deadline:
                raise TimeoutError(
                    f"Warmup timeout while waiting for {minimum_ready} trajectories; "
                    f"current={self._queue.qsize()}"
                )
            time.sleep(0.5)

    def stats(self) -> Dict[str, float]:
        with self._lock:
            summary = AsyncBufferStats(
                queue_size=self._queue.qsize(),
                max_size=self.queue_max_size,
                total_generated=self._total_generated,
                total_consumed=self._total_consumed,
                worker_failures=self._worker_failures,
                last_fill_seconds=self._last_fill_seconds,
            )
        return {
            "queue_size": float(summary.queue_size),
            "max_size": float(summary.max_size),
            "total_generated": float(summary.total_generated),
            "total_consumed": float(summary.total_consumed),
            "worker_failures": float(summary.worker_failures),
            "last_fill_seconds": float(summary.last_fill_seconds),
        }

