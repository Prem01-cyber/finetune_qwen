"""
Smoke test for Ray distributed rollout generation.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_ppo_training_distributed import load_reference_questions
from src.rl.distributed_coordinator import DistributedRolloutCoordinator, WorkerRuntimeConfig


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test distributed rollout workers")
    parser.add_argument("--base-model", type=str, default="checkpoints/dual_task_v1")
    parser.add_argument("--ray-address", type=str, default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--rollout-batch-size", type=int, default=4)
    parser.add_argument("--reference-data", type=str, default="data/sft/gsm8k_sft.jsonl")
    parser.add_argument("--no-vllm-workers", action="store_true")
    parser.add_argument("--no-gpu-workers", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    worker_cfg = WorkerRuntimeConfig(
        base_model=args.base_model,
        reference_questions=load_reference_questions(args.reference_data),
        curriculum_checkpoint_dir="checkpoints/ppo_training_distributed/curriculum_smoke",
        max_question_tokens=120,
        max_solution_tokens=256,
        temperature=0.7,
        top_p=0.9,
        consensus_temperature=0.5,
        use_vllm=not args.no_vllm_workers,
        vllm_batch_size=max(1, int(args.rollout_batch_size)),
        vllm_tensor_parallel_size=1,
    )

    coordinator = DistributedRolloutCoordinator(
        ray_address=args.ray_address,
        num_workers=max(1, int(args.num_workers)),
        worker_config=worker_cfg,
        use_gpu_workers=not args.no_gpu_workers,
    )
    logger.info("Worker health: %s", coordinator.health_check())

    start = time.perf_counter()
    result = coordinator.collect_rollouts_parallel(
        total_rollouts=max(1, int(args.rollouts)),
        batch_size=max(1, int(args.rollout_batch_size)),
        verbose=False,
    )
    elapsed = time.perf_counter() - start
    logger.info(
        "Smoke test successful: trajectories=%d elapsed=%.2fs worker_timings=%s",
        len(result["trajectories"]),
        elapsed,
        result["per_worker_timings"],
    )
    coordinator.shutdown()


if __name__ == "__main__":
    main()

