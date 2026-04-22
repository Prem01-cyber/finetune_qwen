"""
Distributed PPO training with Ray rollout workers and optional async prefetching.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
import wandb
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.eval_sft_inference import evaluate_gsm8k
from src.rl.async_rollout_buffer import AsyncRolloutBuffer
from src.rl.distributed_coordinator import DistributedRolloutCoordinator, WorkerRuntimeConfig
from src.rl.ppo_trainer import PPOTrainer
from src.rl.rollout_buffer import RolloutBuffer
from src.rl.value_network import ValueHead


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DistributedTrainingConfig:
    base_model = "checkpoints/dual_task_v1"
    output_dir = "checkpoints/ppo_training_distributed"
    curriculum_checkpoint_dir = "checkpoints/ppo_training_distributed/curriculum"
    eval_data_path = "data/sft/dual_task_val.jsonl"
    gsm8k_reference_data = "data/sft/gsm8k_sft.jsonl"

    num_iterations = 100
    num_rollouts_per_iter = 100
    rollout_batch_size = 8
    learning_rate = 1e-6
    ppo_epochs = 3
    batch_size = 32
    clip_range = 0.3
    clip_range_vf = 0.25
    vf_coef = 0.5
    ent_coef = 0.02
    max_grad_norm = 0.5
    target_kl = 0.15
    gamma = 1.0
    gae_lambda = 0.95

    max_question_tokens = 200
    max_solution_tokens = 500
    temperature = 0.7
    top_p = 0.9
    consensus_temperature = 0.5

    ray_address = "auto"
    num_workers = 8
    use_gpu_workers = True
    use_vllm_workers = True
    vllm_tensor_parallel_size = 1
    sync_every = 1

    use_async_buffer = True
    async_queue_size = 300
    async_refill_chunk_size = 100
    async_warmup_size = 100

    eval_every = 10
    save_every = 1
    use_wandb = True
    wandb_project = "math-ppo-curriculum"
    wandb_run_name = None


def load_reference_questions(path: str) -> List[str]:
    questions: List[str] = []
    file_path = Path(path)
    if not file_path.exists():
        logger.warning("Reference data %s not found; using empty reference set", path)
        return questions

    with file_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = record.get("messages", [])
            for msg in messages:
                if msg.get("role") != "user":
                    continue
                content = str(msg.get("content", ""))
                if "Problem:" in content:
                    questions.append(content.split("Problem:", 1)[1].strip())
                else:
                    questions.append(content.strip())
                break
    return questions


def initialize_models(base_model: str):
    model_path = Path(base_model)
    is_adapter = (model_path / "adapter_config.json").exists()

    if is_adapter:
        logger.info("Detected LoRA adapter at: %s", base_model)
        meta_file = model_path / "pipeline_meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            base_model_name = meta.get("base_model", "Qwen/Qwen2.5-Math-1.5B-Instruct")
        else:
            base_model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_lm = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        policy = PeftModel.from_pretrained(base_lm, base_model).merge_and_unload()
        value = ValueHead(base_model_name).to(policy.device)
    else:
        logger.info("Loading full model: %s", base_model)
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        policy = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        value = ValueHead(base_model).to(policy.device)

    logger.info("Trainer model device: %s", policy.device)
    return policy, value, tokenizer


def evaluate_policy(policy, tokenizer, eval_data_path: str) -> Dict[str, float]:
    results = evaluate_gsm8k(
        model=policy,
        tokenizer=tokenizer,
        data_path=eval_data_path,
        max_samples=500,
    )
    logger.info(
        "GSM8K Accuracy: %.2f%% (%d/%d)",
        results["accuracy"] * 100.0,
        results["correct"],
        results["total"],
    )
    return results


def save_iteration_metrics(output_dir: str, iteration: int, metrics: Dict[str, object]) -> None:
    iter_dir = Path(output_dir) / f"iteration_{iteration:04d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    (iter_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed PPO training with Ray workers")
    parser.add_argument("--base-model", type=str, default=DistributedTrainingConfig.base_model)
    parser.add_argument("--output-dir", type=str, default=DistributedTrainingConfig.output_dir)
    parser.add_argument("--num-iterations", type=int, default=DistributedTrainingConfig.num_iterations)
    parser.add_argument("--rollouts-per-iter", type=int, default=DistributedTrainingConfig.num_rollouts_per_iter)
    parser.add_argument("--rollout-batch-size", type=int, default=DistributedTrainingConfig.rollout_batch_size)
    parser.add_argument("--ray-address", type=str, default=DistributedTrainingConfig.ray_address)
    parser.add_argument("--num-workers", type=int, default=DistributedTrainingConfig.num_workers)
    parser.add_argument("--sync-every", type=int, default=DistributedTrainingConfig.sync_every)
    parser.add_argument("--eval-data-path", type=str, default=DistributedTrainingConfig.eval_data_path)
    parser.add_argument("--gsm8k-reference-data", type=str, default=DistributedTrainingConfig.gsm8k_reference_data)
    parser.add_argument("--no-async-buffer", action="store_true")
    parser.add_argument("--async-queue-size", type=int, default=DistributedTrainingConfig.async_queue_size)
    parser.add_argument("--async-refill-chunk-size", type=int, default=DistributedTrainingConfig.async_refill_chunk_size)
    parser.add_argument("--async-warmup-size", type=int, default=DistributedTrainingConfig.async_warmup_size)
    parser.add_argument("--no-vllm-workers", action="store_true")
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=DistributedTrainingConfig.vllm_tensor_parallel_size)
    parser.add_argument("--no-gpu-workers", action="store_true")
    parser.add_argument("--skip-initial-eval", action="store_true")
    parser.add_argument("--eval-every", type=int, default=DistributedTrainingConfig.eval_every)
    parser.add_argument("--save-every", type=int, default=DistributedTrainingConfig.save_every)
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DistributedTrainingConfig()
    cfg.base_model = args.base_model
    cfg.output_dir = args.output_dir
    cfg.curriculum_checkpoint_dir = str(Path(args.output_dir) / "curriculum")
    cfg.num_iterations = args.num_iterations
    cfg.num_rollouts_per_iter = args.rollouts_per_iter
    cfg.rollout_batch_size = max(1, int(args.rollout_batch_size))
    cfg.ray_address = args.ray_address
    cfg.num_workers = max(1, int(args.num_workers))
    cfg.sync_every = max(1, int(args.sync_every))
    cfg.eval_data_path = args.eval_data_path
    cfg.gsm8k_reference_data = args.gsm8k_reference_data
    cfg.use_async_buffer = not args.no_async_buffer
    cfg.async_queue_size = max(1, int(args.async_queue_size))
    cfg.async_refill_chunk_size = max(1, int(args.async_refill_chunk_size))
    cfg.async_warmup_size = max(1, int(args.async_warmup_size))
    cfg.use_vllm_workers = not args.no_vllm_workers
    cfg.vllm_tensor_parallel_size = max(1, int(args.vllm_tensor_parallel_size))
    cfg.use_gpu_workers = not args.no_gpu_workers
    cfg.eval_every = max(1, int(args.eval_every))
    cfg.save_every = max(1, int(args.save_every))
    cfg.use_wandb = not args.no_wandb

    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name or f"ppo_distributed_{datetime.now():%Y%m%d_%H%M%S}",
            config=vars(cfg),
        )

    policy, value, tokenizer = initialize_models(cfg.base_model)
    reference_questions = load_reference_questions(cfg.gsm8k_reference_data)
    trainer = PPOTrainer(
        policy_model=policy,
        value_model=value,
        tokenizer=tokenizer,
        learning_rate=cfg.learning_rate,
        ppo_epochs=cfg.ppo_epochs,
        batch_size=cfg.batch_size,
        clip_range=cfg.clip_range,
        clip_range_vf=cfg.clip_range_vf,
        vf_coef=cfg.vf_coef,
        ent_coef=cfg.ent_coef,
        max_grad_norm=cfg.max_grad_norm,
        target_kl=cfg.target_kl,
    )

    worker_cfg = WorkerRuntimeConfig(
        base_model=cfg.base_model,
        reference_questions=reference_questions,
        curriculum_checkpoint_dir=cfg.curriculum_checkpoint_dir,
        max_question_tokens=cfg.max_question_tokens,
        max_solution_tokens=cfg.max_solution_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        consensus_temperature=cfg.consensus_temperature,
        use_vllm=cfg.use_vllm_workers,
        vllm_batch_size=cfg.rollout_batch_size,
        vllm_tensor_parallel_size=cfg.vllm_tensor_parallel_size,
    )
    coordinator = DistributedRolloutCoordinator(
        ray_address=cfg.ray_address,
        num_workers=cfg.num_workers,
        worker_config=worker_cfg,
        use_gpu_workers=cfg.use_gpu_workers,
    )
    logger.info("Ray workers healthy: %s", coordinator.health_check())

    async_buffer = None
    if cfg.use_async_buffer:
        async_buffer = AsyncRolloutBuffer(
            coordinator=coordinator,
            queue_max_size=cfg.async_queue_size,
            refill_chunk_size=cfg.async_refill_chunk_size,
            worker_batch_size=cfg.rollout_batch_size,
        )
        async_buffer.start()
        async_buffer.warmup(minimum_ready=min(cfg.async_warmup_size, cfg.async_queue_size))

    if args.skip_initial_eval:
        initial_eval = {"accuracy": 0.0}
    else:
        initial_eval = evaluate_policy(policy, tokenizer, cfg.eval_data_path)

    best_accuracy = float(initial_eval["accuracy"])
    try:
        for iteration in range(1, cfg.num_iterations + 1):
            iter_start = time.perf_counter()
            rollout_start = time.perf_counter()
            if async_buffer is not None:
                trajectories = async_buffer.pop_batch(size=cfg.num_rollouts_per_iter)
                rollout_meta = {
                    "mode": "async",
                    "elapsed_seconds": time.perf_counter() - rollout_start,
                    "buffer": async_buffer.stats(),
                    "per_worker_timings": {},
                }
            else:
                rollout_result = coordinator.collect_rollouts_parallel(
                    total_rollouts=cfg.num_rollouts_per_iter,
                    batch_size=cfg.rollout_batch_size,
                    verbose=False,
                )
                trajectories = rollout_result["trajectories"]
                rollout_meta = {
                    "mode": "sync",
                    "elapsed_seconds": rollout_result["elapsed_seconds"],
                    "buffer": {},
                    "per_worker_timings": rollout_result["per_worker_timings"],
                }

            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
            rollout_buffer = RolloutBuffer(
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
                pad_token_id=int(pad_id),
            )
            for trajectory in trajectories:
                rollout_buffer.add_trajectory(trajectory)

            train_start = time.perf_counter()
            training_metrics = trainer.train_step(rollout_buffer)
            train_seconds = time.perf_counter() - train_start

            sync_seconds = 0.0
            sync_metrics: Dict[str, object] = {"workers_synced": 0}
            if iteration % cfg.sync_every == 0:
                sync_start = time.perf_counter()
                sync_metrics = coordinator.sync_weights(policy_model=trainer.policy, value_model=trainer.value)
                sync_seconds = time.perf_counter() - sync_start

            if iteration % cfg.eval_every == 0:
                eval_results = evaluate_policy(policy, tokenizer, cfg.eval_data_path)
                best_accuracy = max(best_accuracy, float(eval_results["accuracy"]))
            else:
                eval_results = {}

            iter_seconds = time.perf_counter() - iter_start
            metrics = {
                "iteration": iteration,
                "timing": {
                    "iteration_seconds": iter_seconds,
                    "rollout_seconds": rollout_meta["elapsed_seconds"],
                    "train_seconds": train_seconds,
                    "sync_seconds": sync_seconds,
                },
                "rollout": rollout_meta,
                "training": training_metrics,
                "sync": sync_metrics,
                "eval": eval_results,
                "buffer": rollout_buffer.get_stats(),
            }
            save_iteration_metrics(cfg.output_dir, iteration, metrics)

            logger.info(
                "Iter %d | rollouts %.1fs | train %.1fs | sync %.1fs | total %.1fs",
                iteration,
                float(rollout_meta["elapsed_seconds"]),
                train_seconds,
                sync_seconds,
                iter_seconds,
            )

            if cfg.use_wandb:
                wandb_payload = {
                    "iteration": iteration,
                    "timing/iteration_seconds": iter_seconds,
                    "timing/rollout_seconds": float(rollout_meta["elapsed_seconds"]),
                    "timing/train_seconds": train_seconds,
                    "timing/sync_seconds": sync_seconds,
                    "training/policy_loss": training_metrics["policy_loss"],
                    "training/value_loss": training_metrics["value_loss"],
                    "training/entropy": training_metrics["entropy"],
                    "training/approx_kl": training_metrics["approx_kl"],
                    "training/clip_fraction": training_metrics["clip_fraction"],
                    "buffer/mean_reward": rollout_buffer.get_stats()["mean_episode_reward"],
                    "buffer/mean_episode_length": rollout_buffer.get_stats()["mean_episode_length"],
                    "sync/workers_synced": float(sync_metrics.get("workers_synced", 0)),
                }
                if async_buffer is not None:
                    async_stats = async_buffer.stats()
                    wandb_payload.update(
                        {
                            "async/queue_size": async_stats["queue_size"],
                            "async/total_generated": async_stats["total_generated"],
                            "async/total_consumed": async_stats["total_consumed"],
                            "async/worker_failures": async_stats["worker_failures"],
                        }
                    )
                if eval_results:
                    wandb_payload["eval/accuracy"] = eval_results["accuracy"]
                wandb.log(wandb_payload)

            if iteration % cfg.save_every == 0:
                checkpoint_path = Path(cfg.output_dir) / f"iteration_{iteration:04d}" / "checkpoint.pt"
                trainer.save_checkpoint(str(checkpoint_path))

    finally:
        if async_buffer is not None:
            async_buffer.stop()
        coordinator.shutdown()
        if cfg.use_wandb:
            wandb.finish()

    final_eval = evaluate_policy(policy, tokenizer, cfg.eval_data_path)
    logger.info(
        "Training complete. Initial acc: %.2f%% | Final acc: %.2f%% | Delta: %.2f%% | Best: %.2f%%",
        initial_eval["accuracy"] * 100.0,
        final_eval["accuracy"] * 100.0,
        (final_eval["accuracy"] - initial_eval["accuracy"]) * 100.0,
        best_accuracy * 100.0,
    )


if __name__ == "__main__":
    main()

