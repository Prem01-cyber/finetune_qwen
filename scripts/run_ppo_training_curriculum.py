"""
PPO training with curriculum-guided dual-task rewards.

Single entry point for both single-GPU and multi-GPU (DeepSpeed ZeRO-3) runs.

    # Single GPU
    python scripts/run_ppo_training_curriculum.py --base-model checkpoints/dual_task_v1

    # Multi-GPU (N GPUs)
    deepspeed --num_gpus=N scripts/run_ppo_training_curriculum.py \\
        --use-deepspeed --base-model checkpoints/dual_task_v1

Under DeepSpeed, each rank loads the merged weights, ZeRO-3 shards them across
ranks, and rollouts are collected *data-parallel* with a temporary
``GatheredParameters`` context.  PPO updates then shard mini-batches across
ranks and DeepSpeed's built-in gradient all-reduce stitches them back together.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.eval_sft_inference import evaluate_gsm8k
from src.rl.checkpoint_manager import CheckpointManager
from src.rl.deepspeed_utils import (
    all_gather_objects,
    barrier,
    broadcast_object,
    current_cuda_device,
    gather_params_for_generation,
    get_local_rank,
    get_rank,
    get_world_size,
    is_main_process,
    my_share,
)
from src.rl.math_environment_curriculum import CurriculumMathEnvironment
from src.rl.ppo_trainer import PPOTrainer
from src.rl.ppo_trainer_deepspeed import PPOTrainerDeepSpeed
from src.rl.rollout_buffer import RolloutBuffer
from src.rl.training_monitor import TrainingMonitor
from src.rl.value_network import ValueHead
from src.utils.csv_logger import CSVLogger


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TeeStream:
    """Write stream output to both terminal and a log file."""

    def __init__(self, primary, secondary):
        self.primary = primary
        self.secondary = secondary

    def write(self, data: str) -> int:
        self.primary.write(data)
        self.secondary.write(data)
        return len(data)

    def flush(self) -> None:
        self.primary.flush()
        self.secondary.flush()

    def isatty(self) -> bool:
        return getattr(self.primary, "isatty", lambda: False)()

    def fileno(self) -> int:
        return self.primary.fileno()


class CurriculumTrainingConfig:
    base_model = "checkpoints/dual_task_v1"
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

    num_rollouts_per_iter = 100
    max_question_tokens = 200
    max_solution_tokens = 500
    temperature = 0.7
    top_p = 0.9
    consensus_temperature = 0.5

    num_iterations = 10
    eval_every = 5
    save_every = 1
    use_torch_compile = True

    output_dir = "checkpoints/ppo_training_curriculum"
    curriculum_checkpoint_dir = "checkpoints/ppo_training_curriculum/curriculum"
    eval_data_path = "data/sft/dual_task_val.jsonl"
    gsm8k_reference_data = "data/sft/gsm8k_sft.jsonl"

    disk_warning_gb = 5.0
    checkpoint_keep_last = 2
    checkpoint_keep_every = 100
    compress_old_logs = True

    log_dir = "logs"
    run_name = None


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


def log_gpu_memory(stage: str) -> None:
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
        total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
        logger.info(
            "[%s] GPU %d: %.2fGB allocated, %.2fGB reserved, %.2fGB total (%.1f%% used)",
            stage, i, allocated, reserved, total, allocated / total * 100,
        )


def _ensure_peft_tensor_parallel_shim() -> None:
    """
    PEFT <= 0.12 unconditionally imports
    ``transformers.integrations.tensor_parallel`` on attribute lookup.  Older
    transformers versions don't ship that module and the import crashes under
    torchrun/deepspeed.  Install a harmless stub to unblock the merge path.
    """
    import sys as _sys
    import types

    if "transformers.integrations.tensor_parallel" not in _sys.modules:
        _sys.modules["transformers.integrations.tensor_parallel"] = types.ModuleType(
            "tensor_parallel"
        )


def initialize_models(
    config: CurriculumTrainingConfig,
    use_deepspeed: bool = False,
):
    """
    Load the policy, value network and tokenizer.

    Single-GPU: models go straight to cuda:0.
    DeepSpeed:  models stay on CPU until ``deepspeed.initialize`` shards them
                across ranks.  Every rank has to perform this load (ZeRO-3
                requires initialised weights on every rank to partition them).
    """
    model_path = Path(config.base_model)
    is_adapter = (model_path / "adapter_config.json").exists()

    if is_adapter:
        meta_file = model_path / "pipeline_meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            base_model_name = meta.get("base_model", "Qwen/Qwen2.5-Math-1.5B-Instruct")
        else:
            base_model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    else:
        base_model_name = config.base_model

    log_gpu_memory("Before model loading")

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None and is_adapter:
        logger.info("Chat template not found in adapter, loading from base model")
        base_tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, trust_remote_code=True
        )
        if base_tokenizer.chat_template is not None:
            tokenizer.chat_template = base_tokenizer.chat_template

    _ensure_peft_tensor_parallel_shim()

    # --- Policy --------------------------------------------------------
    if is_adapter:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": "cpu"},
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        policy = PeftModel.from_pretrained(base_model, config.base_model).merge_and_unload()
    else:
        policy = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.bfloat16,
            device_map={"": "cpu"},
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

    if not use_deepspeed:
        policy = policy.to("cuda:0")
        log_gpu_memory("After policy loaded on GPU 0")
    else:
        log_gpu_memory("After policy loaded on CPU (pre-DeepSpeed)")

    # --- Value network -------------------------------------------------
    value = ValueHead(base_model_name, model_device_map={"": "cpu"})
    if not use_deepspeed:
        value.backbone = value.backbone.to("cuda:0")
        value.value_head = value.value_head.to("cuda:0")
    log_gpu_memory("After ValueHead loaded")

    # torch.compile conflicts with DeepSpeed's ZeRO-3 pre/post-forward hooks,
    # so only enable it for the single-process path.
    if not use_deepspeed and config.use_torch_compile:
        try:
            logger.info("Compiling policy with torch.compile (may take 2-3 min)")
            policy = torch.compile(policy, mode="reduce-overhead")
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("torch.compile failed: %s. Continuing without.", exc)

    return policy, value, tokenizer


def evaluate_policy(policy, tokenizer, eval_data_path: str) -> Dict[str, float]:
    logger.info("Evaluating policy on GSM8K test set...")
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


def aggregate_curriculum_metrics(trajectories: List) -> Dict[str, object]:
    topic_counts: Dict[str, int] = {}
    topic_successes: Dict[str, int] = {}
    topic_difficulty: Dict[str, List[float]] = {}

    topic_match_scores: List[float] = []
    difficulty_match_scores: List[float] = []
    clarity_scores: List[float] = []
    solvability_scores: List[float] = []
    novelty_scores: List[float] = []
    question_rewards: List[float] = []
    solution_rewards: List[float] = []
    combined_rewards: List[float] = []
    pre_expert_rewards: List[float] = []
    expert_modifiers: List[float] = []
    expert_phase_counts: Dict[str, int] = {}
    source_counts: Dict[str, int] = {}
    replay_added_count = 0

    for trajectory in trajectories:
        meta = trajectory.metadata
        topic = str(meta["target_topic"])
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
        topic_successes[topic] = topic_successes.get(topic, 0) + int(
            meta["consensus_achieved"] and meta["primary_matches_majority"]
        )
        topic_difficulty.setdefault(topic, []).append(
            float(meta["estimated_difficulty"])
        )

        topic_match_scores.append(float(meta["topic_match_score"]))
        difficulty_match_scores.append(
            1.0 - abs(float(meta["estimated_difficulty"]) - float(meta["target_difficulty"]))
        )
        clarity_scores.append(float(meta["clarity_score"]))
        solvability_scores.append(1.0 if bool(meta["sympy_verified"]) else 0.0)
        novelty_scores.append(float(meta["novelty_scores"]["combined"]))
        question_rewards.append(float(meta["question_reward"]))
        solution_rewards.append(float(meta["solution_reward"]))
        combined_rewards.append(float(meta["combined_reward"]))
        pre_expert_rewards.append(
            float(meta.get("pre_expert_reward", meta["combined_reward"]))
        )
        expert_modifiers.append(float(meta.get("expert_reward_modifier", 0.0)))
        phase = str(meta.get("expert_phase", "unknown"))
        expert_phase_counts[phase] = expert_phase_counts.get(phase, 0) + 1
        source = str(meta.get("rollout_source", "fresh"))
        source_counts[source] = source_counts.get(source, 0) + 1
        replay_added_count += int(bool(meta.get("replay_added", False)))

    per_topic_success = {
        topic: (topic_successes.get(topic, 0) / max(1, count))
        for topic, count in topic_counts.items()
    }
    per_topic_difficulty = {
        topic: float(sum(values) / max(1, len(values)))
        for topic, values in topic_difficulty.items()
    }

    def _mean_reward_for(source: str) -> float:
        vals = [
            float(t.metadata["combined_reward"])
            for t in trajectories
            if str(t.metadata.get("rollout_source", "fresh")) == source
        ]
        return float(sum(vals) / max(1, len(vals)))

    return {
        "topics_in_sweet_spot": len(
            [s for s in per_topic_success.values() if 0.4 <= s <= 0.7]
        ),
        "avg_difficulty": float(
            sum(difficulty_match_scores) / max(1, len(difficulty_match_scores))
        ),
        "topic_diversity": len(topic_counts),
        "per_topic_success": per_topic_success,
        "per_topic_difficulty": per_topic_difficulty,
        "avg_topic_match": float(sum(topic_match_scores) / max(1, len(topic_match_scores))),
        "avg_difficulty_match": float(
            sum(difficulty_match_scores) / max(1, len(difficulty_match_scores))
        ),
        "avg_clarity": float(sum(clarity_scores) / max(1, len(clarity_scores))),
        "avg_solvability": float(sum(solvability_scores) / max(1, len(solvability_scores))),
        "avg_novelty": float(sum(novelty_scores) / max(1, len(novelty_scores))),
        "avg_question_reward": float(sum(question_rewards) / max(1, len(question_rewards))),
        "avg_solution_reward": float(sum(solution_rewards) / max(1, len(solution_rewards))),
        "avg_combined_reward": float(sum(combined_rewards) / max(1, len(combined_rewards))),
        "avg_pre_expert_reward": float(
            sum(pre_expert_rewards) / max(1, len(pre_expert_rewards))
        ),
        "avg_expert_modifier": float(sum(expert_modifiers) / max(1, len(expert_modifiers))),
        "expert_phase_counts": expert_phase_counts,
        "source_counts": source_counts,
        "replay_added_count": replay_added_count,
        "fresh_mean_reward": _mean_reward_for("fresh"),
        "replay_mean_reward": _mean_reward_for("replay"),
    }


def save_iteration_results(
    iteration: int,
    trajectories: List,
    metrics: Dict[str, object],
    config: CurriculumTrainingConfig,
) -> None:
    output_dir = Path(config.output_dir) / f"iteration_{iteration:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "trajectories.jsonl").open("w", encoding="utf-8") as handle:
        for idx, trajectory in enumerate(trajectories):
            payload = {
                "trajectory_id": idx,
                "total_reward": trajectory.total_reward,
                "length": len(trajectory),
                "metadata": trajectory.metadata,
            }
            handle.write(json.dumps(payload) + "\n")

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )


def _move_trajectories_to_cpu(trajectories: List) -> None:
    """
    Detach rollout tensors to CPU before cross-rank pickling.

    all_gather_object serialises the tensors with their original device, so a
    tensor produced on rank 0's ``cuda:0`` is revived on rank 1 as ``cuda:0`` —
    which is a *different* physical GPU than rank 1's own ``cuda:1``.  Moving
    to CPU first lets the PPO trainer do an unambiguous ``.to(self.device)``.
    """
    for traj in trajectories:
        for trans in traj.transitions:
            trans.state.input_ids = trans.state.input_ids.detach().cpu()
            trans.state.attention_mask = trans.state.attention_mask.detach().cpu()
            if trans.next_state is not None:
                trans.next_state.input_ids = trans.next_state.input_ids.detach().cpu()
                trans.next_state.attention_mask = (
                    trans.next_state.attention_mask.detach().cpu()
                )


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO with curriculum-guided dual-task rewards"
    )
    parser.add_argument("--base-model", type=str, default="checkpoints/dual_task_v1")
    parser.add_argument(
        "--output-dir", type=str, default="checkpoints/ppo_training_curriculum"
    )
    parser.add_argument("--num-iterations", type=int, default=10)
    parser.add_argument("--rollouts-per-iter", type=int, default=100)
    parser.add_argument(
        "--eval-data-path", type=str, default="data/sft/dual_task_val.jsonl"
    )
    parser.add_argument(
        "--gsm8k-reference-data", type=str, default="data/sft/gsm8k_sft.jsonl"
    )
    parser.add_argument("--skip-initial-eval", action="store_true")
    parser.add_argument("--disk-warning-gb", type=float, default=5.0)
    parser.add_argument("--checkpoint-keep-last", type=int, default=2)
    parser.add_argument("--checkpoint-keep-every", type=int, default=100)
    parser.add_argument("--no-compress-old-logs", action="store_true")
    parser.add_argument(
        "--run-name", type=str, default=None, help="Optional run name for logging"
    )
    parser.add_argument(
        "--use-deepspeed",
        action="store_true",
        help="Enable DeepSpeed ZeRO-3 multi-GPU training",
    )
    parser.add_argument(
        "--deepspeed-config",
        type=str,
        default="configs/deepspeed_zero3_rl.json",
        help="Path to DeepSpeed config JSON",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1,
        help="Local rank set by the DeepSpeed launcher",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------
    # DeepSpeed launcher hygiene: pin this process to its local GPU
    # *before* any CUDA allocation so deepspeed.initialize() lays params
    # out on the right device.
    # -----------------------------------------------------------------
    if args.use_deepspeed and torch.cuda.is_available():
        local_rank = args.local_rank if args.local_rank >= 0 else get_local_rank()
        torch.cuda.set_device(local_rank)

    config = CurriculumTrainingConfig()
    config.base_model = args.base_model
    config.output_dir = args.output_dir
    config.num_iterations = args.num_iterations
    config.num_rollouts_per_iter = args.rollouts_per_iter
    config.eval_data_path = args.eval_data_path
    config.gsm8k_reference_data = args.gsm8k_reference_data
    config.curriculum_checkpoint_dir = str(Path(args.output_dir) / "curriculum")
    config.run_name = args.run_name
    config.disk_warning_gb = float(args.disk_warning_gb)
    config.checkpoint_keep_last = max(1, int(args.checkpoint_keep_last))
    config.checkpoint_keep_every = max(1, int(args.checkpoint_keep_every))
    config.compress_old_logs = not args.no_compress_old_logs

    # Deterministic but per-rank seeding so every rank samples a different
    # slice of rollouts (true data parallelism) while still being reproducible.
    base_seed = 1234
    random.seed(base_seed + get_rank())
    np.random.seed(base_seed + get_rank())
    torch.manual_seed(base_seed)

    # CSV logger and stdout tee live on rank 0 only.  Other ranks keep their
    # stdout so tracebacks still surface.
    logger_csv = None
    console_log_file = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    console_log_path: Optional[Path] = None
    if is_main_process():
        logger_csv = CSVLogger(
            project="ppo-curriculum",
            run_name=config.run_name or f"curriculum_{datetime.now():%Y%m%d_%H%M%S}",
            log_dir=config.log_dir,
            config=vars(config),
            log_detailed=True,
        )
        console_log_path = Path(logger_csv.log_path) / "console_output.log"
        console_log_file = console_log_path.open("a", encoding="utf-8", buffering=1)
        sys.stdout = TeeStream(original_stdout, console_log_file)
        sys.stderr = TeeStream(original_stderr, console_log_file)
        logger.info("Full console output is being captured at %s", console_log_path)

    try:
        policy, value, tokenizer = initialize_models(
            config, use_deepspeed=args.use_deepspeed
        )
        reference_questions = load_reference_questions(config.gsm8k_reference_data)

        # Pin the env to this rank's local GPU.  Without this, a ZeRO-3
        # partitioned/offloaded policy would make the env put tensors on CPU
        # during generation.
        env_device = (
            current_cuda_device()
            if args.use_deepspeed
            else (next(policy.parameters()).device if torch.cuda.is_available() else torch.device("cpu"))
        )

        math_env = CurriculumMathEnvironment(
            policy_model=policy,
            value_model=value,
            tokenizer=tokenizer,
            reference_questions=reference_questions,
            curriculum_checkpoint_dir=config.curriculum_checkpoint_dir,
            max_question_tokens=config.max_question_tokens,
            max_solution_tokens=config.max_solution_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            consensus_temperature=config.consensus_temperature,
            device=env_device,
        )
        training_monitor = TrainingMonitor(
            output_dir=config.output_dir,
            disk_warning_gb=config.disk_warning_gb,
        )
        checkpoint_manager = CheckpointManager(
            output_dir=config.output_dir,
            keep_last_n=config.checkpoint_keep_last,
            keep_every_n=config.checkpoint_keep_every,
            compress_old_logs=config.compress_old_logs,
        )

        if args.use_deepspeed:
            logger.info(
                "Using DeepSpeed ZeRO-3 PPO trainer (rank=%d, world=%d)",
                get_rank(), get_world_size(),
            )
            ppo_trainer = PPOTrainerDeepSpeed(
                policy_model=policy,
                value_model=value,
                tokenizer=tokenizer,
                learning_rate=config.learning_rate,
                ppo_epochs=config.ppo_epochs,
                batch_size=config.batch_size,
                clip_range=config.clip_range,
                clip_range_vf=config.clip_range_vf,
                vf_coef=config.vf_coef,
                ent_coef=config.ent_coef,
                max_grad_norm=config.max_grad_norm,
                target_kl=config.target_kl,
                ds_config=args.deepspeed_config,
            )
            # DeepSpeed returned the same nn.Module references (engine.module is
            # our policy_model), but point env at the canonical handles kept
            # by the trainer so weight sync is unambiguous.
            policy = ppo_trainer.policy
            value = ppo_trainer.value
            math_env.policy = policy
            math_env.value = value
            math_env.triple_verifier.model = policy
        else:
            logger.info("Using single-GPU PPO trainer")
            ppo_trainer = PPOTrainer(
                policy_model=policy,
                value_model=value,
                tokenizer=tokenizer,
                learning_rate=config.learning_rate,
                ppo_epochs=config.ppo_epochs,
                batch_size=config.batch_size,
                clip_range=config.clip_range,
                clip_range_vf=config.clip_range_vf,
                vf_coef=config.vf_coef,
                ent_coef=config.ent_coef,
                max_grad_norm=config.max_grad_norm,
                target_kl=config.target_kl,
            )

        def _collect_rollouts_this_iteration() -> List:
            """
            Collect ``num_rollouts_per_iter`` trajectories.

            Under DeepSpeed: each rank runs ``collect_rollouts`` locally on its
            share of the work (with full params temporarily gathered), then we
            all-gather the lists so every rank trains on the same buffer.
            Single-GPU: one process does the whole thing.
            """
            if not args.use_deepspeed:
                return math_env.collect_rollouts(
                    num_trajectories=config.num_rollouts_per_iter, verbose=True,
                )

            my_count = my_share(config.num_rollouts_per_iter)
            logger.info(
                "[rank %d] Generating %d/%d rollouts with gathered weights",
                get_rank(), my_count, config.num_rollouts_per_iter,
            )
            with gather_params_for_generation(policy, value):
                local_trajs = math_env.collect_rollouts(
                    num_trajectories=my_count, verbose=False
                )

            _move_trajectories_to_cpu(local_trajs)

            gathered_lists = all_gather_objects(local_trajs)
            combined: List = []
            for chunk in gathered_lists:
                combined.extend(chunk)
            logger.info(
                "[rank %d] Gathered %d rollouts from %d ranks",
                get_rank(), len(combined), get_world_size(),
            )
            return combined

        def _rank0_evaluate(tag: str) -> Dict[str, float]:
            """
            Run GSM8K eval on rank 0 and broadcast the result.  Every rank
            must enter and exit the ``gather_params_for_generation`` context
            together — it's a collective on the ZeRO-3 shards.
            """
            if not args.use_deepspeed:
                return evaluate_policy(policy, tokenizer, config.eval_data_path)

            results: Optional[Dict[str, float]] = None
            with gather_params_for_generation(policy, value):
                if is_main_process():
                    logger.info("[rank 0] %s (gathered weights)", tag)
                    results = evaluate_policy(
                        policy, tokenizer, config.eval_data_path
                    )
            results = broadcast_object(results, src_rank=0)
            return results or {"accuracy": 0.0, "correct": 0, "total": 0}

        if args.skip_initial_eval:
            if is_main_process():
                logger.info(
                    "\n%s\nSKIPPING INITIAL EVALUATION (--skip-initial-eval)\n%s",
                    "=" * 80, "=" * 80,
                )
            initial_eval = {"accuracy": 0.0}
            best_accuracy = 0.0
        else:
            if is_main_process():
                logger.info(
                    "\n%s\nINITIAL EVALUATION (Iteration 0)\n%s", "=" * 80, "=" * 80
                )
            initial_eval = _rank0_evaluate("Initial evaluation")
            best_accuracy = float(initial_eval.get("accuracy", 0.0))

            if is_main_process() and logger_csv is not None:
                logger_csv.log(
                    {
                        "eval/accuracy": initial_eval.get("accuracy", 0.0),
                        "eval/correct": initial_eval.get("correct", 0),
                        "eval/total": initial_eval.get("total", 0),
                    },
                    step=0,
                )

        for iteration in range(1, config.num_iterations + 1):
            iteration_start = time.perf_counter()
            if is_main_process():
                logger.info(
                    "\n%s\nITERATION %d/%d\n%s",
                    "=" * 80, iteration, config.num_iterations, "=" * 80,
                )
            current_phase = math_env.expert_panel.get_current_expert(
                math_env.curriculum_manager.current_iteration
            )
            if is_main_process():
                logger.info(
                    "Active expert phase: %s (%s)",
                    current_phase.name, current_phase.description,
                )

            rollout_start = time.perf_counter()
            trajectories = _collect_rollouts_this_iteration()
            rollout_seconds = time.perf_counter() - rollout_start

            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
            # Shared seed for PPO shuffle so every rank agrees on batch order;
            # per-rank sharding happens inside PPOTrainerDeepSpeed.
            np.random.seed(base_seed + iteration)
            rollout_buffer = RolloutBuffer(
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                pad_token_id=int(pad_id),
            )
            for trajectory in trajectories:
                rollout_buffer.add_trajectory(trajectory)

            buffer_stats = rollout_buffer.get_stats()
            curriculum_stats = aggregate_curriculum_metrics(trajectories)
            replay_stats = math_env.replay_buffer.get_buffer_stats(
                current_iteration=math_env.curriculum_manager.current_iteration
            )
            if is_main_process():
                logger.info("Running PPO update...")
            train_start = time.perf_counter()
            training_metrics = ppo_trainer.train_step(rollout_buffer)
            train_seconds = time.perf_counter() - train_start

            if is_main_process():
                logger.info(
                    "PPO update metrics: policy_loss=%.4f value_loss=%.4f "
                    "entropy=%.4f approx_kl=%.4f clip_fraction=%.4f update_steps=%d",
                    training_metrics["policy_loss"],
                    training_metrics["value_loss"],
                    training_metrics["entropy"],
                    training_metrics["approx_kl"],
                    training_metrics["clip_fraction"],
                    int(training_metrics.get("update_steps", 0.0)),
                )

            eval_start = time.perf_counter()
            if iteration % config.eval_every == 0:
                eval_results = _rank0_evaluate(f"Iteration {iteration} evaluation")
                best_accuracy = max(
                    best_accuracy, float(eval_results.get("accuracy", 0.0))
                )
            else:
                eval_results = {}
            eval_seconds = time.perf_counter() - eval_start

            save_start = time.perf_counter()
            cleanup_metrics = {"deleted_checkpoints": 0, "compressed_logs": 0}
            if iteration % config.save_every == 0:
                # Every rank participates in save_checkpoint; the CheckpointManager
                # delegates to trainer.save_checkpoint which does the right thing
                # for both PPOTrainer and PPOTrainerDeepSpeed.
                checkpoint_manager.save_checkpoint(
                    iteration=iteration, trainer=ppo_trainer
                )
                if is_main_process():
                    cleanup_metrics = checkpoint_manager.cleanup_old_checkpoints(
                        current_iteration=iteration
                    )
                barrier()
            save_seconds = time.perf_counter() - save_start

            total_seconds = time.perf_counter() - iteration_start
            timing_metrics = {
                "rollout_seconds": float(rollout_seconds),
                "train_seconds": float(train_seconds),
                "eval_seconds": float(eval_seconds),
                "save_seconds": float(save_seconds),
                "total_seconds": float(total_seconds),
                "num_rollouts": float(len(trajectories)),
                "estimated_tokens_generated": float(
                    len(trajectories)
                    * (config.max_question_tokens + 4 * config.max_solution_tokens)
                ),
            }
            if is_main_process():
                throughput_metrics = training_monitor.log_iteration_timing(
                    iteration=iteration, timings=timing_metrics
                )
                disk_metrics = training_monitor.check_disk_space()
                gpu_metrics = training_monitor.log_gpu_utilization(
                    gpu_ids=list(range(torch.cuda.device_count()))
                )
                logger.info(
                    "Timing breakdown: rollout=%.1fs train=%.1fs eval=%.1fs save=%.1fs total=%.1fs",
                    timing_metrics["rollout_seconds"],
                    timing_metrics["train_seconds"],
                    timing_metrics["eval_seconds"],
                    timing_metrics["save_seconds"],
                    timing_metrics["total_seconds"],
                )

                all_metrics = {
                    "iteration": iteration,
                    "buffer": buffer_stats,
                    "curriculum": curriculum_stats,
                    "training": training_metrics,
                    "eval": eval_results,
                    "timing": timing_metrics,
                    "throughput": throughput_metrics,
                    "disk": disk_metrics,
                    "gpu": gpu_metrics,
                    "checkpoint_cleanup": cleanup_metrics,
                    "curriculum_state": math_env.curriculum_manager.get_curriculum_stats(),
                    "replay_buffer": replay_stats,
                    "rollout_mix": dict(math_env.last_rollout_mix),
                    "replay_ratio": math_env.last_replay_ratio,
                }
                save_iteration_results(iteration, trajectories, all_metrics, config)

                csv_metrics = {
                    "iteration": iteration,
                    "train/policy_loss": training_metrics["policy_loss"],
                    "train/value_loss": training_metrics["value_loss"],
                    "train/entropy": training_metrics["entropy"],
                    "train/approx_kl": training_metrics["approx_kl"],
                    "train/clip_fraction": training_metrics["clip_fraction"],
                    "rollout/mean_reward": buffer_stats["mean_episode_reward"],
                    "rollout/num_trajectories": len(trajectories),
                    "rollout/mean_length": buffer_stats["mean_episode_length"],
                    "curriculum/topic_diversity": curriculum_stats["topic_diversity"],
                    "curriculum/avg_difficulty": curriculum_stats["avg_difficulty"],
                    "curriculum/avg_novelty": curriculum_stats["avg_novelty"],
                    "curriculum/replay_ratio": math_env.last_replay_ratio,
                    "perf/rollout_time": rollout_seconds,
                    "perf/train_time": train_seconds,
                    "perf/total_time": total_seconds,
                    "perf/tokens_per_second": throughput_metrics.get(
                        "tokens_per_second", 0.0
                    ),
                    "system/disk_free_gb": disk_metrics.get("free_gb", 0.0),
                }
                if eval_results:
                    csv_metrics["eval/accuracy"] = eval_results.get("accuracy", 0.0)
                    csv_metrics["eval/correct"] = eval_results.get("correct", 0)
                    csv_metrics["eval/total"] = eval_results.get("total", 0)
                if gpu_metrics:
                    csv_metrics["system/gpu_util_percent"] = (
                        sum(gpu_metrics.values()) / max(1, len(gpu_metrics))
                    )

                if logger_csv is not None:
                    logger_csv.log(csv_metrics, step=iteration)

            # Keep ranks in lockstep between iterations: without this a fast
            # rank could start the next rollout while another is still saving.
            barrier()

        final_eval = _rank0_evaluate("Final evaluation")
        if is_main_process():
            logger.info(
                "Training complete. Initial acc: %.2f%% | Final acc: %.2f%% | Delta: %.2f%%",
                initial_eval.get("accuracy", 0.0) * 100.0,
                final_eval.get("accuracy", 0.0) * 100.0,
                (final_eval.get("accuracy", 0.0) - initial_eval.get("accuracy", 0.0))
                * 100.0,
            )

            if logger_csv is not None:
                logger_csv.save_summary(
                    {
                        "initial_accuracy": initial_eval.get("accuracy", 0.0),
                        "final_accuracy": final_eval.get("accuracy", 0.0),
                        "improvement": final_eval.get("accuracy", 0.0)
                        - initial_eval.get("accuracy", 0.0),
                        "best_accuracy": best_accuracy,
                        "total_iterations": config.num_iterations,
                        "console_output_path": str(console_log_path)
                        if console_log_path
                        else "",
                    }
                )
                logger_csv.finish()
    finally:
        if console_log_file is not None:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            console_log_file.close()


if __name__ == "__main__":
    main()
