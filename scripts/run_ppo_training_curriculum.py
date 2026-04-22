"""
PPO training with curriculum-guided dual-task rewards.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
import deepspeed
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.eval_sft_inference import evaluate_gsm8k
from src.rl.math_environment_curriculum import CurriculumMathEnvironment
from src.rl.checkpoint_manager import CheckpointManager
from src.rl.ppo_trainer_deepspeed import PPOTrainerDeepSpeed
from src.rl.ppo_trainer import PPOTrainer
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
        """Check if primary stream is a TTY (for colored output detection)."""
        return getattr(self.primary, 'isatty', lambda: False)()
    
    def fileno(self) -> int:
        """Return file descriptor of primary stream."""
        return self.primary.fileno()


class CurriculumTrainingConfig:
    base_model = "checkpoints/dual_task_v1"
    learning_rate = 1e-6  # Reduced from 5e-6 for more conservative updates on specialized model
    ppo_epochs = 3  # Increased from 2 to allow more gradient steps per batch
    batch_size = 32
    clip_range = 0.3  # Increased from 0.25 to allow larger policy updates
    clip_range_vf = 0.25  # Increased proportionally with clip_range
    vf_coef = 0.5
    ent_coef = 0.02  # Doubled from 0.01 to encourage more exploration
    max_grad_norm = 0.5  # Reduced from 1.0 to prevent large gradient updates
    target_kl = 0.15  # Increased from 0.08 to allow more policy divergence for specialized models

    gamma = 1.0
    gae_lambda = 0.95

    num_rollouts_per_iter = 100
    max_question_tokens = 200
    max_solution_tokens = 500
    temperature = 0.7
    top_p = 0.9
    consensus_temperature = 0.5  # Lowered from 0.7 for higher consensus rate

    num_iterations = 10
    eval_every = 5  # Reduced from 1 to save 80% of eval time
    save_every = 1
    use_torch_compile = True  # Enable torch.compile for faster inference
    use_multi_gpu_rollouts = True
    use_vllm_rollouts = False
    num_rollout_gpus: Optional[int] = None
    rollout_batch_size = 16
    vllm_tensor_parallel_size = 1
    vllm_gpu_memory_utilization = 0.85
    worker_timeout_seconds = 900.0

    output_dir = "checkpoints/ppo_training_curriculum"
    curriculum_checkpoint_dir = "checkpoints/ppo_training_curriculum/curriculum"
    eval_data_path = "data/sft/dual_task_val.jsonl"
    gsm8k_reference_data = "data/sft/gsm8k_sft.jsonl"

    disk_warning_gb = 5.0
    checkpoint_keep_last = 2
    checkpoint_keep_every = 100
    compress_old_logs = True
    
    # Logging
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


def initialize_models(config: CurriculumTrainingConfig, use_deepspeed: bool = False):
    model_path = Path(config.base_model)
    is_adapter = (model_path / "adapter_config.json").exists()
    
    # For DeepSpeed, only rank 0 loads the model to avoid PEFT distributed issues
    is_main_process = not use_deepspeed or int(os.environ.get("LOCAL_RANK", "0")) == 0

    if is_adapter:
        logger.info("Detected LoRA adapter at: %s", config.base_model)
        meta_file = model_path / "pipeline_meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            base_model_name = meta.get("base_model", "Qwen/Qwen2.5-Math-1.5B-Instruct")
        else:
            base_model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Ensure chat template is set - load from base model if needed
        if tokenizer.chat_template is None:
            logger.info("Chat template not found in adapter, loading from base model")
            base_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
            if base_tokenizer.chat_template is not None:
                tokenizer.chat_template = base_tokenizer.chat_template

        if is_main_process or not use_deepspeed:
            # Workaround for PEFT tensor parallel import bug in distributed context
            # Temporarily mock the missing module
            import sys
            import types
            if 'transformers.integrations.tensor_parallel' not in sys.modules:
                logger.info("Creating mock transformers.integrations.tensor_parallel to work around PEFT bug")
                mock_module = types.ModuleType('tensor_parallel')
                sys.modules['transformers.integrations.tensor_parallel'] = mock_module
            
            # Load and merge adapter (only on rank 0 for DeepSpeed)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            policy = PeftModel.from_pretrained(base_model, config.base_model)
            policy = policy.merge_and_unload()
            
            # For DeepSpeed, move merged model to CPU so DeepSpeed can manage placement
            if use_deepspeed:
                logger.info("Moving merged model to CPU for DeepSpeed initialization")
                policy = policy.cpu()
        else:
            # Non-main ranks: load base architecture without weights (DeepSpeed will sync)
            logger.info("Rank %s: Loading model architecture only", os.environ.get("LOCAL_RANK"))
            from transformers import AutoConfig
            config_obj = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
            policy = AutoModelForCausalLM.from_config(config_obj, torch_dtype=torch.bfloat16)

        value = ValueHead(base_model_name, model_device_map=None if use_deepspeed else "auto")
    else:
        logger.info("Loading full model: %s", config.base_model)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if is_main_process or not use_deepspeed:
            policy = AutoModelForCausalLM.from_pretrained(
                config.base_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            
            # For DeepSpeed, move model to CPU so DeepSpeed can manage placement
            if use_deepspeed:
                logger.info("Moving model to CPU for DeepSpeed initialization")
                policy = policy.cpu()
        else:
            # Non-main ranks: load architecture only
            logger.info("Rank %s: Loading model architecture only", os.environ.get("LOCAL_RANK"))
            from transformers import AutoConfig
            config_obj = AutoConfig.from_pretrained(config.base_model, trust_remote_code=True)
            policy = AutoModelForCausalLM.from_config(config_obj, torch_dtype=torch.bfloat16)
        
        value = ValueHead(config.base_model, model_device_map=None if use_deepspeed else "auto")

    policy_device = getattr(policy, "device", "sharded")
    logger.info("Policy loaded on device: %s", policy_device)
    
    # Disable torch.compile when using VLLM (inference tensors incompatible with CUDA graphs)
    if use_deepspeed:
        logger.info("Skipping torch.compile when DeepSpeed training is enabled")
    elif config.use_torch_compile and not config.use_vllm_rollouts:
        try:
            logger.info("Compiling policy model with torch.compile (may take 2-3 min on first run)...")
            policy = torch.compile(policy, mode="reduce-overhead")
            logger.info("Policy model compiled successfully")
        except Exception as e:
            logger.warning("torch.compile failed: %s. Continuing without compilation.", e)
    elif config.use_vllm_rollouts:
        logger.info("Skipping torch.compile (incompatible with VLLM inference mode tensors)")
    
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

    topic_match_scores = []
    difficulty_match_scores = []
    clarity_scores = []
    solvability_scores = []
    novelty_scores = []
    question_rewards = []
    solution_rewards = []
    combined_rewards = []
    pre_expert_rewards = []
    expert_modifiers = []
    expert_phase_counts: Dict[str, int] = {}
    source_counts: Dict[str, int] = {}
    replay_added_count = 0

    for trajectory in trajectories:
        meta = trajectory.metadata
        topic = str(meta["target_topic"])
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
        topic_successes[topic] = topic_successes.get(topic, 0) + int(meta["consensus_achieved"] and meta["primary_matches_majority"])
        topic_difficulty.setdefault(topic, []).append(float(meta["estimated_difficulty"]))

        topic_match_scores.append(float(meta["topic_match_score"]))
        difficulty_match_scores.append(1.0 - abs(float(meta["estimated_difficulty"]) - float(meta["target_difficulty"])))
        clarity_scores.append(float(meta["clarity_score"]))
        solvability_scores.append(1.0 if bool(meta["sympy_verified"]) else 0.0)
        novelty_scores.append(float(meta["novelty_scores"]["combined"]))
        question_rewards.append(float(meta["question_reward"]))
        solution_rewards.append(float(meta["solution_reward"]))
        combined_rewards.append(float(meta["combined_reward"]))
        pre_expert_rewards.append(float(meta.get("pre_expert_reward", meta["combined_reward"])))
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

    return {
        "topics_in_sweet_spot": len(
            [
                topic
                for topic, success in per_topic_success.items()
                if 0.4 <= success <= 0.7
            ]
        ),
        "avg_difficulty": float(sum(difficulty_match_scores) / max(1, len(difficulty_match_scores))),
        "topic_diversity": len(topic_counts),
        "per_topic_success": per_topic_success,
        "per_topic_difficulty": per_topic_difficulty,
        "avg_topic_match": float(sum(topic_match_scores) / max(1, len(topic_match_scores))),
        "avg_difficulty_match": float(sum(difficulty_match_scores) / max(1, len(difficulty_match_scores))),
        "avg_clarity": float(sum(clarity_scores) / max(1, len(clarity_scores))),
        "avg_solvability": float(sum(solvability_scores) / max(1, len(solvability_scores))),
        "avg_novelty": float(sum(novelty_scores) / max(1, len(novelty_scores))),
        "avg_question_reward": float(sum(question_rewards) / max(1, len(question_rewards))),
        "avg_solution_reward": float(sum(solution_rewards) / max(1, len(solution_rewards))),
        "avg_combined_reward": float(sum(combined_rewards) / max(1, len(combined_rewards))),
        "avg_pre_expert_reward": float(sum(pre_expert_rewards) / max(1, len(pre_expert_rewards))),
        "avg_expert_modifier": float(sum(expert_modifiers) / max(1, len(expert_modifiers))),
        "expert_phase_counts": expert_phase_counts,
        "source_counts": source_counts,
        "replay_added_count": replay_added_count,
        "fresh_mean_reward": float(
            sum(
                float(t.metadata["combined_reward"])
                for t in trajectories
                if str(t.metadata.get("rollout_source", "fresh")) == "fresh"
            )
            / max(1, sum(1 for t in trajectories if str(t.metadata.get("rollout_source", "fresh")) == "fresh"))
        ),
        "replay_mean_reward": float(
            sum(
                float(t.metadata["combined_reward"])
                for t in trajectories
                if str(t.metadata.get("rollout_source", "fresh")) == "replay"
            )
            / max(1, sum(1 for t in trajectories if str(t.metadata.get("rollout_source", "fresh")) == "replay"))
        ),
    }


def save_iteration_results(iteration: int, trajectories: List, metrics: Dict[str, object], config: CurriculumTrainingConfig) -> None:
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

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Train PPO with curriculum-guided dual-task rewards")
    parser.add_argument("--base-model", type=str, default="checkpoints/dual_task_v1")
    parser.add_argument("--output-dir", type=str, default="checkpoints/ppo_training_curriculum")
    parser.add_argument("--num-iterations", type=int, default=10)
    parser.add_argument("--rollouts-per-iter", type=int, default=100)
    parser.add_argument("--eval-data-path", type=str, default="data/sft/dual_task_val.jsonl")
    parser.add_argument("--gsm8k-reference-data", type=str, default="data/sft/gsm8k_sft.jsonl")
    parser.add_argument("--skip-initial-eval", action="store_true")
    parser.add_argument("--disable-multi-gpu-rollouts", action="store_true")
    parser.add_argument("--use-vllm-rollouts", action="store_true")
    parser.add_argument("--num-rollout-gpus", type=int, default=None)
    parser.add_argument("--rollout-batch-size", type=int, default=16)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--worker-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--disk-warning-gb", type=float, default=5.0)
    parser.add_argument("--checkpoint-keep-last", type=int, default=2)
    parser.add_argument("--checkpoint-keep-every", type=int, default=100)
    parser.add_argument("--no-compress-old-logs", action="store_true")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run name for logging")
    parser.add_argument("--use-deepspeed", action="store_true", help="Enable DeepSpeed ZeRO-3 training")
    parser.add_argument(
        "--deepspeed-config",
        type=str,
        default="configs/deepspeed_zero3_rl.json",
        help="Path to DeepSpeed config JSON",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank used by DeepSpeed launcher")
    args = parser.parse_args()

    # NOTE: Do NOT initialize distributed here if using adapters
    # We need to load/merge adapters before distributed init to avoid PEFT tensor parallel bug
    if args.use_deepspeed and args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)

    config = CurriculumTrainingConfig()
    config.base_model = args.base_model
    config.output_dir = args.output_dir
    config.num_iterations = args.num_iterations
    config.num_rollouts_per_iter = args.rollouts_per_iter
    config.eval_data_path = args.eval_data_path
    config.gsm8k_reference_data = args.gsm8k_reference_data
    config.curriculum_checkpoint_dir = str(Path(args.output_dir) / "curriculum")
    config.run_name = args.run_name
    config.use_multi_gpu_rollouts = not args.disable_multi_gpu_rollouts
    config.use_vllm_rollouts = args.use_vllm_rollouts
    config.num_rollout_gpus = args.num_rollout_gpus
    config.rollout_batch_size = max(1, int(args.rollout_batch_size))
    config.vllm_tensor_parallel_size = max(1, int(args.vllm_tensor_parallel_size))
    config.vllm_gpu_memory_utilization = float(args.vllm_gpu_memory_utilization)
    config.worker_timeout_seconds = float(args.worker_timeout_seconds)
    config.disk_warning_gb = float(args.disk_warning_gb)
    config.checkpoint_keep_last = max(1, int(args.checkpoint_keep_last))
    config.checkpoint_keep_every = max(1, int(args.checkpoint_keep_every))
    config.compress_old_logs = not args.no_compress_old_logs

    # Initialize CSV logger
    logger_csv = CSVLogger(
        project="ppo-curriculum",
        run_name=config.run_name or f"curriculum_{datetime.now():%Y%m%d_%H%M%S}",
        log_dir=config.log_dir,
        config=vars(config),
        log_detailed=True,
    )
    console_log_path = Path(logger_csv.log_path) / "console_output.log"
    console_log_file = console_log_path.open("a", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, console_log_file)
    sys.stderr = TeeStream(original_stderr, console_log_file)
    logger.info("Full console output is being captured at %s", console_log_path)

    math_env = None
    try:
        policy, value, tokenizer = initialize_models(config, use_deepspeed=args.use_deepspeed)
        reference_questions = load_reference_questions(config.gsm8k_reference_data)

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
            use_vllm=config.use_vllm_rollouts,
            vllm_tensor_parallel_size=config.vllm_tensor_parallel_size,
            vllm_batch_size=config.rollout_batch_size,
            vllm_gpu_memory_utilization=config.vllm_gpu_memory_utilization,
            base_model_path=config.base_model,
            parallel_worker_timeout_seconds=config.worker_timeout_seconds,
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
            logger.info("Using DeepSpeed ZeRO-3 PPO trainer")
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
            policy = ppo_trainer.policy
            value = ppo_trainer.value
            math_env.policy = policy
            math_env.value = value
        else:
            logger.info("Using standard PPO trainer")
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

        if args.skip_initial_eval:
            logger.info("\n%s\nSKIPPING INITIAL EVALUATION (use --skip-initial-eval)\n%s", "=" * 80, "=" * 80)
            initial_eval = {"accuracy": 0.0}
            best_accuracy = 0.0
        else:
            logger.info("\n%s\nINITIAL EVALUATION (Iteration 0)\n%s", "=" * 80, "=" * 80)
            initial_eval = evaluate_policy(policy, tokenizer, config.eval_data_path)
            best_accuracy = float(initial_eval["accuracy"])

            logger_csv.log({
                "eval/accuracy": initial_eval["accuracy"],
                "eval/correct": initial_eval.get("correct", 0),
                "eval/total": initial_eval.get("total", 0),
            }, step=0)

        for iteration in range(1, config.num_iterations + 1):
            iteration_start = time.perf_counter()
            logger.info("\n%s\nITERATION %d/%d\n%s", "=" * 80, iteration, config.num_iterations, "=" * 80)
            current_phase = math_env.expert_panel.get_current_expert(math_env.curriculum_manager.current_iteration)
            logger.info(
                "Active expert phase: %s (%s)",
                current_phase.name,
                current_phase.description,
            )

            rollout_start = time.perf_counter()
            if config.use_multi_gpu_rollouts:
                trajectories = math_env.collect_rollouts_parallel(
                    num_trajectories=config.num_rollouts_per_iter,
                    num_gpus=config.num_rollout_gpus,
                    batch_size=config.rollout_batch_size,
                    verbose=True,
                )
            elif config.use_vllm_rollouts:
                trajectories = math_env.collect_rollouts_batched(
                    num_trajectories=config.num_rollouts_per_iter,
                    batch_size=config.rollout_batch_size,
                    verbose=True,
                )
            else:
                trajectories = math_env.collect_rollouts(
                    num_trajectories=config.num_rollouts_per_iter,
                    verbose=True,
                )
            rollout_seconds = time.perf_counter() - rollout_start

            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
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
            logger.info("Running PPO update...")
            train_start = time.perf_counter()
            training_metrics = ppo_trainer.train_step(rollout_buffer)
            train_seconds = time.perf_counter() - train_start
            logger.info(
                "PPO update metrics: policy_loss=%.4f value_loss=%.4f entropy=%.4f "
                "approx_kl=%.4f clip_fraction=%.4f update_steps=%d",
                training_metrics["policy_loss"],
                training_metrics["value_loss"],
                training_metrics["entropy"],
                training_metrics["approx_kl"],
                training_metrics["clip_fraction"],
                int(training_metrics.get("update_steps", 0.0)),
            )

            sync_metrics = {"workers_synced": 0}
            sync_seconds = 0.0
            if config.use_multi_gpu_rollouts:
                sync_start = time.perf_counter()
                try:
                    sync_metrics = math_env.sync_parallel_rollout_workers()
                except Exception as exc:
                    logger.error("Failed to sync rollout workers: %s", exc)
                    sync_metrics = {"workers_synced": 0, "error": str(exc)}
                sync_seconds = time.perf_counter() - sync_start

            eval_start = time.perf_counter()
            if iteration % config.eval_every == 0:
                eval_results = evaluate_policy(policy, tokenizer, config.eval_data_path)
                best_accuracy = max(best_accuracy, float(eval_results["accuracy"]))
            else:
                eval_results = {}
            eval_seconds = time.perf_counter() - eval_start

            save_start = time.perf_counter()
            cleanup_metrics = {"deleted_checkpoints": 0, "compressed_logs": 0}
            if iteration % config.save_every == 0:
                checkpoint_manager.save_checkpoint(iteration=iteration, trainer=ppo_trainer)
                cleanup_metrics = checkpoint_manager.cleanup_old_checkpoints(current_iteration=iteration)
            save_seconds = time.perf_counter() - save_start

            total_seconds = time.perf_counter() - iteration_start
            timing_metrics = {
                "rollout_seconds": float(rollout_seconds),
                "train_seconds": float(train_seconds),
                "sync_seconds": float(sync_seconds),
                "eval_seconds": float(eval_seconds),
                "save_seconds": float(save_seconds),
                "total_seconds": float(total_seconds),
                "num_rollouts": float(len(trajectories)),
                "estimated_tokens_generated": float(
                    len(trajectories) * (config.max_question_tokens + 4 * config.max_solution_tokens)
                ),
            }
            throughput_metrics = training_monitor.log_iteration_timing(iteration=iteration, timings=timing_metrics)
            disk_metrics = training_monitor.check_disk_space()
            gpu_metrics = training_monitor.log_gpu_utilization(
                gpu_ids=list(range(config.num_rollout_gpus))
                if config.num_rollout_gpus is not None
                else list(range(torch.cuda.device_count()))
            )
            logger.info(
                "Timing breakdown: rollout=%.1fs train=%.1fs sync=%.1fs eval=%.1fs save=%.1fs total=%.1fs",
                timing_metrics["rollout_seconds"],
                timing_metrics["train_seconds"],
                timing_metrics["sync_seconds"],
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
                "sync": sync_metrics,
                "checkpoint_cleanup": cleanup_metrics,
                "parallel_rollout_details": dict(math_env.last_parallel_rollout_details),
                "curriculum_state": math_env.curriculum_manager.get_curriculum_stats(),
                "replay_buffer": replay_stats,
                "rollout_mix": dict(math_env.last_rollout_mix),
                "replay_ratio": math_env.last_replay_ratio,
            }
            save_iteration_results(iteration, trajectories, all_metrics, config)
            
            # Log to CSV (simplified metrics)
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
                "perf/tokens_per_second": throughput_metrics.get("tokens_per_second", 0.0),
                "system/disk_free_gb": disk_metrics.get("free_gb", 0.0),
            }
            
            if eval_results:
                csv_metrics["eval/accuracy"] = eval_results["accuracy"]
                csv_metrics["eval/correct"] = eval_results.get("correct", 0)
                csv_metrics["eval/total"] = eval_results.get("total", 0)
            
            if gpu_metrics:
                avg_gpu_util = sum(gpu_metrics.values()) / max(1, len(gpu_metrics))
                csv_metrics["system/gpu_util_percent"] = avg_gpu_util
            
            logger_csv.log(csv_metrics, step=iteration)
        final_eval = evaluate_policy(policy, tokenizer, config.eval_data_path)
        logger.info(
            "Training complete. Initial acc: %.2f%% | Final acc: %.2f%% | Delta: %.2f%%",
            initial_eval["accuracy"] * 100.0,
            final_eval["accuracy"] * 100.0,
            (final_eval["accuracy"] - initial_eval["accuracy"]) * 100.0,
        )

        # Save final summary
        logger_csv.save_summary({
            "initial_accuracy": initial_eval["accuracy"],
            "final_accuracy": final_eval["accuracy"],
            "improvement": final_eval["accuracy"] - initial_eval["accuracy"],
            "best_accuracy": best_accuracy,
            "total_iterations": config.num_iterations,
            "console_output_path": str(console_log_path),
        })

        logger_csv.finish()
    finally:
        if math_env is not None:
            math_env.shutdown_parallel_rollout_workers()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        console_log_file.close()


if __name__ == "__main__":
    main()
