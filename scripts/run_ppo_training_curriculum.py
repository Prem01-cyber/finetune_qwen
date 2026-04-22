"""
PPO training with curriculum-guided dual-task rewards.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
import wandb
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.eval_sft_inference import evaluate_gsm8k
from src.rl.math_environment_curriculum import CurriculumMathEnvironment
from src.rl.ppo_trainer import PPOTrainer
from src.rl.rollout_buffer import RolloutBuffer
from src.rl.value_network import ValueHead


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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
    eval_every = 1
    save_every = 1

    output_dir = "checkpoints/ppo_training_curriculum"
    curriculum_checkpoint_dir = "checkpoints/ppo_training_curriculum/curriculum"
    eval_data_path = "data/sft/dual_task_val.jsonl"
    gsm8k_reference_data = "data/sft/gsm8k_sft.jsonl"

    use_wandb = True
    wandb_project = "math-self-improvement-curriculum"
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


def initialize_models(config: CurriculumTrainingConfig):
    model_path = Path(config.base_model)
    is_adapter = (model_path / "adapter_config.json").exists()

    if is_adapter:
        logger.info("Detected LoRA adapter at: %s", config.base_model)
        meta_file = model_path / "pipeline_meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            base_model_name = meta.get("base_model", "Qwen/Qwen2.5-Math-1.5B-Instruct")
        else:
            base_model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        policy = PeftModel.from_pretrained(base_model, config.base_model)
        policy = policy.merge_and_unload()

        value = ValueHead(base_model_name).to(policy.device)
    else:
        logger.info("Loading full model: %s", config.base_model)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        policy = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        value = ValueHead(config.base_model).to(policy.device)

    logger.info("Policy loaded on device: %s", policy.device)
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
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    config = CurriculumTrainingConfig()
    config.base_model = args.base_model
    config.output_dir = args.output_dir
    config.num_iterations = args.num_iterations
    config.num_rollouts_per_iter = args.rollouts_per_iter
    config.eval_data_path = args.eval_data_path
    config.gsm8k_reference_data = args.gsm8k_reference_data
    config.curriculum_checkpoint_dir = str(Path(args.output_dir) / "curriculum")
    config.use_wandb = not args.no_wandb

    if config.use_wandb:
        try:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name or f"ppo_curriculum_{datetime.now():%Y%m%d_%H%M%S}",
                config=vars(config),
            )
            logger.info("W&B initialized successfully: project=%s", config.wandb_project)
        except Exception as e:
            logger.error("Failed to initialize W&B: %s", e, exc_info=True)
            logger.warning("Continuing without W&B logging")
            config.use_wandb = False

    policy, value, tokenizer = initialize_models(config)
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
    )

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

    for iteration in range(1, config.num_iterations + 1):
        logger.info("\n%s\nITERATION %d/%d\n%s", "=" * 80, iteration, config.num_iterations, "=" * 80)
        current_phase = math_env.expert_panel.get_current_expert(math_env.curriculum_manager.current_iteration)
        logger.info(
            "Active expert phase: %s (%s)",
            current_phase.name,
            current_phase.description,
        )
        trajectories = math_env.collect_rollouts(
            num_trajectories=config.num_rollouts_per_iter,
            verbose=True,
        )

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
        training_metrics = ppo_trainer.train_step(rollout_buffer)
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

        if iteration % config.eval_every == 0:
            eval_results = evaluate_policy(policy, tokenizer, config.eval_data_path)
            best_accuracy = max(best_accuracy, float(eval_results["accuracy"]))
        else:
            eval_results = {}

        all_metrics = {
            "iteration": iteration,
            "buffer": buffer_stats,
            "curriculum": curriculum_stats,
            "training": training_metrics,
            "eval": eval_results,
            "curriculum_state": math_env.curriculum_manager.get_curriculum_stats(),
            "replay_buffer": replay_stats,
            "rollout_mix": dict(math_env.last_rollout_mix),
            "replay_ratio": math_env.last_replay_ratio,
        }
        save_iteration_results(iteration, trajectories, all_metrics, config)

        if config.use_wandb:
            wandb_metrics = {
                "iteration": iteration,
                "buffer/mean_reward": buffer_stats["mean_episode_reward"],
                "buffer/mean_episode_length": buffer_stats["mean_episode_length"],
                "training/policy_loss": training_metrics["policy_loss"],
                "training/value_loss": training_metrics["value_loss"],
                "training/entropy": training_metrics["entropy"],
                "training/approx_kl": training_metrics["approx_kl"],
                "training/clip_fraction": training_metrics["clip_fraction"],
                "curriculum/num_topics_in_sweet_spot": curriculum_stats["topics_in_sweet_spot"],
                "curriculum/avg_topic_match": curriculum_stats["avg_topic_match"],
                "curriculum/avg_difficulty_match": curriculum_stats["avg_difficulty_match"],
                "curriculum/avg_clarity": curriculum_stats["avg_clarity"],
                "curriculum/avg_solvability": curriculum_stats["avg_solvability"],
                "curriculum/avg_novelty": curriculum_stats["avg_novelty"],
                "reward/question_component": curriculum_stats["avg_question_reward"],
                "reward/solution_component": curriculum_stats["avg_solution_reward"],
                "reward/combined": curriculum_stats["avg_combined_reward"],
                "reward/pre_expert": curriculum_stats["avg_pre_expert_reward"],
                "reward/expert_modifier": curriculum_stats["avg_expert_modifier"],
                "reward/fresh_mean": curriculum_stats["fresh_mean_reward"],
                "reward/replay_mean": curriculum_stats["replay_mean_reward"],
                "expert/phase_counts/pedagogy": curriculum_stats["expert_phase_counts"].get("pedagogy", 0),
                "expert/phase_counts/accuracy": curriculum_stats["expert_phase_counts"].get("accuracy", 0),
                "expert/phase_counts/challenge": curriculum_stats["expert_phase_counts"].get("challenge", 0),
                "replay/ratio": math_env.last_replay_ratio,
                "replay/fresh_rollouts": math_env.last_rollout_mix.get("fresh", 0),
                "replay/replayed_rollouts": math_env.last_rollout_mix.get("replay", 0),
                "replay/new_admissions": curriculum_stats["replay_added_count"],
                "replay/buffer_size": replay_stats.get("buffer_size", 0.0),
                "replay/avg_quality": replay_stats.get("avg_quality", 0.0),
                "replay/quality_variance": replay_stats.get("quality_variance", 0.0),
                "replay/staleness": replay_stats.get("staleness", 0.0),
                "replay/topic_entropy": replay_stats.get("topic_entropy", 0.0),
                "replay/replay_success_rate": replay_stats.get("replay_success_rate", 0.0),
                "replay/buffer_turnover_rate": replay_stats.get("buffer_turnover_rate", 0.0),
                "replay/topics_in_buffer": replay_stats.get("topics_in_buffer", 0.0),
                "replay/buffer_health": replay_stats.get("buffer_health", 0.0),
            }
            for topic, success in curriculum_stats["per_topic_success"].items():
                wandb_metrics[f"curriculum/topic_success/{topic}"] = success
            for topic, difficulty in curriculum_stats["per_topic_difficulty"].items():
                wandb_metrics[f"curriculum/topic_difficulty/{topic}"] = difficulty
            if eval_results:
                wandb_metrics["eval/accuracy"] = eval_results["accuracy"]
            try:
                wandb.log(wandb_metrics)
            except Exception as e:
                logger.error("Failed to log metrics to W&B at iteration %d: %s", iteration, e)
                config.use_wandb = False

        if iteration % config.save_every == 0:
            checkpoint_path = Path(config.output_dir) / f"iteration_{iteration:03d}" / "checkpoint.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            ppo_trainer.save_checkpoint(str(checkpoint_path))

    final_eval = evaluate_policy(policy, tokenizer, config.eval_data_path)
    logger.info(
        "Training complete. Initial acc: %.2f%% | Final acc: %.2f%% | Delta: %.2f%%",
        initial_eval["accuracy"] * 100.0,
        final_eval["accuracy"] * 100.0,
        (final_eval["accuracy"] - initial_eval["accuracy"]) * 100.0,
    )

    if config.use_wandb:
        try:
            wandb.finish()
            logger.info("W&B run finished successfully")
        except Exception as e:
            logger.error("Failed to finish W&B run: %s", e)


if __name__ == "__main__":
    main()
