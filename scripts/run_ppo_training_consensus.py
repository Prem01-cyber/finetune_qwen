"""
PPO Training with Consensus Verification

Extends the standard PPO training script to use triple-consensus verification.
This provides better reward signals by catching both semantic and arithmetic errors.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
import wandb
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.eval_sft_inference import evaluate_gsm8k
from src.rl.math_environment_consensus import ConsensusMathEnvironment
from src.rl.ppo_trainer import PPOTrainer
from src.rl.rollout_buffer import RolloutBuffer
from src.rl.value_network import ValueHead


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ConsensusTrainingConfig:
    """PPO training hyperparameters with consensus verification."""

    # Model
    base_model = "checkpoints/dual_task_v1"  # Phase 1 dual-task model

    # PPO hyperparameters (adjusted for smaller model and stability)
    learning_rate = 5e-6  # Reduced from 1e-5 for smaller steps
    ppo_epochs = 2        # Reduced from 4 to avoid early KL stopping
    batch_size = 32       # Standard batch size
    clip_range = 0.2
    clip_range_vf = 0.2
    vf_coef = 0.5
    ent_coef = 0.01
    max_grad_norm = 1.0
    target_kl = 0.03      # Increased from 0.01 to allow more policy change

    # GAE
    gamma = 1.0
    gae_lambda = 0.95

    # Rollout
    num_rollouts_per_iter = 100
    max_question_tokens = 200
    max_solution_tokens = 500
    temperature = 0.7
    top_p = 0.9
    consensus_temperature = 0.7  # Temperature for alternative solutions

    # Training loop
    num_iterations = 10
    eval_every = 1
    save_every = 1

    # Paths
    output_dir = "checkpoints/ppo_training_consensus"

    # Logging
    use_wandb = True
    wandb_project = "math-self-improvement-consensus"
    wandb_run_name = None


def initialize_models(config: ConsensusTrainingConfig):
    """
    Load policy and value networks.

    Returns:
        policy, value, tokenizer
    """
    model_path = Path(config.base_model)
    
    # Check if this is an adapter or full model
    is_adapter = (model_path / "adapter_config.json").exists()
    
    if is_adapter:
        logger.info(f"Detected LoRA adapter at: {config.base_model}")
        
        # Load base model name from metadata
        meta_file = model_path / "pipeline_meta.json"
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
                base_model_name = meta.get("base_model", "Qwen/Qwen2.5-Math-1.5B-Instruct")
        else:
            base_model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"
        
        logger.info(f"Loading base model: {base_model_name}")
        
        # Load tokenizer from adapter
        tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Apply adapter to create policy
        logger.info(f"Applying adapter from: {config.base_model}")
        policy = PeftModel.from_pretrained(base_model, config.base_model)
        policy = policy.merge_and_unload()
        
        # ValueHead uses base model
        value = ValueHead(base_model_name)
        value = value.to(policy.device)
        
    else:
        # Full model path
        logger.info(f"Loading full model: {config.base_model}")
        
        tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        policy = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        value = ValueHead(config.base_model)
        value = value.to(policy.device)

    logger.info(f"Policy loaded on device: {policy.device}")
    logger.info("Value network initialized")

    return policy, value, tokenizer


def compute_consensus_metrics(trajectories: list) -> dict:
    """
    Compute consensus-specific metrics from trajectories.
    
    Args:
        trajectories: List of Trajectory objects
    
    Returns:
        Dict with consensus metrics
    """
    consensus_rates = []
    answer_diversities = []
    sympy_scores = []
    consensus_scores = []
    
    for traj in trajectories:
        sol_metrics = traj.metadata["reward_breakdown"]["solution_metrics"]
        
        # Extract consensus info if available
        if "verification_details" in sol_metrics:
            consensus_info = sol_metrics["verification_details"]["consensus"]
            consensus_rates.append(1.0 if consensus_info["has_majority"] else 0.0)
            answer_diversities.append(consensus_info["answer_diversity"])
        
        # Extract scores
        if "consensus_score" in sol_metrics:
            consensus_scores.append(sol_metrics["consensus_score"])
        if "sympy_score" in sol_metrics:
            sympy_scores.append(sol_metrics["sympy_score"])
    
    return {
        "consensus_rate": sum(consensus_rates) / len(consensus_rates) if consensus_rates else 0.0,
        "mean_answer_diversity": sum(answer_diversities) / len(answer_diversities) if answer_diversities else 0.0,
        "mean_consensus_score": sum(consensus_scores) / len(consensus_scores) if consensus_scores else 0.0,
        "mean_sympy_score": sum(sympy_scores) / len(sympy_scores) if sympy_scores else 0.0,
    }


def save_iteration_results(
    iteration: int,
    trajectories: list,
    metrics: dict,
    config: ConsensusTrainingConfig,
):
    """
    Save iteration outputs with consensus metadata.
    """
    output_dir = Path(config.output_dir) / f"iteration_{iteration:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectories_data = []
    for i, traj in enumerate(trajectories):
        sol_metrics = traj.metadata["reward_breakdown"]["solution_metrics"]
        
        # Extract verification details
        verification_details = sol_metrics.get("verification_details", {})
        
        trajectories_data.append(
            {
                "trajectory_id": i,
                "instruction": traj.metadata["instruction"],
                "generated_question": traj.metadata["generated_question"],
                "generated_solution": traj.metadata["generated_solution"],
                "total_reward": traj.total_reward,
                "reward_breakdown": traj.metadata["reward_breakdown"],
                "length": len(traj),
                # Consensus-specific metadata
                "verification": {
                    "sympy": verification_details.get("sympy_verification", {}),
                    "consensus": verification_details.get("consensus", {}),
                },
                "primary_solution": verification_details.get("primary_solution", ""),
                "alternative_solutions": verification_details.get("alternative_solutions", []),
            }
        )

    with open(output_dir / "trajectories.jsonl", "w") as f:
        for item in trajectories_data:
            f.write(json.dumps(item) + "\n")

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Iteration {iteration} results saved to {output_dir}")


def evaluate_policy(
    policy,
    tokenizer,
    eval_data_path: str = "data/sft/dual_task_val.jsonl",
) -> dict:
    """
    Evaluate policy on GSM8K test set.
    """
    logger.info("Evaluating policy on GSM8K test set...")

    results = evaluate_gsm8k(
        model=policy,
        tokenizer=tokenizer,
        data_path=eval_data_path,
        max_samples=500,
    )

    logger.info(
        f"GSM8K Accuracy: {results['accuracy']:.2%} "
        f"({results['correct']}/{results['total']})"
    )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train mathematical reasoning agent with PPO + Consensus Verification"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="checkpoints/dual_task_v1",
        help="Path to Phase 1 dual-task model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/ppo_training_consensus",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=10,
        help="Number of PPO iterations",
    )
    parser.add_argument(
        "--rollouts-per-iter",
        type=int,
        default=100,
        help="Number of trajectories per iteration",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--skip-initial-eval",
        action="store_true",
        help="Skip initial evaluation at iteration 0",
    )

    args = parser.parse_args()

    config = ConsensusTrainingConfig()
    config.base_model = args.base_model
    config.output_dir = args.output_dir
    config.num_iterations = args.num_iterations
    config.num_rollouts_per_iter = args.rollouts_per_iter
    config.use_wandb = not args.no_wandb

    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name
            or f"ppo_consensus_{datetime.now():%Y%m%d_%H%M%S}",
            config=vars(config),
        )

    policy, value, tokenizer = initialize_models(config)

    # Create consensus-based math environment
    math_env = ConsensusMathEnvironment(
        policy_model=policy,
        value_model=value,
        tokenizer=tokenizer,
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

    # Initial evaluation
    if not args.skip_initial_eval:
        logger.info("\n" + "=" * 80)
        logger.info("INITIAL EVALUATION (Iteration 0)")
        logger.info("=" * 80)

        initial_eval = evaluate_policy(policy, tokenizer)
        best_accuracy = initial_eval["accuracy"]

        if config.use_wandb:
            wandb.log(
                {
                    "iteration": 0,
                    "eval/accuracy": initial_eval["accuracy"],
                    "eval/exact_match": initial_eval.get("exact_match", 0.0),
                }
            )
    else:
        logger.info("\n" + "=" * 80)
        logger.info("SKIPPING INITIAL EVALUATION (use --skip-initial-eval)")
        logger.info("=" * 80)
        initial_eval = {"accuracy": 0.0}
        best_accuracy = 0.0

    # Training loop
    for iteration in range(1, config.num_iterations + 1):
        logger.info("\n" + "=" * 80)
        logger.info(f"ITERATION {iteration}/{config.num_iterations}")
        logger.info("=" * 80)

        # ===== STEP 1: COLLECT ROLLOUTS =====
        logger.info(f"\nCollecting {config.num_rollouts_per_iter} rollouts...")

        trajectories = math_env.collect_rollouts(
            num_trajectories=config.num_rollouts_per_iter, verbose=True
        )

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id or 0
        rollout_buffer = RolloutBuffer(
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            pad_token_id=int(pad_id),
        )

        for traj in trajectories:
            rollout_buffer.add_trajectory(traj)

        buffer_stats = rollout_buffer.get_stats()
        
        # Compute consensus-specific metrics
        consensus_metrics = compute_consensus_metrics(trajectories)
        
        logger.info(
            f"Buffer stats: "
            f"{buffer_stats['num_trajectories']} trajectories, "
            f"{buffer_stats['total_steps']} steps, "
            f"mean reward: {buffer_stats['mean_episode_reward']:.3f}"
        )
        logger.info(
            f"Consensus metrics: "
            f"consensus_rate={consensus_metrics['consensus_rate']:.2%}, "
            f"answer_diversity={consensus_metrics['mean_answer_diversity']:.2f}"
        )

        # ===== STEP 2: PPO UPDATE =====
        logger.info("\nRunning PPO update...")

        training_metrics = ppo_trainer.train_step(rollout_buffer)

        logger.info(
            f"PPO update complete: "
            f"policy_loss={training_metrics['policy_loss']:.4f}, "
            f"value_loss={training_metrics['value_loss']:.4f}, "
            f"entropy={training_metrics['entropy']:.4f}"
        )

        # ===== STEP 3: EVALUATION =====
        if iteration % config.eval_every == 0:
            logger.info("\nEvaluating policy...")

            eval_results = evaluate_policy(policy, tokenizer)

            if eval_results["accuracy"] > best_accuracy:
                logger.info(
                    f"New best accuracy: {eval_results['accuracy']:.2%} "
                    f"(previous: {best_accuracy:.2%})"
                )
                best_accuracy = eval_results["accuracy"]

            # Early stopping if degradation
            if eval_results["accuracy"] < best_accuracy - 0.05:
                logger.warning(
                    f"Accuracy degraded by >5% "
                    f"({eval_results['accuracy']:.2%} vs {best_accuracy:.2%}). "
                    "Stopping training."
                )
                break
        else:
            eval_results = {}

        # ===== STEP 4: LOGGING =====
        all_metrics = {
            "iteration": iteration,
            "buffer": buffer_stats,
            "training": training_metrics,
            "consensus": consensus_metrics,
            "eval": eval_results,
        }

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
                "consensus/consensus_rate": consensus_metrics["consensus_rate"],
                "consensus/answer_diversity": consensus_metrics["mean_answer_diversity"],
                "consensus/consensus_score": consensus_metrics["mean_consensus_score"],
                "consensus/sympy_score": consensus_metrics["mean_sympy_score"],
            }

            if eval_results:
                wandb_metrics["eval/accuracy"] = eval_results["accuracy"]

            wandb.log(wandb_metrics)

        # ===== STEP 5: SAVE =====
        save_iteration_results(iteration, trajectories, all_metrics, config)

        if iteration % config.save_every == 0:
            checkpoint_path = (
                Path(config.output_dir)
                / f"iteration_{iteration:03d}"
                / "checkpoint.pt"
            )
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            ppo_trainer.save_checkpoint(str(checkpoint_path))

    # Final evaluation
    if not args.skip_initial_eval:
        logger.info("\n" + "=" * 80)
        logger.info("FINAL EVALUATION")
        logger.info("=" * 80)

        final_eval = evaluate_policy(policy, tokenizer)

        logger.info(
            f"\nTraining complete!"
            f"\nInitial accuracy: {initial_eval['accuracy']:.2%}"
            f"\nFinal accuracy:   {final_eval['accuracy']:.2%}"
            f"\nImprovement:      {final_eval['accuracy'] - initial_eval['accuracy']:.2%}"
        )
    else:
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE (evaluations skipped)")
        logger.info("=" * 80)

    if config.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
