"""
PPO training with remote HTTP-based rollout generation.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import requests
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rl.ppo_trainer import PPOTrainer
from src.rl.rollout_buffer import RolloutBuffer
from src.rl.value_network import ValueHead
from scripts.run_ppo_training_curriculum import load_reference_questions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def initialize_models(base_model: str):
    model_path = Path(base_model)
    is_adapter = (model_path / "adapter_config.json").exists()
    
    if is_adapter:
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
    
    return policy, value, tokenizer


def main():
    parser = argparse.ArgumentParser(description="PPO training with remote rollout server")
    parser.add_argument("--server-url", type=str, required=True, help="HTTP URL of rollout server")
    parser.add_argument("--base-model", type=str, default="checkpoints/dual_task_v1")
    parser.add_argument("--output-dir", type=str, default="checkpoints/ppo_training_remote")
    parser.add_argument("--rollouts-per-iter", type=int, default=100)
    parser.add_argument("--num-iterations", type=int, default=1000)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()
    
    logger.info("Testing connection to rollout server: %s", args.server_url)
    try:
        r = requests.get(f"{args.server_url}/health", timeout=10)
        r.raise_for_status()
        health = r.json()
        logger.info("Rollout server healthy: %s", health)
    except Exception as e:
        logger.error("Cannot reach rollout server at %s: %s", args.server_url, e)
        sys.exit(1)
    
    logger.info("Loading trainer models on H100...")
    policy, value, tokenizer = initialize_models(args.base_model)
    
    trainer = PPOTrainer(
        policy_model=policy,
        value_model=value,
        tokenizer=tokenizer,
        learning_rate=1e-6,
        ppo_epochs=3,
        batch_size=32,
        clip_range=0.3,
        clip_range_vf=0.25,
        vf_coef=0.5,
        ent_coef=0.02,
        max_grad_norm=0.5,
        target_kl=0.15,
    )
    
    logger.info("Starting remote PPO training loop...")
    
    for iteration in range(1, args.num_iterations + 1):
        iter_start = time.time()
        
        # Generate prompts (placeholder - replace with your curriculum logic)
        prompts = [
            f"Generate a grade-school math problem about addition. Problem {i} for iteration {iteration}."
            for i in range(args.rollouts_per_iter)
        ]
        
        # Request batch generation from remote server
        logger.info("[Iter %d] Requesting %d rollouts from remote server...", iteration, len(prompts))
        rollout_start = time.time()
        
        try:
            r = requests.post(
                f"{args.server_url}/generate_batch",
                json={
                    "prompts": prompts,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                },
                timeout=600,
            )
            r.raise_for_status()
            data = r.json()
            generations = data["generations"]
            rollout_elapsed = time.time() - rollout_start
            
            logger.info(
                "[Iter %d] Received %d generations in %.2fs (server: %.2fs)",
                iteration,
                len(generations),
                rollout_elapsed,
                data["elapsed_s"],
            )
        except Exception as e:
            logger.error("Rollout generation failed: %s", e)
            continue
        
        # TODO: Convert generations to trajectories and run PPO update
        # For now, just log timing
        
        iter_elapsed = time.time() - iter_start
        logger.info("[Iter %d] Complete in %.2fs", iteration, iter_elapsed)
    
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
