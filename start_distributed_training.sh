#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Start Distributed Training (on coordinator/trainer GPU)
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
ROLLOUT_SERVER_URL="${1:-}"

if [ -z "$ROLLOUT_SERVER_URL" ]; then
    echo "Error: Rollout server URL required"
    echo ""
    echo "Usage: $0 <ROLLOUT_SERVER_URL>"
    echo ""
    echo "Examples:"
    echo "  $0 http://198.53.64.194:5000"
    echo "  $0 http://127.0.0.1:5000"
    exit 1
fi

# Configuration
BASE_MODEL="${BASE_MODEL:-checkpoints/dual_task_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/ppo_training_distributed}"
NUM_ITERATIONS="${NUM_ITERATIONS:-1000}"
ROLLOUTS_PER_ITER="${ROLLOUTS_PER_ITER:-100}"
EVAL_EVERY="${EVAL_EVERY:-10}"

# Activate environment
source .venv/bin/activate

# Health check
echo "→ Checking rollout server health..."
if curl -sf "$ROLLOUT_SERVER_URL/health" > /dev/null; then
    echo "✓ Rollout server is healthy"
    curl -s "$ROLLOUT_SERVER_URL/health" | python3 -m json.tool
else
    echo "✗ Cannot reach rollout server at $ROLLOUT_SERVER_URL"
    echo ""
    echo "Make sure the rollout server is running:"
    echo "  ssh <worker-machine> 'cd /workspace/finetune_qwen && ./start_rollout_server.sh'"
    exit 1
fi

# Ensure training script exists
if [ ! -f "scripts/run_ppo_training_http_rollouts.py" ]; then
    echo "→ Creating HTTP-based training script..."
    cat > scripts/run_ppo_training_http_rollouts.py <<'PYEOF'
"""PPO training with HTTP-based remote rollout generation."""
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
from src.rl.mdp_components import Trajectory, Transition, State, Action

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def initialize_models(base_model: str):
    """Load policy and value models on coordinator GPU."""
    model_path = Path(base_model)
    is_adapter = (model_path / "adapter_config.json").exists()

    if is_adapter:
        meta_file = model_path / "pipeline_meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
            base_model_name = meta.get("base_model", "Qwen/Qwen2.5-Math-1.5B-Instruct")
        else:
            base_model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_lm = AutoModelForCausalLM.from_pretrained(
            base_model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        policy = PeftModel.from_pretrained(base_lm, base_model).merge_and_unload()
        value = ValueHead(base_model_name).to(policy.device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        policy = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        value = ValueHead(base_model).to(policy.device)

    return policy, value, tokenizer


def fetch_rollouts_from_server(
    server_url: str, num_rollouts: int, prompts: list, max_tokens: int = 512
) -> list:
    """Request batch generation from remote rollout server."""
    response = requests.post(
        f"{server_url}/generate_batch",
        json={
            "prompts": prompts,
            "max_new_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
        },
        timeout=600,
    )
    response.raise_for_status()
    data = response.json()
    logger.info(
        f"Fetched {data['count']} rollouts in {data['elapsed_s']:.1f}s from {server_url}"
    )
    return data["generations"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-server-url", type=str, required=True)
    parser.add_argument("--base-model", type=str, default="checkpoints/dual_task_v1")
    parser.add_argument("--output-dir", type=str, default="checkpoints/ppo_training_distributed")
    parser.add_argument("--num-iterations", type=int, default=1000)
    parser.add_argument("--rollouts-per-iter", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=10)
    args = parser.parse_args()

    logger.info("Initializing coordinator models...")
    policy, value, tokenizer = initialize_models(args.base_model)

    trainer = PPOTrainer(
        policy_model=policy,
        value_model=value,
        tokenizer=tokenizer,
        learning_rate=1e-6,
        ppo_epochs=3,
        batch_size=32,
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for iteration in range(1, args.num_iterations + 1):
        iter_start = time.time()
        
        # Generate placeholder prompts (replace with your curriculum logic)
        prompts = [
            f"Generate a math problem about fractions. Problem {i}:" 
            for i in range(args.rollouts_per_iter)
        ]
        
        logger.info(f"[Iter {iteration}] Fetching {len(prompts)} rollouts from server...")
        generations = fetch_rollouts_from_server(
            args.rollout_server_url, args.rollouts_per_iter, prompts
        )
        
        # TODO: Convert generations to trajectories with proper rewards
        # For now, create dummy trajectories for structure
        logger.info(f"[Iter {iteration}] Processing rollouts into trajectories...")
        
        # Placeholder training step
        logger.info(f"[Iter {iteration}] Running PPO update...")
        
        elapsed = time.time() - iter_start
        logger.info(f"[Iter {iteration}] Complete in {elapsed:.1f}s")

        if iteration % args.eval_every == 0:
            checkpoint_path = Path(args.output_dir) / f"iter_{iteration:04d}" / "checkpoint.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(str(checkpoint_path))
            logger.info(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
PYEOF
fi

echo ""
echo "=========================================="
echo "Starting Distributed Training"
echo "=========================================="
echo "Rollout Server: $ROLLOUT_SERVER_URL"
echo "Base Model: $BASE_MODEL"
echo "Output: $OUTPUT_DIR"
echo "Iterations: $NUM_ITERATIONS"
echo "Rollouts/Iter: $ROLLOUTS_PER_ITER"
echo "=========================================="
echo ""

exec python scripts/run_ppo_training_http_rollouts.py \
    --rollout-server-url "$ROLLOUT_SERVER_URL" \
    --base-model "$BASE_MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --num-iterations "$NUM_ITERATIONS" \
    --rollouts-per-iter "$ROLLOUTS_PER_ITER" \
    --eval-every "$EVAL_EVERY"
