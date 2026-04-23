#!/usr/bin/env bash
# Launch single-GPU PPO training on the curriculum-guided math environment.
#
# Stack (all on one GPU, loaded sequentially in the same process):
#   * Policy         : Qwen2.5-Math-1.5B-Instruct   (bf16, trainable)
#   * Critic         : ValueHead (shared backbone + MLP)
#   * PRM scorer     : Qwen/Qwen2.5-Math-PRM-7B      (4-bit, eval-only)
#   * Rollouts       : 70% self-play + 30% GSM8K-anchored by default
#
# Peak VRAM ~22 GB on A100, comfortably fits 40 GB or 80 GB cards.
# Override any flag by appending it (``"$@"`` is forwarded unchanged).
#
#   bash launch_ppo_training.sh --num-iterations 50 --rollouts-per-iter 32
#   bash launch_ppo_training.sh --no-prm   # skip the 7B PRM download
#   bash launch_ppo_training.sh --grounded-ratio 0.0  # pure self-play
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python scripts/run_ppo_training_curriculum.py \
  --base-model checkpoints/dual_task_v1 \
  --output-dir checkpoints/ppo_curriculum \
  --num-iterations 50 \
  --rollouts-per-iter 32 \
  --eval-data-path data/sft/dual_task_val.jsonl \
  --gsm8k-reference-data data/sft/gsm8k_sft.jsonl \
  --grounded-ratio 0.3 \
  --use-prm \
  --prm-model Qwen/Qwen2.5-Math-PRM-7B \
  --eval-every 5 \
  --eval-max-samples 500 \
  --eval-max-new-tokens 512 \
  --checkpoint-keep-last 5 \
  --checkpoint-keep-every 10 \
  --run-name "ppo_curriculum_$(date +%Y%m%d_%H%M)" \
  "$@"
