#!/usr/bin/env bash
# Launch single-GPU PPO training on the curriculum-guided math environment.
#
# A 1.5B policy + ValueHead critic + AdamW optimiser + rollouts fit in
# under 25 GB in bfloat16, so we pin to one GPU (CUDA_VISIBLE_DEVICES=0 by
# default) and let rollouts + PPO updates run sequentially in the same
# process.  Override any flag below via env vars or positional args.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

python scripts/run_ppo_training_curriculum.py \
  --base-model checkpoints/dual_task_v1 \
  --output-dir checkpoints/ppo_curriculum \
  --num-iterations 5 \
  --rollouts-per-iter 96 \
  --eval-data-path data/sft/dual_task_val.jsonl \
  --gsm8k-reference-data data/sft/gsm8k_sft.jsonl \
  --skip-initial-eval \
  --checkpoint-keep-last 2 \
  --run-name "ppo_curriculum_$(date +%Y%m%d_%H%M)" \
  "$@"
