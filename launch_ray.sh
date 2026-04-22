#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1

python scripts/run_ppo_training_curriculum.py \
  --use-ray \
  --num-ray-workers 2 \
  --base-model checkpoints/dual_task_v1 \
  --output-dir checkpoints/ppo_ray \
  --num-iterations 5 \
  --rollouts-per-iter 96 \
  --rollout-batch-size 64 \
  --eval-data-path data/sft/dual_task_val.jsonl \
  --gsm8k-reference-data data/sft/gsm8k_sft.jsonl \
  --skip-initial-eval \
  --checkpoint-keep-last 2 \
  --run-name hackathon_ray
