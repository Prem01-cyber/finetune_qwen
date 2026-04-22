#!/usr/bin/env bash
# Launch 2-GPU DeepSpeed ZeRO-3 PPO training.
#
# Use N GPUs by changing --num_gpus=N and CUDA_VISIBLE_DEVICES.  ZeRO-3 shards
# both actor and critic across every visible GPU; rollouts are data-parallel.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
NUM_GPUS=${NUM_GPUS:-2}

deepspeed --num_gpus="${NUM_GPUS}" scripts/run_ppo_training_curriculum.py \
  --use-deepspeed \
  --deepspeed-config configs/deepspeed_zero3_rl.json \
  --base-model checkpoints/dual_task_v1 \
  --output-dir checkpoints/ppo_deepspeed_2gpu \
  --num-iterations 5 \
  --rollouts-per-iter 96 \
  --eval-data-path data/sft/dual_task_val.jsonl \
  --gsm8k-reference-data data/sft/gsm8k_sft.jsonl \
  --skip-initial-eval \
  --checkpoint-keep-last 2 \
  --run-name "ppo_ds_${NUM_GPUS}gpu_$(date +%Y%m%d_%H%M)"
