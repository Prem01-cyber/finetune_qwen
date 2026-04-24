#!/usr/bin/env bash
# GRPO training launcher — much simpler and more stable than PPO.
#
# Why GRPO works:
#   - No value function → no value_loss explosion
#   - No GAE → no lambda^400 ≈ 0 at first token
#   - Group-relative advantages → automatic normalisation per question
#   - Proven on math RL (DeepSeek-Math, Qwen-Math, DAPO)
#
# Expected improvement:
#   - Iters 1-5:  rewards trending up as model avoids wrong-format responses
#   - Iters 5-15: GSM8K accuracy starts moving (+2-5%)
#   - Iters 15-30: continued lift toward 70-75%+ from ~63.6% baseline
#
# Tune --group-size (K): higher K = more signal per question, more VRAM.
#   K=2: ~8 GB VRAM  K=4: ~12 GB VRAM  K=8: ~20 GB VRAM
#
# Tune --questions-per-iter: higher = more gradient signal per step,
#   more wall time per iteration.

set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export TOKENIZERS_PARALLELISM=false

# Disable any hanging triton compile cache on some containers
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"

python scripts/run_grpo_training.py \
    --base-model checkpoints/dual_task_v1 \
    --output-dir checkpoints/grpo \
    --gsm8k-data data/sft/gsm8k_sft.jsonl \
    --eval-data-path data/sft/dual_task_val.jsonl \
    --num-iterations 30 \
    --group-size 4 \
    --questions-per-iter 16 \
    --learning-rate 5e-6 \
    --max-new-tokens 400 \
    --temperature 0.8 \
    --eval-every 5 \
    --eval-max-samples 250 \
    --eval-max-new-tokens 512 \
    --use-prm \
    --prm-model Qwen/Qwen2.5-Math-PRM-7B \
    --run-name "grpo_$(date +%Y%m%d_%H%M%S)" \
    "$@"
