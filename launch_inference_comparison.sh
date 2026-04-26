#!/usr/bin/env bash
# =============================================================================
# launch_inference_comparison.sh
#
# Runs the base Qwen2.5-Math-1.5B-Instruct model vs the RL fine-tuned model
# side-by-side on GSM8K questions and writes three report files:
#
#   reports/<run_id>/results.json    — full per-question data
#   reports/<run_id>/report.html     — rich HTML with side-by-side solutions
#   reports/<run_id>/summary.md      — markdown table for docs / README
#
# The RL checkpoint is auto-detected from checkpoints/grpo*/  (picks the most
# recently modified best_policy or latest iter_* checkpoint).  Override with
# --finetuned <path>.
#
# ── Quick-start examples ─────────────────────────────────────────────────────
#
#   # Default: 50 questions, greedy decode, auto-detect RL checkpoint
#   bash launch_inference_comparison.sh
#
#   # Pin a specific GRPO checkpoint
#   bash launch_inference_comparison.sh \
#       --finetuned checkpoints/grpo/grpo_20260425_151304/best_policy
#
#   # Compare SFT-only vs GRPO best_policy (skip HuggingFace download)
#   bash launch_inference_comparison.sh \
#       --base-checkpoint checkpoints/dual_task_v1 \
#       --base-label      "SFT (dual-task v1)" \
#       --finetuned       checkpoints/grpo/grpo_20260425_151304/best_policy \
#       --finetuned-label "GRPO best (iter 10)"
#
#   # More samples with stochastic decode for diversity
#   bash launch_inference_comparison.sh \
#       --max-samples 200 \
#       --temperature 0.7 \
#       --max-new-tokens 1000
#
# =============================================================================

set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export TRANSFORMERS_VERBOSITY=error
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# ── GPU info ──────────────────────────────────────────────────────────────────
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "─── GPU ──────────────────────────────────────────────────────────"
    nvidia-smi --query-gpu=name,memory.total,memory.free \
               --format=csv,noheader || true
    echo "──────────────────────────────────────────────────────────────────"
fi

python -u scripts/run_inference_comparison.py \
    --data-path     data/sft/gsm8k_test.jsonl \
    --max-samples   50 \
    --max-new-tokens 800 \
    --temperature   0.0 \
    --reports-dir   reports \
    "$@"
