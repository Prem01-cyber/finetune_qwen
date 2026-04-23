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
#
# Smoke test (3 iters, 16 rollouts, no initial eval) — verifies the
# speed-path changes end-to-end without committing to a long run:
#   bash launch_ppo_training.sh \
#       --num-iterations 3 --rollouts-per-iter 16 --skip-initial-eval \
#       --run-name "smoke_$(date +%Y%m%d_%H%M)"
#
# Speed path (active by default, no flags needed):
#   * Flash-Attn 2 on policy + value + PRM when `flash-attn` is installed,
#     auto-fallback to SDPA otherwise.  Expect ~1.5-2.5x faster attention
#     and O(T) attention memory — which is why gradient checkpointing is
#     auto-disabled when Flash is active (force on with --grad-checkpoint).
#   * KV-cached rollouts via HF generate(output_logits=True).  The old
#     custom loop re-forwarded the whole growing sequence every step
#     (O(T^2)); this is O(T).  Expect rollouts ~4-5x faster at T=500.
#   * Batched value computation: one backbone forward over the full
#     trajectory, all T value estimates gathered from hidden states.
#
# PPO KL knobs (see run_ppo_training_curriculum.py for full docs):
#   --target-kl 0.05          looser than canonical 0.015-0.03 on purpose —
#                             grounded rollouts bound collapse risk and a
#                             tighter threshold cut most iterations to 1/3
#                             of planned epochs.  Per-iter log prints
#                             "updates=X/Y" so you can tell immediately
#                             whether the budget is being honored.
#   --kl-trip-multiplier 1.5  canonical; push to 2.0-2.5 to make early-stop
#                             effectively never fire (pair with lower kl).
#
# Memory / throughput knobs (tuned after Flash-Attn 2 + KV-cached rollouts):
#   --batch-size 16           Flash-Attn 2 gave O(T) attention memory
#                             instead of O(T^2), so we can double B
#                             from the OOM-safe 8 without hitting the
#                             80 GB A100 ceiling.  Halves the number of
#                             PPO mini-batches → ~35-40% faster PPO.
#   --ppo-epochs 2            down from 3.  KL-trip fires inside epoch 2
#                             in most iterations anyway; the third epoch
#                             barely contributed to learning and cost a
#                             flat ~33% of the PPO-update budget.
#   grad checkpointing OFF    auto-disabled when flash_attention_2 is
#                             active (use --grad-checkpoint to force on).
#                             Skipping the recompute-during-backward
#                             gives ~30% faster backward.
#   PYTORCH_CUDA_ALLOC_CONF   expandable_segments:True handles the
#                             "2-3 GB reserved-but-unallocated"
#                             fragmentation the runtime warns about.
#   fused AdamW               auto-selected in PPOTrainer when all
#                             trainable params are on CUDA (~5-10%
#                             faster optimiser step, free of charge).
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# Tell the PyTorch caching allocator to grow segments on demand instead
# of keeping large fragmented blocks.  This is the exact mitigation the
# OOM error message tells us to apply and typically recovers 2-4 GB of
# usable VRAM in PPO-style workloads where tensor shapes fluctuate.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

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
  --target-kl 0.05 \
  --kl-trip-multiplier 1.5 \
  --ppo-epochs 2 \
  --clip-range 0.2 \
  --batch-size 16 \
  --eval-every 5 \
  --eval-max-samples 500 \
  --eval-max-new-tokens 512 \
  --checkpoint-keep-last 5 \
  --checkpoint-keep-every 10 \
  --run-name "ppo_curriculum_$(date +%Y%m%d_%H%M)" \
  "$@"
