#!/usr/bin/env bash
# Launch single-GPU PPO training on the curriculum-guided math environment.
#
# Stack (all on one GPU, loaded sequentially in the same process):
#   * Policy         : Qwen2.5-Math-1.5B-Instruct   (bf16, trainable)
#   * Critic         : ValueHead (frozen backbone + MLP head only)
#   * PRM scorer     : Qwen/Qwen2.5-Math-PRM-7B      (4-bit, eval-only)
#   * Rollouts       : 70% self-play + 30% GSM8K-anchored by default
#
# Peak VRAM ~22 GB on A100, comfortably fits 40 GB or 80 GB cards.
# Override any flag by appending it ("$@" is forwarded unchanged).
#
#   bash launch_ppo_training.sh --num-iterations 100
#   bash launch_ppo_training.sh --grounded-ratio 0.5   # more ground-truth signal
#   bash launch_ppo_training.sh --no-prm               # skip the 7B PRM download
#
# Smoke test (3 iters, 16 rollouts, no initial eval):
#   bash launch_ppo_training.sh \
#       --num-iterations 3 --rollouts-per-iter 16 --skip-initial-eval \
#       --run-name "smoke_$(date +%Y%m%d_%H%M)"
#
# ── Reward design (all fixes as of 2026-04-24) ───────────────────────────────
#
#   Five bugs were diagnosed and patched that caused the original flat-line
#   (63-64% GSM8K for 20 iterations with zero net improvement):
#
#   1. Reward saturation: expert panel used multiplicative shaping
#      clip01(base × (1+modifier)) which pushed most rollouts to combined=1.0,
#      collapsing advantages to zero after buffer whitening.
#      FIX: additive shaping, |modifier| ≤ 0.08, no clip-to-1 inside panel.
#
#   2. PRM triple-counting: PRM_mean appeared in sol (0.55 weight) AND in the
#      expert panel via correctness/consensus keys (another ~0.3 weight).
#      FIX: expert panel now only shapes on format_compliance + question
#      quality — correctness signal lives in sol alone.
#
#   3. Broken-solution reward leak: PRM-degraded rollouts still earned
#      0.4 × question_reward ≈ 0.1–0.2 combined reward.
#      FIX: sol_valid=False → effective_question_reward=0, combined=0.
#
#   4. PRM gaming: combined=0.83 with SymPy=0, Format=0.30 was possible.
#      FIX: format_score < 0.5 caps combined at 0.3 (FLOOR tag in logs).
#
#   5. Misleading log: "combined=X = 0.55×PRM_mean..." hid Q and modifier.
#      FIX: log now shows clip(base + mod, cap) | Q= sol= components.
#
# ── PPO gradient fixes ────────────────────────────────────────────────────────
#
#   6. Sparse terminal reward: only the last token of ~400 got reward R.
#      GAE decay 0.95^400 ≈ 1e-9 → 99%+ of transitions had advantage ≈ 0
#      after normalization → approx_kl ≈ 0.001, clip_frac ≈ 1%, no learning.
#      FIX: reward spread across ALL output tokens (per-token = R / T) so
#      every token's advantage ∝ R − V(s_t), matching TRL/InstructGPT.
#      Normalised by T to keep GAE return targets in [0, R] ≈ [0, 1] and
#      avoid value_loss explosion (was 275 before normalisation, now ~0.08).
#
#   7. Learning rate too low: 3e-6 produced approx_kl ≈ 0.001 (target 0.05)
#      after 1024 gradient steps — weights essentially frozen.
#      FIX: raised to 1e-5 (standard RLHF range for 1.5B models).
#      GAE lambda raised from 0.95 → 0.98 for smoother advantage propagation.
#
# ── KL / training-budget knobs ───────────────────────────────────────────────
#
#   --target-kl 0.05          Looser than canonical 0.015-0.03; grounded
#                             rollouts bound collapse risk.  After the reward
#                             fixes, approx_kl is expected in 0.03-0.06 range
#                             rather than the old 0.001 (too low) or 0.26
#                             (too high from un-normalised reward spreading).
#   --kl-trip-multiplier 1.5  Trip fires at 0.075.  Raise to 2.0 if you still
#                             see early-stop before 80% budget is used.
#   --ppo-epochs 2            Down from 3 (third epoch barely landed updates).
#
# ── Memory / throughput knobs ────────────────────────────────────────────────
#
#   --batch-size 16           Safe with Flash-Attn 2 (O(T) attention memory).
#   --learning-rate 1e-5      Exposed as CLI flag; override to 3e-5 for faster
#                             early learning or 3e-6 if KL keeps tripping early.
#   PYTORCH_CUDA_ALLOC_CONF   expandable_segments prevents 2-4 GB fragmentation.
#   fused AdamW               Auto-selected when all trainable params are CUDA.
#
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
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
  --learning-rate 1e-5 \
  --target-kl 0.05 \
  --kl-trip-multiplier 1.5 \
  --ppo-epochs 2 \
  --clip-range 0.2 \
  --batch-size 16 \
  --eval-every 5 \
  --eval-max-samples 250 \
  --eval-max-new-tokens 512 \
  --checkpoint-keep-last 5 \
  --checkpoint-keep-every 10 \
  --run-name "ppo_curriculum_$(date +%Y%m%d_%H%M)" \
  "$@"
