#!/usr/bin/env bash
# GRPO training launcher — tuned for 1× A100 PCIe 80 GB.
#
# Host profile this script targets:
#   GPU    : 1× A100 PCIe, 80 GB HBM2e, 1583 GB/s, PCIe 4.0 x16
#   CPU    : AMD EPYC 7V13 64-core (96 vCPU available)
#   RAM    : 221.7 GB
#   Disk   : 300 GB virtual (1920 MB/s)
#   Network: ~6 Gbps up
#
# Why this config gives the best results for the hackathon run:
#
#   1. K=8 (group-size) — doubles the learning signal per question vs K=4.
#      GRPO advantages are z-scores within a group; K=8 makes mean/std
#      estimates much tighter, so weak solutions stop pulling the
#      gradient sideways. Marginal VRAM cost (~4 GB extra for K=8 vs K=4)
#      is trivial on an 80 GB card — we were using ~12 GB at K=4.
#
#   2. --questions-per-iter 16 — unchanged. Already healthy (mean_r=0.609,
#      std=0.357, 0 groups skipped). Going higher linearly increases iter
#      wall-time and makes the eval/train ratio worse. Sweet spot.
#
#   3. --learning-rate 5e-6 — halved from 1e-5 after iter 5 LR shock. At 1e-5
#      the policy collapsed the moment warmup ended (mean_r dropped 0.917→0.448,
#      grounded_acc 87%→53%). 5e-6 keeps the update stable through the full run.
#
#   4. --max-grad-norm 0.5 — tighter than 1.0 default, matches PPO. Prevents
#      rare high-variance groups from spiking the gradient norm.
#
#   5. --warmup-iters 8 — extended from 5. At 5 iters the ramp to peak LR
#      coincided exactly with the crash. 8 iters gives a gentler ramp and
#      lets the policy adapt before hitting peak learning rate.
#
#   6. --save-every 5 + --keep-last 3 — at 3 GB per merged 1.5B bf16
#      checkpoint, naive "save every iter" burns 90 GB. Rolling window keeps
#      the disk tidy; best_policy/ is always preserved independently.
#
# ── Environment-variable tuning ──────────────────────────────────────────
#
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#     HF + Flash-Attn 2 fragment the allocator during generation.
#     Expandable segments let PyTorch grow allocations without copying,
#     recovering 2-4 GB of peak VRAM over a long run.
#
#   OMP_NUM_THREADS=8
#     EPYC 7V13 has 96 vCPU. Letting PyTorch spawn 96 OpenMP threads for
#     tiny per-op workloads causes catastrophic contention. 8 is enough
#     for the 1.5B forward passes inside reward scoring.
#
#   TOKENIZERS_PARALLELISM=false
#     Eliminates a fork-after-thread warning from HF tokenizers when we
#     spawn subprocesses for generation.
#
#   HF_HUB_DISABLE_XET=1
#     Falls back to HTTP downloads for the 7B PRM. Xet transport can
#     silently hang for hours on some datacenter networks.
#
# ── Expected wall-time on this host ──────────────────────────────────────
#
#   Initial eval (250 samples)                        : ~5 min
#   Train iter @ K=8 batched, N=16, max_new=400       : ~90 s  (was ~6-7 min sequential)
#   Eval every 10 iters (250 samples) × 10 evals      : ~5 min each × 10 = 50 min
#   100 iterations × 1.5 min                          : ~2.5 h
#   Total                                             : ~3.5 h end-to-end
#
#   This is 3× more iterations than the previous 30-iter run in the same wall-time.
#
# ── Overrides ────────────────────────────────────────────────────────────
#
#   bash launch_grpo.sh                       # run with these defaults
#   bash launch_grpo.sh --group-size 12       # push VRAM harder
#   bash launch_grpo.sh --no-prm              # skip 7B PRM download
#   bash launch_grpo.sh --num-iterations 50   # longer run
#
#   Smoke test:
#     bash launch_grpo.sh --num-iterations 2 --questions-per-iter 8 \
#         --group-size 4 --skip-initial-eval --run-name smoke

set -euo pipefail

# ── GPU / allocator ──────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# ── CPU / threading ──────────────────────────────────────────────────────
# Cap CPU threads so tokenizer/scoring doesn't thrash the 96-vCPU EPYC.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

# ── HuggingFace hub robustness ───────────────────────────────────────────
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}
export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-warning}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton_cache}

# ── Python path ──────────────────────────────────────────────────────────
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# ── Pre-flight: sanity-check the GPU before a 4-hour run ─────────────────
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "─── nvidia-smi ───────────────────────────────────────────────"
    nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version \
               --format=csv,noheader || true
    echo "─────────────────────────────────────────────────────────────"
fi

# ── Log tee: every run gets its own log file under logs/grpo/ ────────────
RUN_NAME="grpo_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/grpo"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"
echo "[launch] run_name = $RUN_NAME"
echo "[launch] log_file = $LOG_FILE"

# ── Smoke test (15 samples, 2 iters, ~3 min) ─────────────────────────────
#
#   bash launch_grpo.sh \
#     --num-iterations 2 --questions-per-iter 4 --group-size 4 \
#     --max-new-tokens 128 --eval-every 1 --eval-max-samples 15 \
#     --eval-max-new-tokens 256 --save-every 0 --keep-last 0 \
#     --math-mix-ratio 0 --self-play-ratio 0 \
#     --output-dir checkpoints/grpo_smoke --run-name smoke
#
#   Plots are auto-saved to checkpoints/grpo_smoke/smoke/plots/ after training.
#   Generate from an existing run at any time:
#     python scripts/plot_grpo_run.py --latest
#
# ── Train ────────────────────────────────────────────────────────────────
#
# What each flag does on this hardware:
#
#   --group-size 8            K=8 batched in ONE model.generate call — near-100%
#                             GPU utilisation vs the old sequential K=8 loop.
#
#   --questions-per-iter 16   N=16 groups per step (well-conditioned gradient).
#
#   --clip-eps 0.2            PPO-style IS clip: prevents policy spikes at 1e-5 LR.
#
#   --kl-coef 0.04            Reference-policy KL penalty: anchors the policy to the
#                             SFT starting checkpoint.  Prevents forgetting GSM8K
#                             facts learned during dual_task_v1 training.
#                             β=0.04 matches DeepSeekMath-GRPO default.
#
#   --warmup-iters 3          Linear LR ramp: 1e-6 → 1e-5 over 3 iterations.
#
#   --min-lr-ratio 0.1        Cosine decays to 1e-6 (10% of peak) by iter 100.
#
#   --difficulty-alpha 3.0    Difficulty-weighted sampling: questions where the model
#                             scores 40-60% of K=8 solutions are sampled most often.
#
#   --math-mix-ratio 0.3      30% of each question batch comes from MATH competition
#                             dataset (difficulty ≤ 3), 70% from GSM8K.  MATH problems
#                             expose the model to harder multi-step reasoning and raise
#                             the accuracy ceiling beyond GSM8K's ~85% saturation.
#                             First run downloads + caches to data/math/math_numeric.jsonl.
#
#   --math-max-difficulty 3   Level 1-2 ≈ AMC-8 (comparable to hard GSM8K).
#                             Level 3 ≈ AMC-10.  Levels 4-5 are too hard for a 1.5B
#                             model to reliably get any reward signal from.
#
#   --self-play-ratio 0.35    35% of groups use SELF-PLAY: the model generates its own
#                             question from a curriculum instruction, then solves it.
#                             Reward = 0.40×question_quality + 0.60×solution_quality.
#                             This is the core Theme #4 self-improvement loop — the model
#                             is rewarded not only for solving correctly but for creating
#                             well-formed, appropriately difficult, solvable challenges.
#                             Raised from 0.30 → 0.35 after smoke run confirmed q_acc=100%
#                             and q_reward≈0.77 — the model generates high-quality questions.
#                             The remaining 65% use GROUNDED (dataset) questions with
#                             gold-answer reward — the primary accuracy anchor.
#
#   --num-iterations 100      ~90s/iter on A100 → 100 iters ≈ 3.5 h total.
#
#   --eval-every 10           10 eval checkpoints over 100 iters.
#
#   --save-every 10           Save every 10 iters; best_policy/ saved on any improvement.
#
#   --keep-last 3             Rolling window of 3 iter_* checkpoints (~9 GB).
#
python -u scripts/run_grpo_training.py \
    --base-model checkpoints/dual_task_v1 \
    --output-dir checkpoints/grpo \
    --gsm8k-data data/sft/gsm8k_sft.jsonl \
    --eval-data-path data/sft/gsm8k_test.jsonl \
    --num-iterations 30 \
    --group-size 8 \
    --q-group-size 2 \
    --questions-per-iter 16 \
    --learning-rate 5e-6 \
    --max-new-tokens 400 \
    --temperature 0.8 \
    --max-grad-norm 0.5 \
    --clip-eps 0.2 \
    --kl-coef 0.04 \
    --warmup-iters 8 \
    --min-lr-ratio 0.1 \
    --difficulty-alpha 3.0 \
    --self-play-ratio 0.30 \
    --math-mix-ratio 0.3 \
    --math-mix-ratio-late 0.5 \
    --math-ramp-start 15 \
    --math-max-difficulty 3 \
    --overlong-filter \
    --eval-every 5 \
    --eval-max-samples 100 \
    --eval-max-new-tokens 400 \
    --eval-pass-at-k 0 \
    --save-every 5 \
    --keep-last 3 \
    --use-prm \
    --prm-model Qwen/Qwen2.5-Math-PRM-7B \
    --run-name "$RUN_NAME" \
    "$@" 2>&1 | tee "$LOG_FILE"
