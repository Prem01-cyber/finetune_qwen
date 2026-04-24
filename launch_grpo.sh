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
#   3. --learning-rate 1e-5 — 2× the previous 5e-6. PPO runs converged at
#      this rate (approx_kl ~0.05 target). Loss was only -0.0032 at 5e-6,
#      meaning the policy was barely moving each step. 1e-5 is the standard
#      RLHF range for 1.5B models and is what Qwen-Math / DeepSeek-Math use.
#
#   4. --max-grad-norm 0.5 — tighter than 1.0 default, matches PPO. Prevents
#      rare high-variance groups from spiking the gradient norm.
#
#   5. --eval-every 3 — more frequent eval (10 points vs 6 over 30 iters).
#      Gives much nicer accuracy curves for the hackathon demo plot.
#      Cost: ~4 extra eval runs × ~6 min = ~24 min over the full run.
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

# ── Train ────────────────────────────────────────────────────────────────
#
# What each flag does on this hardware:
#
#   --group-size 8            K=8 batched in ONE model.generate call — near-100%
#                             GPU utilisation vs the old sequential K=8 loop.
#                             Halved wall-time per iteration → 2× iterations in 4.5 h.
#
#   --questions-per-iter 16   N=16 groups per step; gradient is already well-conditioned
#                             at this size — going higher adds wall-time without signal.
#
#   --clip-eps 0.2            PPO-style IS clip: prevents any single high-advantage
#                             group from spiking the policy at 1e-5 LR.
#
#   --warmup-iters 3          Linear LR ramp: 1e-6 → 1e-5 over 3 iterations so the
#                             first gradient step doesn't jolt a freshly-merged model.
#
#   --min-lr-ratio 0.1        Cosine decays to 1e-6 (10% of peak) by iteration 100.
#
#   --difficulty-alpha 2.0    Difficulty-weighted sampling: questions where the model
#                             scores 40-60% of K=8 solutions are sampled most often.
#                             Questions it always gets right / wrong are deprioritised.
#
#   --num-iterations 100      With batched generation, each iter is ~90s on A100,
#                             so 100 iters ≈ 3.5 h + eval overhead.  Much more signal
#                             than the previous 30-iter limit.
#
#   --eval-every 10           10 eval points over 100 iters — still a clear curve.
#
#   --save-every 10           Save checkpoint every 10 iters; best_policy/ always saved
#                             when accuracy improves regardless of this flag.
#
#   --keep-last 3             Keep the 3 newest iter_* checkpoints on disk (~9 GB).
#
python -u scripts/run_grpo_training.py \
    --base-model checkpoints/dual_task_v1 \
    --output-dir checkpoints/grpo \
    --gsm8k-data data/sft/gsm8k_sft.jsonl \
    --eval-data-path data/sft/dual_task_val.jsonl \
    --num-iterations 100 \
    --group-size 8 \
    --questions-per-iter 16 \
    --learning-rate 1e-5 \
    --max-new-tokens 400 \
    --temperature 0.8 \
    --max-grad-norm 0.5 \
    --clip-eps 0.2 \
    --warmup-iters 3 \
    --min-lr-ratio 0.1 \
    --difficulty-alpha 2.0 \
    --overlong-filter \
    --eval-every 10 \
    --eval-max-samples 250 \
    --eval-max-new-tokens 512 \
    --save-every 10 \
    --keep-last 3 \
    --use-prm \
    --prm-model Qwen/Qwen2.5-Math-PRM-7B \
    --run-name "$RUN_NAME" \
    "$@" 2>&1 | tee "$LOG_FILE"
