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
#   Initial eval (250 samples)           : ~5 min
#   Train iter @ K=8, N=16, max_new=400  : ~6-7 min
#   Eval every 3 iters (250 samples)     : ~5 min each × 10 = 50 min
#   30 iterations × 7 min                : ~3.5 h
#   Total                                : ~4.5 h end-to-end
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
python -u scripts/run_grpo_training.py \
    --base-model checkpoints/dual_task_v1 \
    --output-dir checkpoints/grpo \
    --gsm8k-data data/sft/gsm8k_sft.jsonl \
    --eval-data-path data/sft/dual_task_val.jsonl \
    --num-iterations 30 \
    --group-size 8 \
    --questions-per-iter 16 \
    --learning-rate 1e-5 \
    --max-new-tokens 400 \
    --temperature 0.8 \
    --max-grad-norm 0.5 \
    --eval-every 3 \
    --eval-max-samples 250 \
    --eval-max-new-tokens 512 \
    --save-every 5 \
    --keep-last 3 \
    --use-prm \
    --prm-model Qwen/Qwen2.5-Math-PRM-7B \
    --run-name "$RUN_NAME" \
    "$@" 2>&1 | tee "$LOG_FILE"
