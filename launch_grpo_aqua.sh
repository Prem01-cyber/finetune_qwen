#!/usr/bin/env bash
# =============================================================================
# launch_grpo_aqua.sh  —  Continue GRPO training on AQuA-RAT algebra dataset
#
# Resumes from the best GRPO checkpoint trained on GSM8K and continues with
# AQuA-RAT (Algebraic Question Answering with Rationales) — harder, more
# algebraic problems that push the model beyond grade-school arithmetic.
#
# Pre-requisite: convert the dataset first (one-time, ~2 min):
#   python scripts/prepare_aqua_dataset.py
#
# Then launch:
#   bash launch_grpo_aqua.sh
#
# Overrides (same as launch_grpo.sh):
#   bash launch_grpo_aqua.sh --num-iterations 20
#   bash launch_grpo_aqua.sh --no-prm
# =============================================================================

set -euo pipefail

# ── Checkpoint to resume from ─────────────────────────────────────────────
# Point at the best_policy saved by the GSM8K GRPO run.
# Update this path if a newer run produces a better checkpoint.
RESUME_CKPT="${GRPO_RESUME_CKPT:-checkpoints/grpo/grpo_20260425_151304/best_policy}"

if [ ! -d "$RESUME_CKPT" ]; then
    echo "ERROR: resume checkpoint not found: $RESUME_CKPT"
    echo "  Set GRPO_RESUME_CKPT=<path> or update RESUME_CKPT in this script."
    exit 1
fi

# ── Data paths ───────────────────────────────────────────────────────────
TRAIN_DATA="data/sft/aqua_train.jsonl"
EVAL_DATA="data/sft/aqua_validation.jsonl"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "AQuA-RAT training data not found at $TRAIN_DATA."
    echo "Running prepare_aqua_dataset.py first …"
    python scripts/prepare_aqua_dataset.py --output-dir data/sft --splits train test validation
fi

# ── GPU / allocator ──────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# ── CPU / threading ──────────────────────────────────────────────────────
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

# ── Pre-flight: sanity-check the GPU ─────────────────────────────────────
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "─── nvidia-smi ───────────────────────────────────────────────"
    nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version \
               --format=csv,noheader || true
    echo "─────────────────────────────────────────────────────────────"
fi

# ── Log tee ───────────────────────────────────────────────────────────────
RUN_NAME="grpo_aqua_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/grpo"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"
echo "[launch] run_name    = $RUN_NAME"
echo "[launch] resume_from = $RESUME_CKPT"
echo "[launch] train_data  = $TRAIN_DATA"
echo "[launch] log_file    = $LOG_FILE"
echo "[launch] architecture = AQuA continuation (two-phase self-play, K_q=2, K=8)"
echo "[launch] estimated wall-time ≈ 2.5 h (30 iters × ~3 min + 6 evals × ~4 min)"

# ── Train ────────────────────────────────────────────────────────────────
python -u scripts/run_grpo_training.py \
    --base-model "$RESUME_CKPT" \
    --output-dir checkpoints/grpo_aqua \
    --gsm8k-data "$TRAIN_DATA" \
    --eval-data-path "$EVAL_DATA" \
    --num-iterations 30 \
    --group-size 8 \
    --q-group-size 2 \
    --questions-per-iter 16 \
    --learning-rate 3e-6 \
    --max-new-tokens 800 \
    --temperature 0.8 \
    --max-grad-norm 0.5 \
    --clip-eps 0.2 \
    --kl-coef 0.05 \
    --warmup-iters 4 \
    --min-lr-ratio 0.1 \
    --difficulty-alpha 3.0 \
    --self-play-ratio 0.60 \
    --math-mix-ratio 0.0 \
    --math-mix-ratio-late 0.0 \
    --math-ramp-start 999 \
    --math-max-difficulty 3 \
    --overlong-filter \
    --min-warmup 4 \
    --selfplay-gt-thresh 0.50 \
    --selfplay-grounded-thresh 0.55 \
    --selfplay-step-thresh 0.60 \
    --selfplay-ramp-iters 18 \
    --grounded-floor 0.45 \
    --extractor-model Qwen/Qwen2.5-0.5B-Instruct \
    --extraction-cache data/extraction_cache_aqua.json \
    --eval-every 5 \
    --eval-max-samples 100 \
    --eval-max-new-tokens 800 \
    --eval-pass-at-k 0 \
    --save-every 5 \
    --keep-last 3 \
    --use-prm \
    --prm-model Qwen/Qwen2.5-Math-PRM-7B \
    --run-name "$RUN_NAME" \
    "$@" 2>&1 | tee "$LOG_FILE"
