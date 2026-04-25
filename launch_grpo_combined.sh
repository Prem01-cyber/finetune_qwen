#!/usr/bin/env bash
# =============================================================================
# launch_grpo_combined.sh  —  GRPO training on NuminaMath-CoT + OpenMathInstruct-2
#
# This replaces GSM8K/AQuA with a single high-quality combined dataset that:
#   • Has 14 distinct skill_ids → ZPD CurriculumManager gets real per-topic signal
#   • Clean \\boxed{} / numeric answers → gt_match_rate stays ≥ 60% at all times
#   • Difficulty tiers 1–3 per problem → sparse-reward regime at tiers 2-3
#   • 30K+ diverse problems → prevents reward-hacking via memorisation
#
# Pre-requisite (one-time, ~5 min):
#   python scripts/prepare_combined_dataset.py
#
# Then launch:
#   bash launch_grpo_combined.sh
#
# ── Why these hyperparameter changes vs launch_grpo.sh ─────────────────────
#
#   --learning-rate 3e-6        Reduced from 5e-6. Combined dataset is harder
#                               than GSM8K alone; smaller LR prevents over-
#                               shooting in early iters.
#
#   --warmup-iters 10           More warmup (was 8). New distribution needs a
#                               longer gentle ramp before hitting peak LR.
#
#   --kl-coef 0.06              Slightly tighter KL (was 0.04). Harder problems
#                               increase policy variance; anchor prevents drift.
#
#   --selfplay-ramp-iters 25    Slower self-play ramp (was 18). Let the model
#                               stabilise on grounded harder problems before
#                               the self-play ratio climbs.
#
#   --selfplay-grounded-thresh 0.58   Higher gate (was 0.55). Ensures the
#                               grounded policy is truly solid before self-play
#                               pulls training budget away from it.
#
#   --grounded-floor 0.52       Raised (was 0.50). Tighter recovery trigger;
#                               prevents the gt_match collapse seen in AQuA run.
#
#   --self-play-ratio 0.60      Reduced from 0.70. Harder dataset needs more
#                               grounded signal; 60% sp keeps question-learning
#                               while not starving answer learning.
#
#   --difficulty-alpha 4.0      Sharper difficulty-weighted sampling (was 3.0).
#                               With 3 difficulty tiers the model needs stronger
#                               bias toward 30-70% win-rate problems to get
#                               non-trivial RL signal.
#
#   --math-mix-ratio 0.0        DISABLED — the combined dataset already contains
#   --math-mix-ratio-late 0.0   competition-level problems (numina_competition,
#   --math-ramp-start 999       openmath_competition). A separate MATH mix would
#                               double-count them.
#
#   --num-iterations 60         More iterations (was 30/50). Harder curriculum
#                               needs more steps to show measurable improvement.
#
#   --max-new-tokens 1000       Slightly longer (was 800). NuminaMath problems
#                               often require more reasoning steps.
#
#   --min-warmup 8              Longer minimum GROUNDED_ONLY phase (was 6).
#
# ── Expected reward trajectory ──────────────────────────────────────────────
#
#   Iters  1-10  (GROUNDED_ONLY): mean_reward 0.50→0.70, gt_match ≥ 55%
#   Iters 11-25  (SELFPLAY_RAMP): mean_reward 0.70→0.82, sp_ratio 0%→40%
#   Iters 26-60  (SELFPLAY_RAMP): mean_reward 0.82→0.88, sp_ratio 40%→60%
#   Eval accuracy on combined_val: expect 55-65% (harder than GSM8K's 78-80%)
#
# ── Smoke test (~10 min) ─────────────────────────────────────────────────────
#
#   bash launch_grpo_combined.sh \
#     --num-iterations 4 \
#     --questions-per-iter 8 \
#     --group-size 4 \
#     --eval-every 2 \
#     --eval-max-samples 20 \
#     --skip-initial-eval \
#     --save-every 0 --keep-last 0 \
#     --math-mix-ratio 0 \
#     --output-dir checkpoints/smoke_combined
#
# =============================================================================

set -euo pipefail

# ── Data paths ───────────────────────────────────────────────────────────────
TRAIN_DATA="data/sft/combined_train.jsonl"
EVAL_DATA="data/sft/combined_val.jsonl"
EXTRACTION_CACHE="data/extraction_cache_combined.json"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "Combined training data not found at $TRAIN_DATA."
    echo "Running the data pipeline first (≈5 min for 35K problems) …"
    python scripts/prepare_combined_dataset.py
fi

# ── Base model ────────────────────────────────────────────────────────────────
# Start fresh from the SFT checkpoint (not the GSM8K GRPO checkpoint — that one
# is biased toward GSM8K patterns and will need to unlearn before learning the
# harder combined distribution, which is worse than starting clean).
BASE_MODEL="${GRPO_BASE_MODEL:-checkpoints/dual_task_v1}"

if [ ! -d "$BASE_MODEL" ]; then
    echo "ERROR: base model not found at $BASE_MODEL"
    echo "  Set GRPO_BASE_MODEL=<path> or update BASE_MODEL in this script."
    exit 1
fi

# ── GPU / allocator ───────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# ── CPU / threading ───────────────────────────────────────────────────────────
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

# ── HuggingFace hub robustness ────────────────────────────────────────────────
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}
export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-warning}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton_cache}

# ── Python path ───────────────────────────────────────────────────────────────
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# ── Pre-flight ────────────────────────────────────────────────────────────────
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "─── nvidia-smi ───────────────────────────────────────────────────"
    nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version \
               --format=csv,noheader || true
    echo "──────────────────────────────────────────────────────────────────"
fi

# ── Log tee ───────────────────────────────────────────────────────────────────
RUN_NAME="grpo_combined_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/grpo"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"

echo "[launch] run_name    = $RUN_NAME"
echo "[launch] base_model  = $BASE_MODEL"
echo "[launch] train_data  = $TRAIN_DATA"
echo "[launch] eval_data   = $EVAL_DATA"
echo "[launch] log_file    = $LOG_FILE"
echo "[launch] architecture = two-phase self-play (K_q=2, K=8)"
echo "[launch] dataset     = NuminaMath-CoT + OpenMathInstruct-2 (14 skill_ids)"
echo "[launch] estimated wall-time ≈ 4 h (60 iters × ~120s + 12 evals × ~3 min)"

# ── Train ─────────────────────────────────────────────────────────────────────
python -u scripts/run_grpo_training.py \
    --base-model            "$BASE_MODEL" \
    --output-dir            checkpoints/grpo_combined \
    --gsm8k-data            "$TRAIN_DATA" \
    --eval-data-path        "$EVAL_DATA" \
    \
    --num-iterations        60 \
    --group-size            8 \
    --q-group-size          2 \
    --questions-per-iter    16 \
    \
    --learning-rate         3e-6 \
    --max-new-tokens        1000 \
    --temperature           0.8 \
    --max-grad-norm         0.5 \
    --clip-eps              0.2 \
    --kl-coef               0.06 \
    --warmup-iters          10 \
    --min-lr-ratio          0.1 \
    \
    --difficulty-alpha      4.0 \
    --self-play-ratio       0.60 \
    \
    --math-mix-ratio        0.0 \
    --math-mix-ratio-late   0.0 \
    --math-ramp-start       999 \
    --math-max-difficulty   3 \
    \
    --overlong-filter \
    --min-warmup            8 \
    --selfplay-gt-thresh    0.52 \
    --selfplay-grounded-thresh 0.58 \
    --selfplay-step-thresh  0.63 \
    --selfplay-ramp-iters   25 \
    --grounded-floor        0.52 \
    \
    --extractor-model       Qwen/Qwen2.5-0.5B-Instruct \
    --extraction-cache      "$EXTRACTION_CACHE" \
    \
    --eval-every            5 \
    --eval-max-samples      150 \
    --eval-max-new-tokens   1000 \
    --eval-pass-at-k        0 \
    --save-every            5 \
    --keep-last             3 \
    \
    --use-prm \
    --prm-model             Qwen/Qwen2.5-Math-PRM-7B \
    --run-name              "$RUN_NAME" \
    "$@" 2>&1 | tee "$LOG_FILE"
