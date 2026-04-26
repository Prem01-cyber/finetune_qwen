#!/usr/bin/env bash
# =============================================================================
# launch_grpo_combined.sh  —  GRPO training on NuminaMath-CoT + OpenMathInstruct-2
#
# Host profile:
#   GPU    : 1× A100 SXM4 / PCIe, 80 GB HBM2e, 1583 GB/s
#   CPU    : AMD EPYC 7V13 64-core (96 vCPU available)
#   RAM    : 221.7 GB
#
# Dataset:
#   • NuminaMath-CoT + OpenMathInstruct-2 (combined_train.jsonl)
#   • 14 distinct skill_ids → ZPD CurriculumManager gets real per-topic signal
#   • Clean \\boxed{} / numeric answers → gt_match_rate stays ≥ 55%
#   • Difficulty tiers 1–3 per problem → sparse-reward regime at tiers 2-3
#   • 20K+ diverse problems → prevents reward-hacking via memorisation
#
# Flash-Attention 2
# -----------------
#   src/utils/attn_backend.py picks the BEST available backend at runtime:
#
#       flash_attention_2  (O(T) attention memory, 1.5-2.5× faster backward)
#           ↓  not installed
#       sdpa               (torch.nn.functional.scaled_dot_product_attention)
#           ↓  not available
#       eager              (stock HF, slowest)
#
#   This script installs flash-attn if missing before training begins.
#   Impact on this run (K=10, T≈600, 28 layers, bf16):
#     - Attention activation memory: ~1.5 GB saved per backward pass
#     - Rollout speed: ~1.5× faster .generate() for long contexts
#     - Gradient checkpointing automatically DISABLED when Flash is active
#
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   HF + Flash-Attn 2 fragment the allocator during generation.
#   Expandable segments recover 2-4 GB peak VRAM over a long run.
#
# ── Hyperparameter rationale (v2, refined from grpo_20260425_151304 run) ────
#
# OBSERVED in prior run (GSM8K, K=8, N=16, 20 iters):
#   • mean_reward ~0.85-0.92 from iter 1 → model already near GSM8K ceiling (78%)
#   • 25-56% of groups skipped per iter (K=8 too small → zero variance on easy Qs)
#   • gt_match stayed flat; no meaningful policy improvement after iter 10
#   • SDPA fallback (no flash-attn) → ~2× slower than needed
#
# KEY CHANGES this run:
#
#   --group-size 10             Increased from 8. More rollouts per question
#                               → better reward variance estimate → fewer
#                               zero-variance skipped groups. Expected: skip
#                               rate drops from ~40% to ~20% on harder combined
#                               data.
#
#   --questions-per-iter 20     Increased from 16. More diverse gradient signal
#                               per iteration. Combined with K=10: 200 rollouts
#                               vs prior 128 → ~56% more signal per iter.
#                               VRAM stays safe at K_q=2, K=10, N=20.
#
#   --difficulty-alpha 4.5      Sharper ZPD sampling (was 4.0). With 3 difficulty
#                               tiers and a model that started strong on simple
#                               problems, we need a stronger bias toward tier-2/3
#                               problems (30-70% win-rate zone) to generate
#                               non-trivial gradient signal.
#
#   --num-iterations 80         More iterations (was 60). Combined dataset is
#                               harder — curriculum needs more steps to climb from
#                               ~55% to 70%+ eval accuracy.
#
#   --eval-every 5              Unchanged. Eval every 5 iters gives 16 checkpoints
#                               (plus iter 0) over 80 iters.
#
#   --eval-max-samples 200      Increased from 150. Larger sample → tighter CI
#                               on eval combined_score; avoids spurious saves.
#
#   --selfplay-gt-thresh 0.50   Lowered from 0.52. Combined dataset problems are
#                               harder → natural gt_match is lower; a gate of 0.52
#                               would delay self-play entry too long.
#
#   --selfplay-ramp-iters 30    Slower ramp (was 25). More grounded-only iters
#                               before the self-play ratio climbs, ensuring the
#                               policy is solid on harder combined problems.
#
#   --kl-coef 0.06              Tighter KL anchor (was 0.04 in GSM8K run).
#                               Harder problems increase policy variance; anchor
#                               prevents drift from the SFT checkpoint.
#
#   --grounded-floor 0.52       Raised (was 0.50). Tighter recovery trigger;
#                               prevents gt_match collapse seen in prior runs.
#
#   --self-play-ratio 0.60      Reduced from 0.70. Harder dataset needs more
#                               grounded signal in the mix.
#
#   --warmup-iters 10           Longer LR warmup (was 8). New distribution needs
#                               a gentler ramp before peak LR.
#
#   --min-warmup 10             Longer GROUNDED_ONLY minimum phase (was 8).
#                               Ensures at least 10 iters of grounded-only before
#                               any self-play can activate.
#
#   --learning-rate 3e-6        Unchanged. Right-sized for harder combined data.
#
#   --max-new-tokens 1200       Increased from 1000. NuminaMath competition
#                               problems often need longer chain-of-thought.
#
#   --math-mix-ratio 0.0        DISABLED — the combined dataset already contains
#   --math-mix-ratio-late 0.0   competition-level problems.
#   --math-ramp-start 999
#
# ── Expected reward trajectory ──────────────────────────────────────────────
#
#   Iters  1-10  (GROUNDED_ONLY): mean_reward 0.55→0.75, gt_match ≥ 50%
#   Iters 11-30  (SELFPLAY_RAMP): mean_reward 0.75→0.85, sp_ratio 0%→40%
#   Iters 31-80  (SELFPLAY_RAMP): mean_reward 0.85→0.90, sp_ratio 40%→60%
#   Eval accuracy on combined_val: expect 62-72% at iter 80
#     (harder than GSM8K — prior run peaked at 80% final-ans on GSM8K)
#
#   Wall-time (A100-80GB, K=10, N=20):
#     Model + PRM load             :  ~2 min
#     Initial eval (200 samples)   :  ~4 min
#     Train iter (Flash active)    :  ~130 s  (K=10 vs K=8 → ~15% longer)
#     Train iter (SDPA fallback)   :  ~180 s
#     Eval checkpoint × 16         :  ~4 min each = 64 min
#     80 iterations × 130 s        :  ~173 min
#     Total (Flash active)         :  ~4.1 h
#     Total (SDPA fallback)        :  ~5.5 h
#
# ── Smoke test (~15 min) ─────────────────────────────────────────────────────
#
#   bash launch_grpo_combined.sh \
#     --num-iterations 4 \
#     --questions-per-iter 8 \
#     --group-size 4 \
#     --eval-every 2 \
#     --eval-max-samples 20 \
#     --skip-initial-eval \
#     --save-every 0 --keep-last 0 \
#     --output-dir checkpoints/smoke_combined
#
# =============================================================================

set -euo pipefail

# ── Data paths ───────────────────────────────────────────────────────────────
TRAIN_DATA="data/sft/combined_train.jsonl"
EVAL_DATA="data/sft/combined_val.jsonl"
EXTRACTION_CACHE="data/extraction_cache_combined.json"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "[launch] Combined training data not found at $TRAIN_DATA"
    echo "[launch] Running the data pipeline first (~5 min for 20K problems) …"
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

# ── Flash-Attention 2 install (if missing) ────────────────────────────────────
#
# flash-attn needs to match (torch version, CUDA version, Python version).
# We use MAX_JOBS to cap parallel compilation to avoid OOM during build.
# If the prebuilt wheel is available it installs in <30 s; source build ~10 min.
#
if ! python -c "import flash_attn; assert int(flash_attn.__version__.split('.')[0]) >= 2" 2>/dev/null; then
    echo "[launch] flash-attn not found or < v2 — installing now …"
    MAX_JOBS=4 pip install flash-attn --no-build-isolation -q
    echo "[launch] flash-attn installed."
else
    FLASH_VER=$(python -c "import flash_attn; print(flash_attn.__version__)" 2>/dev/null)
    echo "[launch] flash-attn ${FLASH_VER} already installed — skipping install."
fi

# ── GPU / allocator ───────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# expandable_segments: recovers 2-4 GB fragmented VRAM during long Flash+HF runs
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# ── CPU / threading ───────────────────────────────────────────────────────────
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

# ── Triton / Flash-Attn compilation cache ─────────────────────────────────────
# Persist Triton JIT kernels across runs — avoids ~30 s recompile each launch.
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton_cache}
# Flash-Attn 2 uses Triton for its CUDA kernels.
# Setting this suppresses the "your GPU may not be supported" warning on A100.
export FLASH_ATTENTION_SKIP_CUDA_BUILD=${FLASH_ATTENTION_SKIP_CUDA_BUILD:-FALSE}

# ── HuggingFace hub robustness ────────────────────────────────────────────────
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-0}
export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-warning}

# ── Python path ───────────────────────────────────────────────────────────────
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# ── Pre-flight: GPU info ───────────────────────────────────────────────────────
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "─── nvidia-smi ───────────────────────────────────────────────────"
    nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version \
               --format=csv,noheader || true
    echo "──────────────────────────────────────────────────────────────────"
fi

# ── Confirm attention backend that will be selected ───────────────────────────
python - <<'PYEOF'
import sys; sys.path.insert(0, '.')
from src.utils.attn_backend import select_attn_implementation
impl = select_attn_implementation()
tag = {
    "flash_attention_2": "FAST   — Flash-Attn 2 active (O(T) memory, 1.5-2.5× faster)",
    "sdpa":              "OK     — SDPA active (no flash-attn; install for ~2× speedup)",
    "eager":             "SLOW   — Eager fallback (install flash-attn for best performance)",
}.get(impl, impl)
print(f"[launch] attn_backend = {tag}")
PYEOF

# ── Log tee ───────────────────────────────────────────────────────────────────
RUN_NAME="grpo_combined_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/grpo"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"

echo "[launch] run_name     = $RUN_NAME"
echo "[launch] base_model   = $BASE_MODEL"
echo "[launch] train_data   = $TRAIN_DATA  ($(wc -l < "$TRAIN_DATA") rows)"
echo "[launch] eval_data    = $EVAL_DATA"
echo "[launch] log_file     = $LOG_FILE"
echo "[launch] architecture = two-phase self-play (K_q=2, K=10, N=20)"
echo "[launch] dataset      = NuminaMath-CoT + OpenMathInstruct-2 (14 skill_ids)"
echo "[launch] wall-time    ≈ 4.1 h  Flash active  /  5.5 h  SDPA fallback"

# ── Train ─────────────────────────────────────────────────────────────────────
python -u scripts/run_grpo_training.py \
    --base-model            "$BASE_MODEL" \
    --output-dir            checkpoints/grpo_combined \
    --gsm8k-data            "$TRAIN_DATA" \
    --eval-data-path        "$EVAL_DATA" \
    \
    --num-iterations        80 \
    --group-size            10 \
    --q-group-size          2 \
    --questions-per-iter    20 \
    \
    --learning-rate         3e-6 \
    --max-new-tokens        1200 \
    --temperature           0.8 \
    --max-grad-norm         0.5 \
    --clip-eps              0.2 \
    --kl-coef               0.06 \
    --warmup-iters          10 \
    --min-lr-ratio          0.1 \
    \
    --difficulty-alpha      4.5 \
    --self-play-ratio       0.60 \
    \
    --math-mix-ratio        0.0 \
    --math-mix-ratio-late   0.0 \
    --math-ramp-start       999 \
    --math-max-difficulty   3 \
    \
    --overlong-filter \
    --min-warmup            10 \
    --selfplay-gt-thresh    0.50 \
    --selfplay-grounded-thresh 0.58 \
    --selfplay-step-thresh  0.63 \
    --selfplay-ramp-iters   30 \
    --grounded-floor        0.52 \
    \
    --extractor-model       Qwen/Qwen2.5-0.5B-Instruct \
    --extraction-cache      "$EXTRACTION_CACHE" \
    \
    --eval-every            5 \
    --eval-max-samples      200 \
    --eval-max-new-tokens   1200 \
    --eval-pass-at-k        0 \
    --save-every            5 \
    --keep-last             4 \
    \
    --use-prm \
    --prm-model             Qwen/Qwen2.5-Math-PRM-7B \
    --run-name              "$RUN_NAME" \
    "$@" 2>&1 | tee "$LOG_FILE"
