#!/usr/bin/env bash
# =============================================================================
# launch_grpo.sh  —  GRPO training on GSM8K + MATH  (A100-80 GB)
#
# Host profile:
#   GPU    : 1× A100 SXM4-80GB, 80 GB HBM2e, 1583 GB/s
#   CPU    : AMD EPYC 7V13 64-core (96 vCPU)
#   RAM    : 221.7 GB
#
# ── Architecture: Two-phase self-play GRPO ──────────────────────────────────
#
#   PATH 1 — Grounded (initial phase):
#     [GSM8K/MATH question + gold_final]
#       → K=10 solutions batched in one .generate() call
#       → reward = 0.50×gold_match + 0.40×PRM_process + 0.10×format
#       → GRPO backward on solutions (group relative advantage)
#
#   PATH 2 — Two-phase self-play (--q-group-size 2):
#     [curriculum instruction]
#       → generate_questions_batched(K_q=2) + store old_log_probs
#       → for each of K_q=2 question candidates:
#            K=10 solutions → score → solution GRPO backward
#       → question GRPO backward over K_q=2 questions
#       → optimizer.step()  (applies BOTH gradient sets)
#     This teaches the model to generate questions that lead to
#     high-PRM, correctly-structured solutions.
#
# ── What went wrong in run grpo_20260425_151304 ─────────────────────────────
#
#   Problem 1 — Premature self-play entry (iter 6)
#     gt_match was only 74% (barely above the 0.55 gate).
#     LR was still at 3.3e-6 during warmup — policy not yet stable.
#     Selfplay then ramped 0→44% in just 9 iters (iter 6→15).
#
#   Problem 2 — Dual-pressure collapse (iter 15)
#     MATH ratio was simultaneously ramping 30%→46% (math-ramp-start=8).
#     + Selfplay at 44%  +  MATH at 46%  +  peak LR  →  gt_match dropped
#     to 48.6%, triggering the safety gate suspension.
#     → Eval score: 0.922 (iter 10 best) → 0.912 (iter 15) → 0.907 (iter 20).
#
#   Problem 3 — K=8 too small; ~35-50% groups skipped early
#     Model started at 78% correct on GSM8K.  With K=8, questions the model
#     gets right 7-8 out of 8 times all-correct → zero variance → skipped.
#     Lost 35-56% of compute per iteration as wasted rollouts.
#
#   Problem 4 — Run too short (30 iters)
#     Recovery from the iter-15 collapse was just starting (iter 17-20: gt_match
#     climbing back, sp_ratio stabilising at 70%).  Run ended before the full
#     improvement curve played out.
#
#   Problem 5 — No flash-attn (SDPA fallback ~2× slower)
#     Iter time bloated to 262-330 s once question-gen started.
#
# ── Fixes this run ──────────────────────────────────────────────────────────
#
#   --group-size 10             K=10 (was 8).  With model starting at 78%
#                               accuracy, K=8 had 35-56% zero-var skips.
#                               K=10 reduces skip rate to ~20%, increases
#                               gradient signal per iter by ~50%.
#
#   --questions-per-iter 20     N=20 (was 16).  200 rollouts/iter vs 128.
#                               More diverse question sampling → cleaner
#                               advantages → more stable policy updates.
#
#   --min-warmup 12             (was 6).  In the prior run self-play entered at
#                               iter 6 while LR was still in warmup (3.3e-6 of
#                               the 5e-6 peak).  Now self-play cannot start
#                               until iter 12, by which time warmup is complete
#                               and the policy is stable.
#
#   --selfplay-gt-thresh 0.65   (was 0.55).  Require gt_match ≥ 65% before
#                               self-play entry (prior run entered at 74%, but
#                               it was oscillating).  Higher gate = more stable
#                               grounded policy before question generation begins.
#
#   --selfplay-grounded-thresh 0.65   (was 0.60).  Match the gt_match gate.
#
#   --selfplay-step-thresh 0.68 (was 0.65).  Tighter step-accuracy gate.
#
#   --selfplay-ramp-iters 28    (was 18).  Slower self-play ramp.  Prior run
#                               hit 44% sp_ratio in just 9 iters; now the same
#                               ramp takes 28 iters → less aggressive pressure.
#
#   --grounded-floor 0.55       (was 0.50).  Raise the safety-gate floor.
#                               Prior run triggered the gate at gt_match=48.6%
#                               (which is 2% below the old floor of 0.50!).
#                               Raising to 0.55 triggers earlier, before deep
#                               collapse, giving a shorter recovery time.
#
#   --kl-coef 0.06              (was 0.04).  Tighter KL anchor.  The gt_match
#                               collapse at iter 15 showed the policy was
#                               drifting too far from the SFT checkpoint.
#                               0.06 keeps updates more conservative.
#
#   --math-ramp-start 18        (was 8).  The simultaneous MATH ratio ramp
#                               (30→46%) + selfplay ramp caused the iter-15
#                               collapse.  Delay MATH ramp until iter 18 so
#                               the selfplay ramp has stabilised first.
#
#   --math-mix-ratio 0.30       (unchanged).  GSM8K baseline mix.
#   --math-mix-ratio-late 0.50  (unchanged).  Ramp to 50% MATH in later iters.
#   --math-max-difficulty 3     (unchanged).  Level 3 is max for 1.5B model.
#
#   --max-new-tokens 1000       (was 800).  More room for MATH problems that
#                               require longer chain-of-thought.
#
#   --difficulty-alpha 3.5      (was 3.0).  Slightly sharper ZPD sampling.
#                               With a model starting at 78%, we need a stronger
#                               bias toward tier-2/3 problems (the 30-70% win-
#                               rate zone) to generate non-trivial gradient signal.
#
#   --num-iterations 60         (was 30).  Long enough to observe:
#                                 • full GROUNDED_ONLY phase (iters 1-12)
#                                 • slow self-play ramp (iters 13-40)
#                                 • sustained improvement phase (iters 40-60)
#                               Prior run peaked at iter 10 and was cut while
#                               still recovering from the iter-15 collapse.
#
#   --eval-every 5              (unchanged).  12 eval checkpoints over 60 iters
#                               → enough points for smooth visualization curves.
#
#   --eval-max-samples 150      (was 100).  Tighter confidence interval on the
#                               combined_score → fewer spurious best-policy saves.
#
#   --keep-last 4               (was 3).  Retain one extra checkpoint window for
#                               post-run analysis and rollback if needed.
#
#   flash-attn install          Added pre-flight install check (absent in the
#                               prior launch script).  Drops iter time from
#                               ~300 s to ~180 s once question-gen is active.
#
# ── Expected reward trajectory ──────────────────────────────────────────────
#
#   Iters  1-12  (GROUNDED_ONLY):   mean_reward 0.85→0.91, gt_match ≥ 65%
#   Iters 13-30  (SELFPLAY_RAMP):   mean_reward 0.91→0.94, sp_ratio 0%→35%
#   Iters 31-50  (SELFPLAY_RAMP):   mean_reward 0.94→0.96, sp_ratio 35%→60%
#   Iters 51-60  (SELFPLAY_RAMP):   mean_reward ~0.96,      sp_ratio ~65%
#
#   Eval (final_answer_accuracy) trajectory:
#     Iter  0 (baseline) : ~78%  (matches prior run)
#     Iter 10            : 80-81%  (+2%)
#     Iter 20            : 82-83%  (+4%)
#     Iter 40            : 83-85%  (+5-7%)
#     Iter 60            : 85-87%  (+7-9%) — upper bound for 1.5B on GSM8K
#
#   Wall-time (A100-80GB, K_q=2, K=10, N=20):
#     Model + PRM load             :  ~2 min
#     Initial eval (150 samples)   :  ~3 min
#     Train iter (Flash-Attn 2)    :  ~150 s  (K=10 vs K=8 → +25%)
#     Train iter (SDPA fallback)   :  ~220 s
#     Eval checkpoint × 12         :  ~3.5 min each = ~42 min
#     60 iterations × 150 s        :  ~150 min
#     Total (Flash active)         :  ~3.3 h
#     Total (SDPA fallback)        :  ~4.5 h
#
# ── Smoke test (~12 min) ─────────────────────────────────────────────────────
#
#   bash launch_grpo.sh \
#     --num-iterations 4 \
#     --questions-per-iter 8 \
#     --group-size 4 \
#     --eval-every 2 \
#     --eval-max-samples 20 \
#     --skip-initial-eval \
#     --save-every 0 --keep-last 0 \
#     --math-mix-ratio 0 \
#     --output-dir checkpoints/grpo_smoke
#
# =============================================================================

set -euo pipefail

# ── Flash-Attention 2 install (if missing) ────────────────────────────────────
# flash-attn requires (torch version, CUDA version, Python version) alignment.
# MAX_JOBS caps parallel compilation; prebuilt wheel installs in <30 s.
# In the prior run (grpo_20260425_151304), flash-attn was absent → SDPA fallback
# → iter times of 262-330 s once question-gen started (vs ~150 s with Flash).
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
# Persists JIT kernels across runs — avoids ~30 s recompile each launch.
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton_cache}
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

# ── Confirm attention backend ─────────────────────────────────────────────────
python - <<'PYEOF'
import sys; sys.path.insert(0, '.')
from src.utils.attn_backend import select_attn_implementation
impl = select_attn_implementation()
tag = {
    "flash_attention_2": "FAST   — Flash-Attn 2 active (O(T) memory, ~1.5-2× faster)",
    "sdpa":              "OK     — SDPA active (install flash-attn for ~2× speedup)",
    "eager":             "SLOW   — Eager fallback (install flash-attn for best speed)",
}.get(impl, impl)
print(f"[launch] attn_backend = {tag}")
PYEOF

# ── Log tee ───────────────────────────────────────────────────────────────────
RUN_NAME="grpo_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/grpo"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"

echo "[launch] run_name       = $RUN_NAME"
echo "[launch] base_model     = checkpoints/dual_task_v1"
echo "[launch] train_data     = data/sft/gsm8k_sft.jsonl + data/math/math_numeric.jsonl"
echo "[launch] eval_data      = data/sft/gsm8k_test.jsonl"
echo "[launch] log_file       = $LOG_FILE"
echo "[launch] architecture   = Two-phase self-play (K_q=2, K=10, N=20)"
echo "[launch] fixes_applied  = min-warmup↑12, selfplay-gt-thresh↑0.65, kl-coef↑0.06,"
echo "[launch]                  math-ramp-start↑18, group-size↑10, num-iters↑60"
echo "[launch] wall-time      ≈ 3.3 h (Flash active) / 4.5 h (SDPA fallback)"

# ── Train ─────────────────────────────────────────────────────────────────────
python -u scripts/run_grpo_training.py \
    --base-model            checkpoints/dual_task_v1 \
    --output-dir            checkpoints/grpo \
    --gsm8k-data            data/sft/gsm8k_sft.jsonl \
    --eval-data-path        data/sft/gsm8k_test.jsonl \
    \
    --num-iterations        60 \
    --group-size            10 \
    --q-group-size          2 \
    --questions-per-iter    20 \
    \
    --learning-rate         5e-6 \
    --max-new-tokens        1000 \
    --temperature           0.8 \
    --max-grad-norm         0.5 \
    --clip-eps              0.2 \
    --kl-coef               0.06 \
    --warmup-iters          8 \
    --min-lr-ratio          0.1 \
    \
    --difficulty-alpha      3.5 \
    --self-play-ratio       0.70 \
    \
    --math-mix-ratio        0.30 \
    --math-mix-ratio-late   0.50 \
    --math-ramp-start       18 \
    --math-max-difficulty   3 \
    \
    --overlong-filter \
    --min-warmup            12 \
    --selfplay-gt-thresh    0.65 \
    --selfplay-grounded-thresh 0.65 \
    --selfplay-step-thresh  0.68 \
    --selfplay-ramp-iters   28 \
    --grounded-floor        0.55 \
    \
    --extractor-model       Qwen/Qwen2.5-0.5B-Instruct \
    --extraction-cache      data/extraction_cache.json \
    \
    --eval-every            5 \
    --eval-max-samples      150 \
    --eval-max-new-tokens   1000 \
    --eval-pass-at-k        0 \
    --save-every            5 \
    --keep-last             4 \
    \
    --use-prm \
    --prm-model             Qwen/Qwen2.5-Math-PRM-7B \
    --run-name              "$RUN_NAME" \
    "$@" 2>&1 | tee "$LOG_FILE"
