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
# ── Architecture: Three Training Paths ──────────────────────────────────
#
#   PATH 1 — Grounded (70% of groups):
#     [GSM8K/MATH question + gold_final]
#       → K=8 solutions (batched)
#       → reward = 0.50×gold_match + 0.40×PRM_process + 0.10×format + LCCP bonus
#       → solution GRPO backward
#     Updates: solution generation only
#
#   PATH 2 — Single-question self-play (--q-group-size 1, DISABLED by default):
#     [curriculum instruction]
#       → generate_question() [@torch.no_grad — no question gradient]
#       → K=8 solutions (batched)
#       → reward = 0.40×Q_heuristic + 0.60×PRM_solution
#       → solution GRPO backward
#     Updates: solution generation only (question reward cancels in advantages)
#
#   PATH 3 — Two-phase self-play (--q-group-size 2, DEFAULT in this script):
#     [curriculum instruction]
#       → generate_questions_batched(K_q=2) + store old_log_probs
#       → for each of K_q=2 question candidates:
#            → K=8 solutions, score each, solution GRPO backward
#            → question_reward = mean(solution_rewards)
#       → question GRPO backward over K_q=2 questions
#       → optimizer.step()  ← applies both sets of gradients
#     Updates: BOTH solution AND question generation
#     Question reward: outcome-based (mean of actual downstream solution rewards),
#                      not heuristic — the model learns to ask questions that
#                      lead to high-PRM, correctly-structured solutions.
#
# ── Why these specific values ────────────────────────────────────────────
#
#   --group-size 8            K=8 batched in ONE model.generate call — near-100%
#                             GPU utilisation.  Tight mean/std estimates make GRPO
#                             advantages cleaner.  Marginal VRAM: ~4 GB extra vs K=4.
#
#   --q-group-size 2          K_q=2 question candidates per self-play group.
#                             Each candidate solved K=8 times.  Question GRPO runs
#                             once per group with 2-point reward signal.
#                             Compute overhead: +31% per iteration (self-play groups
#                             double, grounded unaffected; 30% × 2× = +30%).
#                             To keep SAME compute: use --group-size 4 --q-group-size 2.
#
#   --questions-per-iter 16   N=16 groups per step: 11 grounded + 5 self-play.
#                             Per iter solution rollouts:
#                               Grounded : 11 × 8 = 88
#                               Self-play: 5 × 2 × 8 = 80
#                               Total    : 168 (vs 128 with K_q=1)
#
#   --self-play-ratio 0.30    30% self-play (5 groups/iter).  The two-phase update
#                             extracts question-learning signal from these groups
#                             that previously only trained solution generation.
#
#   --learning-rate 5e-6      Halved from 1e-5 after iter 5 LR shock.  At 1e-5
#                             policy collapsed at warmup end (grounded_acc 87%→53%).
#                             5e-6 keeps update stable.
#
#   --warmup-iters 8          Extended from 5.  Gentler ramp lets policy adapt
#                             before hitting peak LR.
#
#   --max-grad-norm 0.5       Tighter than 1.0 default, matches PPO.  Prevents
#                             high-variance groups from spiking gradient norm.
#
#   --kl-coef 0.04            KL penalty anchors solution policy to SFT checkpoint.
#                             kl_coef=0.0 is hardcoded for the question GRPO update
#                             (question generation should stay exploratory).
#
#   --clip-eps 0.2            PPO-style IS clip: prevents policy spikes.
#
#   --difficulty-alpha 3.0    Difficulty-weighted sampling: questions where the model
#                             wins 40-60% of K solutions are sampled most often.
#
#   --math-mix-ratio 0.3      30% MATH competition problems (difficulty ≤ 3).
#   --math-mix-ratio-late 0.5 Ramps to 50% from iter 15–25 once policy is stable.
#   --math-ramp-start 15      Start of the MATH ratio ramp.
#   --math-max-difficulty 3   Level 3 ≈ AMC-10; levels 4-5 are too hard for 1.5B.
#
#   --num-iterations 50       ~120 s/iter (K_q=2 overhead) → 50 iters ≈ 100 min.
#
#   --eval-every 5            10 eval checkpoints over 50 iters.
#   --eval-max-samples 100    Fast evals (~2 min each); best_policy saved on improvement.
#
#   --save-every 5            Rolling window of 3 checkpoints (~9 GB on disk).
#   --keep-last 3
#
# ── Expected wall-time on this host (K_q=2, K=8, N=16, T_anneal=0.8→0.4) ──
#
#   Model + PRM load                              :  ~2 min
#   Initial eval (100 samples)                   :  ~2 min
#   Train iter (168 sol rollouts + 10 q rollouts):  ~120 s  (~2 min/iter)
#   Eval checkpoint × 10 (100 samples)           :  ~2 min each = 20 min
#   50 iterations × 2 min                        :  ~100 min
#   Total                                        :  ~2.5 h end-to-end
#
#   Compare vs K_q=1: 90 s/iter × 50 = 75 min + 20 min eval = ~1.8 h
#   Two-phase overhead: ~40 min extra for full question-generation gradient learning.
#
# ── Smoke test (exercises all three paths, ~8 min) ───────────────────────
#
#   Tests PATH 3 (two-phase) with minimal compute:
#
#   bash launch_grpo.sh \
#     --num-iterations 3 \
#     --questions-per-iter 8 \
#     --group-size 4 \
#     --q-group-size 2 \
#     --self-play-ratio 0.30 \
#     --max-new-tokens 200 \
#     --eval-every 2 \
#     --eval-max-samples 20 \
#     --eval-max-new-tokens 256 \
#     --math-mix-ratio 0 \
#     --skip-initial-eval \
#     --save-every 0 \
#     --keep-last 0 \
#     --output-dir checkpoints/grpo_smoke \
#     --run-name smoke_twophase
#
#   What to check in the smoke logs:
#     [iter 1] "Two-phase SP" block appears (K_q=2 question candidates)
#     [iter 1] "Q-GRPO: loss=..." appears (question backward ran)
#     [iter 1] solution GRPO loss reported per candidate
#     [eval]   combined_score > 0.0 (reward_fn fired correctly)
#
# ── Environment-variable tuning ──────────────────────────────────────────
#
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#     HF + Flash-Attn 2 fragment the allocator during generation.
#     Expandable segments recover 2-4 GB peak VRAM over a long run.
#
#   OMP_NUM_THREADS=8
#     EPYC 7V13 has 96 vCPU. 96 OpenMP threads causes thrashing; 8 is enough.
#
#   TOKENIZERS_PARALLELISM=false
#     Eliminates fork-after-thread warning from HF tokenizers.
#
#   HF_HUB_DISABLE_XET=1
#     Falls back to HTTP downloads for the 7B PRM.
#
# ── Overrides ────────────────────────────────────────────────────────────
#
#   bash launch_grpo.sh                         # default (two-phase, K_q=2)
#   bash launch_grpo.sh --q-group-size 1        # disable question GRPO
#   bash launch_grpo.sh --group-size 4 --q-group-size 2   # same compute as K_q=1 K=8
#   bash launch_grpo.sh --num-iterations 100    # longer run (~5 h)
#   bash launch_grpo.sh --no-prm               # skip 7B PRM (smoke only)

set -euo pipefail

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

# ── Pre-flight: sanity-check the GPU before a long run ───────────────────
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
echo "[launch] architecture = PATH 3 (two-phase self-play, K_q=2, K=8)"
echo "[launch] estimated wall-time ≈ 2.5 h (50 iters × ~2 min + 10 evals × ~2 min)"

# ── Train ────────────────────────────────────────────────────────────────
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
    --self-play-ratio 0.70 \
    --math-mix-ratio 0.3 \
    --math-mix-ratio-late 0.5 \
    --math-ramp-start 15 \
    --math-max-difficulty 3 \
    --overlong-filter \
    --min-warmup 10 \
    --selfplay-gt-thresh 0.55 \
    --selfplay-grounded-thresh 0.60 \
    --selfplay-step-thresh 0.65 \
    --selfplay-ramp-iters 20 \
    --grounded-floor 0.50 \
    --extractor-model Qwen/Qwen2.5-0.5B-Instruct \
    --extraction-cache data/extraction_cache.json \
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
