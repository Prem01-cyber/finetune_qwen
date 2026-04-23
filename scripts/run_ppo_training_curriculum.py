"""
PPO training with curriculum-guided dual-task rewards (single-GPU).

    python scripts/run_ppo_training_curriculum.py \\
        --base-model checkpoints/dual_task_v1 \\
        --num-iterations 10 --rollouts-per-iter 96

A Qwen2.5-Math-1.5B policy + ValueHead critic, plus the AdamW optimiser and
rollout activations, fit comfortably on a single 40GB GPU in bfloat16, so this
script deliberately does one GPU only.  It is fast, simple, and has no
distributed-training failure modes (no ZeRO hooks, no ``accelerate`` device
reshuffles, no cross-rank deadlocks).  Rollouts and PPO updates run on the
same device sequentially.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.convert_gsm8k_to_sft import parse_gsm8k_answer
from scripts.eval_sft_inference import evaluate_gsm8k
from src.rl.checkpoint_manager import CheckpointManager
from src.rl.math_environment_curriculum import CurriculumMathEnvironment
from src.rl.ppo_trainer import PPOTrainer
from src.rl.prm_scorer import ProcessRewardScorer
from src.rl.rollout_buffer import RolloutBuffer
from src.rl.training_monitor import TrainingMonitor
from src.rl.value_network import ValueHead
from src.utils.attn_backend import select_attn_implementation
from src.utils.csv_logger import CSVLogger


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Global GPU performance knobs.  These are free speed on Ampere/Hopper:
#
#   * TF32 matmul ("high") -> ~2x faster fp32 matmuls, negligible quality
#     impact for RL training.  Silences the inductor warning too.
#   * cudnn.benchmark -> lets cuDNN autotune the fastest algo for each
#     conv/input shape.  Safe because our shapes are repeatable.
#   * allow_tf32 on cudnn -> same idea on convs.
#
# Must be set before any CUDA kernel runs, so we do it at import time.
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


class TeeStream:
    """Write stream output to both terminal and a log file."""

    def __init__(self, primary, secondary):
        self.primary = primary
        self.secondary = secondary

    def write(self, data: str) -> int:
        self.primary.write(data)
        self.secondary.write(data)
        return len(data)

    def flush(self) -> None:
        self.primary.flush()
        self.secondary.flush()

    def isatty(self) -> bool:
        return getattr(self.primary, "isatty", lambda: False)()

    def fileno(self) -> int:
        return self.primary.fileno()


class CurriculumTrainingConfig:
    base_model = "checkpoints/dual_task_v1"
    # Canonical RLHF-PPO settings (InstructGPT / TRL / open-rlhf defaults),
    # tuned for a 1.5B policy + grounded-rollout stabilisation:
    #   * lr 3e-6 is the sweet-spot for LoRA-style fine-tuning on 1-2B models;
    #     1e-6 produced visibly sluggish learning and wastes compute.
    #   * clip_range 0.2 is the canonical PPO ε.
    #   * target_kl 0.05 is deliberately LOOSER than the 0.015-0.03 used in
    #     InstructGPT-scale RLHF.  Reasons:
    #       1. Grounded rollouts anchor the policy against gold GSM8K answers,
    #          so the catastrophic-collapse risk that target_kl guards against
    #          is already bounded.
    #       2. At target_kl=0.03 the per-batch approx_kl regularly crossed the
    #          1.5× trip line in epoch 1, giving us ~1/3 of the planned
    #          gradient budget per iteration.
    #     0.05 with kl_trip_multiplier=1.5 → trip at 0.075 ≈ actually lets
    #     all 3 epochs complete in typical mid-training iterations.
    #   * kl_trip_multiplier=1.5 keeps the canonical trip ratio; tune via CLI.
    learning_rate = 3e-6
    ppo_epochs = 3
    # PPO mini-batch size.  Each mini-batch runs one full fwd+bwd through
    # the 1.5B-param policy with activations kept for the backward pass.
    # Qwen2.5-1.5B × seq_len ≈ 500 × 28 layers = ~1.5 GB activations per
    # sample; B=32 overflowed an 80 GB A100 once the policy was actually
    # trainable post-audit-fix.  B=8 fits comfortably with grad
    # checkpointing on (leaves ~15 GB headroom) and does NOT slow
    # throughput much because we just run 4× more micro-batches.
    batch_size = 8
    # Gradient checkpointing on the policy trades ~30% of backward-pass
    # speed for ~40% less activation memory — essential once the
    # 1.5B-param policy is actually trainable.  Forces use_cache=False
    # on the policy (we already do that in _policy_logits_at_state).
    gradient_checkpointing = True
    clip_range = 0.2
    clip_range_vf = 0.2
    vf_coef = 0.5
    ent_coef = 0.02
    max_grad_norm = 0.5
    target_kl = 0.05
    kl_trip_multiplier = 1.5

    gamma = 1.0
    gae_lambda = 0.95

    num_rollouts_per_iter = 100
    max_question_tokens = 200
    max_solution_tokens = 500
    temperature = 0.7
    top_p = 0.9
    consensus_temperature = 0.5

    num_iterations = 10
    # Fraction of each rollout batch that is GSM8K-anchored (real question,
    # reward scored directly against the gold final answer).  0.3 is the
    # sweet spot: enough grounded signal to pull GSM8K accuracy (30 rollouts
    # × ppo_epochs = 90 gradient steps/iter against real answers), while
    # leaving 70% of the batch as self-play — where the policy trains its
    # question-generation skill (the whole point of the Self-Improvement
    # theme).  Set 0.0 to disable grounded rollouts entirely; set higher
    # if you want to sacrifice question-gen training for faster GSM8K gains.
    grounded_ratio = 0.3

    # Process Reward Model (Qwen2.5-Math-PRM) replaces the TripleVerifier
    # consensus signal on self-play rollouts.  PRM gives per-step correctness
    # probabilities against the actual question — a strictly stronger signal
    # than "do three same-model samples agree?" (they often agree on wrong
    # answers; groupthink).  Loaded in 4-bit to stay comfortably under 7 GB.
    use_prm = True
    prm_model = "Qwen/Qwen2.5-Math-PRM-7B"
    prm_load_in_4bit = True

    eval_every = 5
    # GSM8K eval at 1.5B runs at ~1 problem/s greedy, so 500 samples ≈ 8-10 min.
    # Override from CLI with --eval-max-samples / --eval-max-new-tokens if you
    # want cheaper (but noisier) signal between updates.
    eval_max_samples = 500
    eval_max_new_tokens = 512
    save_every = 1
    # torch.compile + HF .generate() with a growing KV cache is broken:
    # reduce-overhead mode uses CUDA graphs that require fixed shapes, so
    # each new-token forward either hangs in recompilation or OOMs trying
    # to stash graphs.  Leave off unless you know what you're doing.
    use_torch_compile = False

    output_dir = "checkpoints/ppo_training_curriculum"
    curriculum_checkpoint_dir = "checkpoints/ppo_training_curriculum/curriculum"
    eval_data_path = "data/sft/dual_task_val.jsonl"
    gsm8k_reference_data = "data/sft/gsm8k_sft.jsonl"

    disk_warning_gb = 5.0
    checkpoint_keep_last = 2
    checkpoint_keep_every = 100
    compress_old_logs = True

    log_dir = "logs"
    run_name = None


def load_reference_questions(path: str) -> List[str]:
    questions: List[str] = []
    file_path = Path(path)
    if not file_path.exists():
        logger.warning("Reference data %s not found; using empty reference set", path)
        return questions

    with file_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = record.get("messages", [])
            for msg in messages:
                if msg.get("role") != "user":
                    continue
                content = str(msg.get("content", ""))
                if "Problem:" in content:
                    questions.append(content.split("Problem:", 1)[1].strip())
                else:
                    questions.append(content.strip())
                break
    return questions


def load_grounded_qa_pairs(path: str) -> List[Dict[str, str]]:
    """
    Parse a GSM8K-style JSONL into ``[{"question": ..., "gold_final": ...}]``.

    Accepts two shapes:
      * Raw GSM8K rows: ``{"question": ..., "answer": "...#### 42"}``.
      * SFT-converted rows: ``{"messages": [...]}`` where the user turn
        contains ``Problem:\\n<question>`` and the assistant turn ends with
        ``Final Answer: <n>``.

    The second shape is what ``data/sft/gsm8k_sft.jsonl`` contains; the
    first shape is the upstream HuggingFace layout.
    """
    from src.sft.solution_format import extract_final_answer_numeric_str

    file_path = Path(path)
    if not file_path.exists():
        logger.warning(
            "Grounded QA data %s not found; grounded rollouts will be disabled",
            path,
        )
        return []

    qa_pairs: List[Dict[str, str]] = []
    with file_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            question = ""
            gold_final = ""

            if "question" in record and "answer" in record:
                question = str(record["question"]).strip()
                _, gold_final = parse_gsm8k_answer(str(record["answer"]))

            elif "messages" in record:
                user_text = ""
                asst_text = ""
                for msg in record["messages"]:
                    role = msg.get("role")
                    if role == "user" and not user_text:
                        user_text = str(msg.get("content", "")).strip()
                    elif role == "assistant" and not asst_text:
                        asst_text = str(msg.get("content", ""))
                if "Problem:" in user_text:
                    question = user_text.split("Problem:", 1)[1].strip()
                else:
                    question = user_text
                gold_final = extract_final_answer_numeric_str(asst_text) or ""

            if question and gold_final:
                qa_pairs.append(
                    {"question": question, "gold_final": str(gold_final).strip()}
                )

    logger.info("Loaded %d grounded (question, gold_final) pairs from %s",
                len(qa_pairs), path)
    return qa_pairs


def log_gpu_memory(stage: str) -> None:
    if not torch.cuda.is_available():
        return
    i = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
    total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
    logger.info(
        "[%s] GPU %d: %.2fGB allocated, %.2fGB reserved, %.2fGB total (%.1f%% used)",
        stage, i, allocated, reserved, total, allocated / total * 100,
    )


def _ensure_peft_tensor_parallel_shim() -> None:
    """
    PEFT <= 0.12 unconditionally imports
    ``transformers.integrations.tensor_parallel`` on attribute lookup.  Older
    transformers versions don't ship that module and the import crashes on the
    ``merge_and_unload`` path.  Install a harmless stub so the merge succeeds.
    """
    import sys as _sys
    import types

    if "transformers.integrations.tensor_parallel" not in _sys.modules:
        _sys.modules["transformers.integrations.tensor_parallel"] = types.ModuleType(
            "tensor_parallel"
        )


def initialize_models(config: CurriculumTrainingConfig):
    """Load the policy, value network and tokenizer on ``cuda:0`` (or CPU)."""
    model_path = Path(config.base_model)
    is_adapter = (model_path / "adapter_config.json").exists()

    if is_adapter:
        meta_file = model_path / "pipeline_meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            base_model_name = meta.get("base_model", "Qwen/Qwen2.5-Math-1.5B-Instruct")
        else:
            base_model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    else:
        base_model_name = config.base_model

    log_gpu_memory("Before model loading")

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Force right-padding: the rollout buffer re-pads single-example state
    # tensors right-aligned via _pad_2d, and PPOTrainer picks the "last
    # non-pad" logit via attention_mask.sum(dim=1) - 1 — both assume right
    # padding.  If any downstream call does tokenizer(..., padding=True)
    # on a batch with the default left-padding, the last-token index
    # picks a pad position and produces silently wrong log-probs.
    tokenizer.padding_side = "right"
    if tokenizer.chat_template is None and is_adapter:
        logger.info("Chat template not found in adapter, loading from base model")
        base_tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, trust_remote_code=True
        )
        if base_tokenizer.chat_template is not None:
            tokenizer.chat_template = base_tokenizer.chat_template

    _ensure_peft_tensor_parallel_shim()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Pick the fastest attention backend the container supports.
    # flash_attention_2 is ~1.5-2.5x faster than SDPA on Ampere/Hopper
    # AND turns attention memory from O(T^2) to O(T) per layer — which
    # is what lets us optionally turn gradient checkpointing OFF (see
    # the "gradient_checkpointing" block below) and claw back the ~30%
    # backward-pass slowdown it introduces.  Falls back to SDPA if the
    # flash-attn wheel is not on the container, so this flag is always
    # safe to set.
    attn_impl = select_attn_implementation()

    policy_load_kwargs = {
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
        "device_map": {"": "cpu"},
        "attn_implementation": attn_impl,
    }

    if is_adapter:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, **policy_load_kwargs
        )
        policy = PeftModel.from_pretrained(
            base_model, config.base_model
        ).merge_and_unload()
    else:
        policy = AutoModelForCausalLM.from_pretrained(
            config.base_model, **policy_load_kwargs
        )

    # CRITICAL: PeftModel.from_pretrained sets requires_grad=False on every
    # base-model parameter (only the LoRA adapter stays trainable).
    # merge_and_unload() folds the LoRA deltas back into the base weights
    # and strips the wrapper, but it does NOT restore requires_grad.
    # Without this loop, the PPO optimiser ends up holding zero policy
    # parameters (only the value-head MLP), the policy never updates, and
    # you get byte-identical eval accuracy across iterations — the exact
    # failure mode we observed in the run that produced this fix.
    trainable_before = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    for param in policy.parameters():
        param.requires_grad_(True)
    trainable_after = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in policy.parameters())
    logger.info(
        "Policy trainable params: %s/%s (%.1f%%) — before unfreeze: %s "
        "[if this was 0, the policy was silently frozen before this fix]",
        f"{trainable_after:,}",
        f"{total_params:,}",
        100.0 * trainable_after / max(total_params, 1),
        f"{trainable_before:,}",
    )
    if trainable_before == 0 and trainable_after > 0:
        logger.warning(
            "Policy was loaded fully frozen (requires_grad=False on every "
            "param) — this is the merge_and_unload + PEFT interaction bug. "
            "Now fixed for this run, but any prior checkpoints from this "
            "codebase were trained on value-head updates only and should "
            "be treated as the SFT baseline, not a PPO-improved policy."
        )

    policy = policy.to(device)

    # Gradient checkpointing: drop intermediate activations on the forward
    # pass and recompute them during backward.  For a 28-layer Qwen2.5-1.5B
    # with B=8, seq_len≈500 this cuts peak activation memory roughly 40%
    # (from ~20 GB to ~12 GB) at the cost of ~30% longer backward.
    # Required combination for HF models:
    #   1. gradient_checkpointing_enable() — registers the hooks
    #   2. use_cache=False on every forward — cache + checkpointing is
    #      incompatible (we already force this in _policy_logits_at_state
    #      and in rollouts).
    #
    # Smart default: when flash_attention_2 is active, attention memory
    # is already O(T) instead of O(T^2), which buys back most of what
    # gradient checkpointing saves.  In that case we leave it OFF by
    # default so we don't pay the ~30% backward-pass cost.  The user
    # can still force it on/off via --grad-checkpoint / --no-grad-checkpoint.
    flash_active = attn_impl == "flash_attention_2"
    grad_ckpt_requested = getattr(config, "gradient_checkpointing", True)
    if grad_ckpt_requested and not args.grad_checkpoint_explicit and flash_active:
        grad_ckpt_effective = False
        logger.info(
            "Flash-Attn 2 active — leaving gradient checkpointing OFF by "
            "default (Flash already gives O(T) attention memory).  Pass "
            "--grad-checkpoint to force it on if you hit OOM."
        )
    else:
        grad_ckpt_effective = grad_ckpt_requested

    if grad_ckpt_effective:
        policy.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(policy, "config"):
            policy.config.use_cache = False
        logger.info(
            "Gradient checkpointing ENABLED on policy "
            "(use_reentrant=False, use_cache=False)."
        )
    else:
        logger.info(
            "Gradient checkpointing DISABLED on policy (attn=%s).", attn_impl
        )

    log_gpu_memory("After policy loaded")

    value = ValueHead(base_model_name, model_device_map={"": "cpu"})
    value.backbone = value.backbone.to(device)
    value.value_head = value.value_head.to(device)
    log_gpu_memory("After ValueHead loaded")

    if config.use_torch_compile:
        try:
            logger.info(
                "Compiling policy with torch.compile(mode='default')"
                " — first forward may stall several minutes"
            )
            # NB: avoid mode='reduce-overhead' (CUDA graphs) here — HF
            # .generate()'s growing KV cache changes shapes every step and
            # triggers endless recompilation or an outright hang.
            policy = torch.compile(policy, mode="default")
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("torch.compile failed: %s. Continuing without.", exc)

    return policy, value, tokenizer, device


def _policy_weight_fingerprint(policy) -> Dict[str, float]:
    """
    Return a cheap fingerprint of the live policy weights so we can
    confirm from logs that eval is hitting the *updated* model and not a
    stale copy.

    Crucially we sample a **trainable** parameter (requires_grad=True) —
    the previous implementation picked the last named parameter, which
    under PEFT + merge_and_unload ends up being ``model.norm.weight`` or
    ``lm_head.weight``, both of which can be frozen even while other
    params train.  A fingerprint over a frozen param gives false "no
    drift" readings even when PPO is learning correctly.

    We also fingerprint the input embedding (usually trainable and large)
    so drift shows up in both a small 1-D vector (fast) and a large 2-D
    matrix (canonical).
    """
    import hashlib

    fingerprint: Dict[str, float] = {}

    # 1) Input embedding — canonical "did anything change" tensor.
    try:
        embed_param = policy.get_input_embeddings().weight
        embed = embed_param.detach()
        fingerprint["embed_l2"] = float(embed.float().norm().item())
        fingerprint["embed_sha8"] = int(
            hashlib.sha1(
                embed.float().cpu().contiguous().numpy().tobytes()[:4096]
            ).hexdigest()[:8],
            16,
        )
        fingerprint["embed_requires_grad"] = float(embed_param.requires_grad)
    except Exception as exc:  # pragma: no cover - best effort
        logger.debug("embed fingerprint failed: %s", exc)

    # 2) First *trainable* named parameter — guaranteed to move across
    # iterations if PPO is actually updating the policy.  Falls back to
    # the last named param only if nothing is trainable (which itself is
    # a red flag worth logging).
    try:
        picked_name, picked_param = None, None
        for name, param in policy.named_parameters():
            if param.requires_grad:
                picked_name, picked_param = name, param
                break
        if picked_param is None:
            fingerprint["trainable_param_name"] = "<none_trainable>"
            for name, param in policy.named_parameters():
                picked_name, picked_param = name, param
        if picked_param is not None:
            w = picked_param.detach().float()
            fingerprint["trainable_param_name"] = (  # type: ignore[assignment]
                picked_name
            )
            fingerprint["trainable_param_l2"] = float(w.norm().item())
    except Exception as exc:  # pragma: no cover - best effort
        logger.debug("trainable-param fingerprint failed: %s", exc)

    return fingerprint


def evaluate_policy(
    policy,
    tokenizer,
    eval_data_path: str,
    max_samples: int = 500,
    max_new_tokens: int = 512,
) -> Dict[str, float]:
    logger.info(
        "Evaluating policy on GSM8K (max_samples=%d, max_new_tokens=%d)",
        max_samples,
        max_new_tokens,
    )
    # Ensure eval runs deterministically — HF models load in .train() mode
    # by default, and we want dropout/BN-style state paths disabled even
    # though Qwen2.5 currently has no dropout (future-proofing).
    policy.eval()
    fp = _policy_weight_fingerprint(policy)
    if fp:
        logger.info(
            "Eval policy fingerprint: embed_l2=%.4f embed_sha8=%08x "
            "embed_trainable=%s | trainable_probe=%s l2=%.4f  "
            "(live object; values should drift across iterations — if not, "
            "the optimizer is not updating the eval'd model)",
            fp.get("embed_l2", float("nan")),
            int(fp.get("embed_sha8", 0)),
            bool(fp.get("embed_requires_grad", 0.0)),
            fp.get("trainable_param_name", "?"),
            fp.get("trainable_param_l2", float("nan")),
        )
    results = evaluate_gsm8k(
        model=policy,
        tokenizer=tokenizer,
        data_path=eval_data_path,
        max_samples=max_samples,
        max_new_tokens=max_new_tokens,
    )
    logger.info(
        "GSM8K Accuracy: %.2f%% (%d/%d)",
        results["accuracy"] * 100.0,
        results["correct"],
        results["total"],
    )
    return results


def aggregate_curriculum_metrics(trajectories: List) -> Dict[str, object]:
    topic_counts: Dict[str, int] = {}
    topic_successes: Dict[str, int] = {}
    topic_difficulty: Dict[str, List[float]] = {}
    # Self-play only — excludes the synthetic "grounded_gsm8k" bucket,
    # which would otherwise skew topic-distribution and question-quality
    # stats (grounded rollouts don't train question generation).
    selfplay_topic_counts: Dict[str, int] = {}

    topic_match_scores: List[float] = []
    difficulty_match_scores: List[float] = []
    clarity_scores: List[float] = []
    solvability_scores: List[float] = []
    novelty_scores: List[float] = []
    question_rewards: List[float] = []
    selfplay_question_rewards: List[float] = []
    solution_rewards: List[float] = []
    combined_rewards: List[float] = []
    pre_expert_rewards: List[float] = []
    expert_modifiers: List[float] = []
    expert_phase_counts: Dict[str, int] = {}
    source_counts: Dict[str, int] = {}
    replay_added_count = 0
    prm_mean_scores: List[float] = []
    prm_min_scores: List[float] = []
    prm_final_scores: List[float] = []
    prm_degraded_count = 0
    prm_sample_count = 0

    for trajectory in trajectories:
        meta = trajectory.metadata
        topic = str(meta["target_topic"])
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
        topic_successes[topic] = topic_successes.get(topic, 0) + int(
            meta["consensus_achieved"] and meta["primary_matches_majority"]
        )
        topic_difficulty.setdefault(topic, []).append(
            float(meta["estimated_difficulty"])
        )

        topic_match_scores.append(float(meta["topic_match_score"]))
        difficulty_match_scores.append(
            1.0 - abs(float(meta["estimated_difficulty"]) - float(meta["target_difficulty"]))
        )
        clarity_scores.append(float(meta["clarity_score"]))
        solvability_scores.append(1.0 if bool(meta["sympy_verified"]) else 0.0)
        novelty_scores.append(float(meta["novelty_scores"]["combined"]))
        question_rewards.append(float(meta["question_reward"]))
        # Only self-play rollouts exercise the question-generation policy;
        # grounded rollouts just solve known GSM8K questions and do not
        # train the question head (their ``question_reward`` metadata is
        # pinned to 0.0 — see rollout_grounded_trajectory).  We therefore
        # track the self-play subset separately so
        # ``avg_question_reward_selfplay`` actually reflects how the
        # policy's question-gen skill is improving over iterations.
        if str(meta.get("rollout_source", "fresh")) != "grounded":
            selfplay_question_rewards.append(float(meta["question_reward"]))
            selfplay_topic_counts[topic] = selfplay_topic_counts.get(topic, 0) + 1
        solution_rewards.append(float(meta["solution_reward"]))
        combined_rewards.append(float(meta["combined_reward"]))
        pre_expert_rewards.append(
            float(meta.get("pre_expert_reward", meta["combined_reward"]))
        )
        expert_modifiers.append(float(meta.get("expert_reward_modifier", 0.0)))
        phase = str(meta.get("expert_phase", "unknown"))
        expert_phase_counts[phase] = expert_phase_counts.get(phase, 0) + 1
        source = str(meta.get("rollout_source", "fresh"))
        source_counts[source] = source_counts.get(source, 0) + 1
        replay_added_count += int(bool(meta.get("replay_added", False)))

        # PRM stats live inside reward_breakdown on grounded rollouts and
        # inside reward_breakdown["solution_metrics"] on self-play rollouts.
        rb = meta.get("reward_breakdown", {})
        prm_mean = None
        prm_min = None
        prm_final = None
        prm_degraded = None
        if isinstance(rb, dict):
            if "prm_mean_score" in rb:
                prm_mean = float(rb.get("prm_mean_score", 0.0))
                prm_degraded = bool(rb.get("prm_degraded", True))
            sol = rb.get("solution_metrics", {}) if isinstance(
                rb.get("solution_metrics"), dict
            ) else {}
            if "prm_mean_score" in sol:
                prm_mean = float(sol.get("prm_mean_score", prm_mean or 0.0))
                prm_min = float(sol.get("prm_min_score", 0.0))
                prm_final = float(sol.get("prm_final_score", 0.0))
                prm_degraded = bool(sol.get("prm_degraded", prm_degraded or False))
        if prm_mean is not None:
            prm_sample_count += 1
            prm_mean_scores.append(prm_mean)
            if prm_min is not None:
                prm_min_scores.append(prm_min)
            if prm_final is not None:
                prm_final_scores.append(prm_final)
            if prm_degraded:
                prm_degraded_count += 1

    per_topic_success = {
        topic: (topic_successes.get(topic, 0) / max(1, count))
        for topic, count in topic_counts.items()
    }
    per_topic_difficulty = {
        topic: float(sum(values) / max(1, len(values)))
        for topic, values in topic_difficulty.items()
    }

    def _mean_reward_for(source: str) -> float:
        vals = [
            float(t.metadata["combined_reward"])
            for t in trajectories
            if str(t.metadata.get("rollout_source", "fresh")) == source
        ]
        return float(sum(vals) / max(1, len(vals)))

    def _shannon_entropy(counts: Dict[str, int]) -> float:
        total = sum(counts.values())
        if total <= 0:
            return 0.0
        from math import log
        ent = 0.0
        for c in counts.values():
            if c <= 0:
                continue
            p = c / total
            ent -= p * log(p)
        return float(ent)

    selfplay_topic_entropy = _shannon_entropy(selfplay_topic_counts)

    return {
        "topics_in_sweet_spot": len(
            [s for s in per_topic_success.values() if 0.4 <= s <= 0.7]
        ),
        "avg_difficulty": float(
            sum(difficulty_match_scores) / max(1, len(difficulty_match_scores))
        ),
        "topic_diversity": len(topic_counts),
        "per_topic_success": per_topic_success,
        "per_topic_difficulty": per_topic_difficulty,
        "avg_topic_match": float(sum(topic_match_scores) / max(1, len(topic_match_scores))),
        "avg_difficulty_match": float(
            sum(difficulty_match_scores) / max(1, len(difficulty_match_scores))
        ),
        "avg_clarity": float(sum(clarity_scores) / max(1, len(clarity_scores))),
        "avg_solvability": float(sum(solvability_scores) / max(1, len(solvability_scores))),
        "avg_novelty": float(sum(novelty_scores) / max(1, len(novelty_scores))),
        "avg_question_reward": float(sum(question_rewards) / max(1, len(question_rewards))),
        # This is the real signal for "is question generation improving?"
        # — grounded rollouts are excluded so we see only the policy's
        # actual question-gen performance.
        "avg_question_reward_selfplay": (
            float(sum(selfplay_question_rewards) / len(selfplay_question_rewards))
            if selfplay_question_rewards else 0.0
        ),
        "selfplay_topic_entropy": selfplay_topic_entropy,
        "selfplay_topic_diversity": len(selfplay_topic_counts),
        "avg_solution_reward": float(sum(solution_rewards) / max(1, len(solution_rewards))),
        "avg_combined_reward": float(sum(combined_rewards) / max(1, len(combined_rewards))),
        "avg_pre_expert_reward": float(
            sum(pre_expert_rewards) / max(1, len(pre_expert_rewards))
        ),
        "avg_expert_modifier": float(sum(expert_modifiers) / max(1, len(expert_modifiers))),
        "expert_phase_counts": expert_phase_counts,
        "source_counts": source_counts,
        "replay_added_count": replay_added_count,
        "fresh_mean_reward": _mean_reward_for("fresh"),
        "replay_mean_reward": _mean_reward_for("replay"),
        "prm_sample_count": prm_sample_count,
        "prm_degraded_count": prm_degraded_count,
        "prm_mean_of_means": (
            float(sum(prm_mean_scores) / len(prm_mean_scores))
            if prm_mean_scores else 0.0
        ),
        "prm_mean_of_mins": (
            float(sum(prm_min_scores) / len(prm_min_scores))
            if prm_min_scores else 0.0
        ),
        "prm_mean_of_finals": (
            float(sum(prm_final_scores) / len(prm_final_scores))
            if prm_final_scores else 0.0
        ),
    }


def save_iteration_results(
    iteration: int,
    trajectories: List,
    metrics: Dict[str, object],
    config: CurriculumTrainingConfig,
) -> None:
    output_dir = Path(config.output_dir) / f"iteration_{iteration:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "trajectories.jsonl").open("w", encoding="utf-8") as handle:
        for idx, trajectory in enumerate(trajectories):
            payload = {
                "trajectory_id": idx,
                "total_reward": trajectory.total_reward,
                "length": len(trajectory),
                "metadata": trajectory.metadata,
            }
            handle.write(json.dumps(payload) + "\n")

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO with curriculum-guided dual-task rewards (single GPU)"
    )
    parser.add_argument("--base-model", type=str, default="checkpoints/dual_task_v1")
    parser.add_argument(
        "--output-dir", type=str, default="checkpoints/ppo_training_curriculum"
    )
    parser.add_argument("--num-iterations", type=int, default=10)
    parser.add_argument("--rollouts-per-iter", type=int, default=100)
    parser.add_argument(
        "--eval-data-path", type=str, default="data/sft/dual_task_val.jsonl"
    )
    parser.add_argument(
        "--gsm8k-reference-data", type=str, default="data/sft/gsm8k_sft.jsonl"
    )
    parser.add_argument("--skip-initial-eval", action="store_true")
    parser.add_argument(
        "--eval-every",
        type=int,
        default=5,
        help="Run GSM8K eval every N iterations (default: 5).  Raise to 10 for "
        "a long run where eval wall-time dominates.",
    )
    parser.add_argument(
        "--eval-max-samples",
        type=int,
        default=500,
        help="GSM8K problems per eval (default: 500, ≈8-10 min on 1.5B).  "
        "Drop to 250 for ~2x faster eval with noisier signal.",
    )
    parser.add_argument(
        "--eval-max-new-tokens",
        type=int,
        default=512,
        help="Max generated tokens per GSM8K answer (default: 512).  Drop to "
        "384 for ~20%% cheaper eval; GSM8K rarely needs more.",
    )
    parser.add_argument("--disk-warning-gb", type=float, default=5.0)
    parser.add_argument("--checkpoint-keep-last", type=int, default=2)
    parser.add_argument("--checkpoint-keep-every", type=int, default=100)
    parser.add_argument("--no-compress-old-logs", action="store_true")
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Opt-in to torch.compile on the policy (slow warm-up, often "
        "broken with HF .generate(); off by default).",
    )
    parser.add_argument(
        "--grounded-ratio",
        type=float,
        default=0.3,
        help="Fraction of each rollout batch that is GSM8K-anchored "
        "(real question, reward from gold final answer).  Default 0.3 — "
        "enough to pull GSM8K accuracy while leaving 70%% of the batch "
        "as self-play, which is where the policy learns QUESTION generation "
        "(the self-improvement loop).  Set 0 for pure self-play, or raise "
        "to 0.5-0.7 to bias toward solving-only fine-tuning.",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.05,
        help="PPO KL-divergence early-stopping target.  Epoch aborts when "
        "approx_kl > kl_trip_multiplier * target_kl.  Default 0.05 (looser "
        "than the 0.015-0.03 canonical RLHF range) because grounded "
        "rollouts already bound collapse risk and a tighter threshold was "
        "cutting most iterations to 1 of 3 planned epochs.  Drop to 0.03 "
        "if you see policy_loss oscillating wildly.",
    )
    parser.add_argument(
        "--kl-trip-multiplier",
        type=float,
        default=1.5,
        help="Multiplier applied to --target-kl for the early-stop trip line. "
        "Canonical RLHF uses 1.5.  Raise to 2.0-2.5 if you want to almost "
        "never trip early (pairs well with a lower --target-kl).  Final "
        "trip threshold = target_kl * kl_trip_multiplier (logged per iter).",
    )
    parser.add_argument(
        "--ppo-epochs",
        type=int,
        default=3,
        help="Number of gradient epochs over each rollout buffer.  Default 3.",
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=0.2,
        help="PPO policy ratio clip ε.  Default 0.2 (canonical).",
    )
    parser.add_argument(
        "--clip-range-vf",
        type=float,
        default=0.2,
        help="PPO value-function clip ε.  Default 0.2.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="PPO mini-batch size.  Default 8: a 1.5B-param policy with "
        "bf16 weights + grads + AdamW states + per-layer backward-pass "
        "activations already uses ~25 GB at this size; B=32 OOMs an 80 GB "
        "A100.  Throughput barely changes because we simply run 4× more "
        "micro-batches.  Drop to 4 if you add more resident modules (e.g. "
        "PRM in bf16 instead of 4-bit).",
    )
    parser.add_argument(
        "--no-grad-checkpoint",
        dest="gradient_checkpointing",
        action="store_false",
        default=True,
        help="Disable gradient checkpointing on the policy.  Default: ON "
        "(saves ~40%% activation memory for ~30%% slower backward).  "
        "With flash_attention_2 active, checkpointing is ALSO auto-disabled "
        "because Flash already gives O(T) attention memory — use "
        "--grad-checkpoint to force it back on if you still OOM.",
    )
    parser.add_argument(
        "--grad-checkpoint",
        dest="grad_checkpoint_explicit",
        action="store_true",
        default=False,
        help="Explicitly force gradient checkpointing on even when "
        "flash_attention_2 is active.  Useful if you bump --batch-size "
        "high enough that Flash alone isn't enough.",
    )
    parser.add_argument(
        "--use-prm",
        dest="use_prm",
        action="store_true",
        default=True,
        help="Use Qwen2.5-Math-PRM as the self-play correctness signal "
        "(replaces TripleVerifier consensus).  Default: on.",
    )
    parser.add_argument(
        "--no-prm",
        dest="use_prm",
        action="store_false",
        help="Disable the PRM and fall back to the legacy consensus-based "
        "self-play reward.",
    )
    parser.add_argument(
        "--prm-model",
        type=str,
        default="Qwen/Qwen2.5-Math-PRM-7B",
        help="HuggingFace repo id of the Process Reward Model.",
    )
    parser.add_argument(
        "--prm-no-4bit",
        dest="prm_load_in_4bit",
        action="store_false",
        default=True,
        help="Load the PRM in full bf16 (~14 GB) instead of 4-bit (~5 GB).",
    )
    parser.add_argument(
        "--run-name", type=str, default=None, help="Optional run name for logging"
    )
    args = parser.parse_args()

    config = CurriculumTrainingConfig()
    config.base_model = args.base_model
    config.output_dir = args.output_dir
    config.num_iterations = args.num_iterations
    config.num_rollouts_per_iter = args.rollouts_per_iter
    config.eval_data_path = args.eval_data_path
    config.gsm8k_reference_data = args.gsm8k_reference_data
    config.curriculum_checkpoint_dir = str(Path(args.output_dir) / "curriculum")
    config.run_name = args.run_name
    config.disk_warning_gb = float(args.disk_warning_gb)
    config.checkpoint_keep_last = max(1, int(args.checkpoint_keep_last))
    config.checkpoint_keep_every = max(1, int(args.checkpoint_keep_every))
    config.compress_old_logs = not args.no_compress_old_logs
    config.eval_every = max(1, int(args.eval_every))
    config.eval_max_samples = max(1, int(args.eval_max_samples))
    config.eval_max_new_tokens = max(32, int(args.eval_max_new_tokens))
    config.grounded_ratio = max(0.0, min(1.0, float(args.grounded_ratio)))
    config.use_prm = bool(args.use_prm)
    config.prm_model = str(args.prm_model)
    config.prm_load_in_4bit = bool(args.prm_load_in_4bit)
    config.target_kl = max(1e-4, float(args.target_kl))
    config.kl_trip_multiplier = max(1.0, float(args.kl_trip_multiplier))
    config.ppo_epochs = max(1, int(args.ppo_epochs))
    config.clip_range = max(1e-3, float(args.clip_range))
    config.clip_range_vf = max(1e-3, float(args.clip_range_vf))
    config.batch_size = max(1, int(args.batch_size))
    config.gradient_checkpointing = bool(args.gradient_checkpointing)
    if args.torch_compile:
        config.use_torch_compile = True

    base_seed = 1234
    random.seed(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)

    logger_csv = CSVLogger(
        project="ppo-curriculum",
        run_name=config.run_name or f"curriculum_{datetime.now():%Y%m%d_%H%M%S}",
        log_dir=config.log_dir,
        config=vars(config),
        log_detailed=True,
    )
    console_log_path = Path(logger_csv.log_path) / "console_output.log"
    console_log_file = console_log_path.open("a", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, console_log_file)
    sys.stderr = TeeStream(original_stderr, console_log_file)
    logger.info("Full console output is being captured at %s", console_log_path)

    try:
        policy, value, tokenizer, device = initialize_models(config)
        reference_questions = load_reference_questions(config.gsm8k_reference_data)
        grounded_qa_pairs = (
            load_grounded_qa_pairs(config.gsm8k_reference_data)
            if config.grounded_ratio > 0.0
            else []
        )
        if config.grounded_ratio > 0.0 and not grounded_qa_pairs:
            logger.warning(
                "grounded_ratio=%.2f requested but no QA pairs were loaded from %s; "
                "falling back to pure self-play",
                config.grounded_ratio, config.gsm8k_reference_data,
            )
            config.grounded_ratio = 0.0

        prm_scorer = None
        if config.use_prm:
            try:
                log_gpu_memory("Before PRM load")
                prm_scorer = ProcessRewardScorer(
                    model_name=config.prm_model,
                    device=device,
                    load_in_4bit=config.prm_load_in_4bit,
                )
                log_gpu_memory("After PRM load")
            except Exception as exc:
                logger.error(
                    "PRM load failed (%s); falling back to legacy "
                    "consensus-based self-play reward.  To silence this, "
                    "pass --no-prm.", exc,
                )
                prm_scorer = None

        math_env = CurriculumMathEnvironment(
            policy_model=policy,
            value_model=value,
            tokenizer=tokenizer,
            reference_questions=reference_questions,
            grounded_qa_pairs=grounded_qa_pairs,
            prm_scorer=prm_scorer,
            curriculum_checkpoint_dir=config.curriculum_checkpoint_dir,
            max_question_tokens=config.max_question_tokens,
            max_solution_tokens=config.max_solution_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            consensus_temperature=config.consensus_temperature,
            device=device,
        )
        training_monitor = TrainingMonitor(
            output_dir=config.output_dir,
            disk_warning_gb=config.disk_warning_gb,
        )
        checkpoint_manager = CheckpointManager(
            output_dir=config.output_dir,
            keep_last_n=config.checkpoint_keep_last,
            keep_every_n=config.checkpoint_keep_every,
            compress_old_logs=config.compress_old_logs,
        )

        logger.info(
            "Using single-GPU PPO trainer on %s | target_kl=%.4f × %.2f → "
            "trip@%.4f | ppo_epochs=%d | clip_range=%.2f",
            device,
            config.target_kl,
            config.kl_trip_multiplier,
            config.target_kl * config.kl_trip_multiplier,
            config.ppo_epochs,
            config.clip_range,
        )
        ppo_trainer = PPOTrainer(
            policy_model=policy,
            value_model=value,
            tokenizer=tokenizer,
            learning_rate=config.learning_rate,
            ppo_epochs=config.ppo_epochs,
            batch_size=config.batch_size,
            clip_range=config.clip_range,
            clip_range_vf=config.clip_range_vf,
            vf_coef=config.vf_coef,
            ent_coef=config.ent_coef,
            max_grad_norm=config.max_grad_norm,
            target_kl=config.target_kl,
            kl_trip_multiplier=config.kl_trip_multiplier,
        )

        if args.skip_initial_eval:
            logger.info(
                "\n%s\nSKIPPING INITIAL EVALUATION (--skip-initial-eval)\n%s",
                "=" * 80, "=" * 80,
            )
            initial_eval = {"accuracy": 0.0}
            best_accuracy = 0.0
        else:
            logger.info(
                "\n%s\nINITIAL EVALUATION (Iteration 0)\n%s", "=" * 80, "=" * 80
            )
            initial_eval = evaluate_policy(
                policy,
                tokenizer,
                config.eval_data_path,
                max_samples=config.eval_max_samples,
                max_new_tokens=config.eval_max_new_tokens,
            )
            best_accuracy = float(initial_eval.get("accuracy", 0.0))
            logger_csv.log(
                {
                    "eval/accuracy": initial_eval.get("accuracy", 0.0),
                    "eval/correct": initial_eval.get("correct", 0),
                    "eval/total": initial_eval.get("total", 0),
                },
                step=0,
            )

        for iteration in range(1, config.num_iterations + 1):
            iteration_start = time.perf_counter()
            logger.info(
                "\n%s\nITERATION %d/%d\n%s",
                "=" * 80, iteration, config.num_iterations, "=" * 80,
            )
            current_phase = math_env.expert_panel.get_current_expert(
                math_env.curriculum_manager.current_iteration
            )
            logger.info(
                "Active expert phase: %s (%s)",
                current_phase.name, current_phase.description,
            )

            rollout_start = time.perf_counter()
            trajectories = math_env.collect_rollouts(
                num_trajectories=config.num_rollouts_per_iter,
                verbose=True,
                grounded_ratio=config.grounded_ratio,
            )
            rollout_seconds = time.perf_counter() - rollout_start

            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
            np.random.seed(base_seed + iteration)
            rollout_buffer = RolloutBuffer(
                gamma=config.gamma,
                gae_lambda=config.gae_lambda,
                pad_token_id=int(pad_id),
            )
            for trajectory in trajectories:
                rollout_buffer.add_trajectory(trajectory)

            buffer_stats = rollout_buffer.get_stats()
            curriculum_stats = aggregate_curriculum_metrics(trajectories)
            replay_stats = math_env.replay_buffer.get_buffer_stats(
                current_iteration=math_env.curriculum_manager.current_iteration
            )

            # One-line summary of the two training signals so it's obvious
            # at a glance whether question generation is improving alongside
            # solution generation.
            logger.info(
                "Rewards: Q_selfplay=%.3f (n=%d, topic_entropy=%.2f, topics=%d)  "
                "Sol=%.3f  Combined=%.3f  grounded_acc=%.2f",
                curriculum_stats["avg_question_reward_selfplay"],
                len([t for t in trajectories
                     if str(t.metadata.get("rollout_source", "fresh")) != "grounded"]),
                curriculum_stats["selfplay_topic_entropy"],
                curriculum_stats["selfplay_topic_diversity"],
                curriculum_stats["avg_solution_reward"],
                curriculum_stats["avg_combined_reward"],
                math_env.last_grounded_stats.get("accuracy", 0.0),
            )

            logger.info("Running PPO update...")
            train_start = time.perf_counter()
            training_metrics = ppo_trainer.train_step(rollout_buffer)
            train_seconds = time.perf_counter() - train_start

            update_steps_done = int(training_metrics.get("update_steps", 0.0))
            update_steps_planned = int(
                training_metrics.get("update_steps_planned", 0.0)
            )
            early_stop_epoch = int(training_metrics.get("early_stop_epoch", -1.0))
            early_stop_frac = (
                1.0 - (update_steps_done / update_steps_planned)
                if update_steps_planned > 0
                else 0.0
            )
            early_stop_tag = (
                "full" if early_stop_epoch < 0
                else f"KL-stopped@epoch{early_stop_epoch + 1}/{config.ppo_epochs}"
            )
            logger.info(
                "PPO update metrics: policy_loss=%.4f value_loss=%.4f "
                "entropy=%.4f approx_kl=%.4f (trip@%.4f) clip_fraction=%.4f "
                "updates=%d/%d (%.0f%% budget used) | %s",
                training_metrics["policy_loss"],
                training_metrics["value_loss"],
                training_metrics["entropy"],
                training_metrics["approx_kl"],
                training_metrics.get(
                    "kl_trip_threshold",
                    config.target_kl * config.kl_trip_multiplier,
                ),
                training_metrics["clip_fraction"],
                update_steps_done,
                update_steps_planned,
                100.0 * (1.0 - early_stop_frac),
                early_stop_tag,
            )

            eval_start = time.perf_counter()
            if iteration % config.eval_every == 0:
                eval_results = evaluate_policy(
                    policy,
                    tokenizer,
                    config.eval_data_path,
                    max_samples=config.eval_max_samples,
                    max_new_tokens=config.eval_max_new_tokens,
                )
                best_accuracy = max(
                    best_accuracy, float(eval_results.get("accuracy", 0.0))
                )
            else:
                eval_results = {}
            eval_seconds = time.perf_counter() - eval_start

            save_start = time.perf_counter()
            cleanup_metrics = {"deleted_checkpoints": 0, "compressed_logs": 0}
            if iteration % config.save_every == 0:
                checkpoint_manager.save_checkpoint(
                    iteration=iteration, trainer=ppo_trainer
                )
                cleanup_metrics = checkpoint_manager.cleanup_old_checkpoints(
                    current_iteration=iteration
                )
            save_seconds = time.perf_counter() - save_start

            total_seconds = time.perf_counter() - iteration_start
            timing_metrics = {
                "rollout_seconds": float(rollout_seconds),
                "train_seconds": float(train_seconds),
                "eval_seconds": float(eval_seconds),
                "save_seconds": float(save_seconds),
                "total_seconds": float(total_seconds),
                "num_rollouts": float(len(trajectories)),
                "estimated_tokens_generated": float(
                    len(trajectories)
                    * (config.max_question_tokens + 4 * config.max_solution_tokens)
                ),
            }
            throughput_metrics = training_monitor.log_iteration_timing(
                iteration=iteration, timings=timing_metrics
            )
            disk_metrics = training_monitor.check_disk_space()
            gpu_metrics = training_monitor.log_gpu_utilization(
                gpu_ids=list(range(torch.cuda.device_count()))
            )
            total_s = max(timing_metrics["total_seconds"], 1e-6)
            rollout_pct = 100.0 * timing_metrics["rollout_seconds"] / total_s
            train_pct = 100.0 * timing_metrics["train_seconds"] / total_s
            eval_pct = 100.0 * timing_metrics["eval_seconds"] / total_s
            save_pct = 100.0 * timing_metrics["save_seconds"] / total_s
            logger.info(
                "Timing breakdown: rollout=%.1fs (%.0f%%)  train=%.1fs (%.0f%%)  "
                "eval=%.1fs (%.0f%%)  save=%.1fs (%.0f%%)  total=%.1fs",
                timing_metrics["rollout_seconds"], rollout_pct,
                timing_metrics["train_seconds"], train_pct,
                timing_metrics["eval_seconds"], eval_pct,
                timing_metrics["save_seconds"], save_pct,
                timing_metrics["total_seconds"],
            )

            all_metrics = {
                "iteration": iteration,
                "buffer": buffer_stats,
                "curriculum": curriculum_stats,
                "training": training_metrics,
                "eval": eval_results,
                "timing": timing_metrics,
                "throughput": throughput_metrics,
                "disk": disk_metrics,
                "gpu": gpu_metrics,
                "checkpoint_cleanup": cleanup_metrics,
                "curriculum_state": math_env.curriculum_manager.get_curriculum_stats(),
                "replay_buffer": replay_stats,
                "rollout_mix": dict(math_env.last_rollout_mix),
                "replay_ratio": math_env.last_replay_ratio,
                "grounded_ratio": config.grounded_ratio,
                "grounded_stats": dict(math_env.last_grounded_stats),
            }
            save_iteration_results(iteration, trajectories, all_metrics, config)

            csv_metrics = {
                "iteration": iteration,
                "train/policy_loss": training_metrics["policy_loss"],
                "train/value_loss": training_metrics["value_loss"],
                "train/entropy": training_metrics["entropy"],
                "train/approx_kl": training_metrics["approx_kl"],
                "train/clip_fraction": training_metrics["clip_fraction"],
                "train/update_steps": training_metrics.get("update_steps", 0.0),
                "train/update_steps_planned": training_metrics.get(
                    "update_steps_planned", 0.0
                ),
                "train/early_stop_epoch": training_metrics.get(
                    "early_stop_epoch", -1.0
                ),
                "train/kl_trip_threshold": training_metrics.get(
                    "kl_trip_threshold",
                    config.target_kl * config.kl_trip_multiplier,
                ),
                "rollout/mean_reward": buffer_stats["mean_episode_reward"],
                "rollout/num_trajectories": len(trajectories),
                "rollout/mean_length": buffer_stats["mean_episode_length"],
                "curriculum/topic_diversity": curriculum_stats["topic_diversity"],
                "curriculum/avg_difficulty": curriculum_stats["avg_difficulty"],
                "curriculum/avg_novelty": curriculum_stats["avg_novelty"],
                "curriculum/replay_ratio": math_env.last_replay_ratio,
                "curriculum/avg_question_reward": curriculum_stats[
                    "avg_question_reward"
                ],
                "curriculum/avg_question_reward_selfplay": curriculum_stats[
                    "avg_question_reward_selfplay"
                ],
                "curriculum/avg_solution_reward": curriculum_stats[
                    "avg_solution_reward"
                ],
                "curriculum/selfplay_topic_entropy": curriculum_stats[
                    "selfplay_topic_entropy"
                ],
                "curriculum/selfplay_topic_diversity": curriculum_stats[
                    "selfplay_topic_diversity"
                ],
                "grounded/ratio": config.grounded_ratio,
                "grounded/count": math_env.last_grounded_stats.get("count", 0),
                "grounded/correct": math_env.last_grounded_stats.get("correct", 0),
                "grounded/accuracy": math_env.last_grounded_stats.get("accuracy", 0.0),
                "grounded/mean_reward": math_env.last_grounded_stats.get(
                    "mean_reward", 0.0
                ),
                "prm/mean_of_means": curriculum_stats.get("prm_mean_of_means", 0.0),
                "prm/mean_of_mins": curriculum_stats.get("prm_mean_of_mins", 0.0),
                "prm/mean_of_finals": curriculum_stats.get("prm_mean_of_finals", 0.0),
                "prm/samples": curriculum_stats.get("prm_sample_count", 0),
                "prm/degraded": curriculum_stats.get("prm_degraded_count", 0),
                "perf/rollout_time": rollout_seconds,
                "perf/train_time": train_seconds,
                "perf/total_time": total_seconds,
                "perf/tokens_per_second": throughput_metrics.get(
                    "tokens_per_second", 0.0
                ),
                "system/disk_free_gb": disk_metrics.get("free_gb", 0.0),
            }
            if eval_results:
                csv_metrics["eval/accuracy"] = eval_results.get("accuracy", 0.0)
                csv_metrics["eval/correct"] = eval_results.get("correct", 0)
                csv_metrics["eval/total"] = eval_results.get("total", 0)
            if gpu_metrics:
                util_vals = [
                    v for k, v in gpu_metrics.items() if k.endswith("_utilization")
                ]
                if util_vals:
                    csv_metrics["system/gpu_util_percent"] = (
                        sum(util_vals) / len(util_vals)
                    )

            logger_csv.log(csv_metrics, step=iteration)

        final_eval = evaluate_policy(
            policy,
            tokenizer,
            config.eval_data_path,
            max_samples=config.eval_max_samples,
            max_new_tokens=config.eval_max_new_tokens,
        )
        logger.info(
            "Training complete. Initial acc: %.2f%% | Final acc: %.2f%% | Delta: %.2f%%",
            initial_eval.get("accuracy", 0.0) * 100.0,
            final_eval.get("accuracy", 0.0) * 100.0,
            (final_eval.get("accuracy", 0.0) - initial_eval.get("accuracy", 0.0))
            * 100.0,
        )

        logger_csv.save_summary(
            {
                "initial_accuracy": initial_eval.get("accuracy", 0.0),
                "final_accuracy": final_eval.get("accuracy", 0.0),
                "improvement": final_eval.get("accuracy", 0.0)
                - initial_eval.get("accuracy", 0.0),
                "best_accuracy": best_accuracy,
                "total_iterations": config.num_iterations,
                "console_output_path": str(console_log_path),
            }
        )
        logger_csv.finish()
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        console_log_file.close()


if __name__ == "__main__":
    main()
