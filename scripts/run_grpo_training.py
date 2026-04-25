"""
GRPO training for self-improvement math environment.

Group Relative Policy Optimization (GRPO) is dramatically simpler and more
stable than PPO for LLM fine-tuning on math tasks:

  - No value function / critic needed
  - No GAE, no gamma, no lambda
  - No KL instability from per-step advantage collapse
  - Advantages computed as within-group z-scores: A_i = (R_i - mean_R) / std_R
  - Proven on math RL: DeepSeek-Math, Qwen-Math, DAPO all use GRPO variants

The algorithm per question:
  1. Generate K solutions (default K=4)
  2. Score each with the existing reward pipeline (PRM + SymPy + format)
  3. A_i = (R_i - mean(R)) / (std(R) + eps)
  4. policy_loss = -mean_i [ A_i * sum_t log pi(a_t | s_{<t}) / T_i ]
  5. Skip the group if all rewards are identical (zero gradient signal)

Expected improvement curve:
  - Iterations 1-5:  reward mean rising, policy learning to avoid R=0 outputs
  - Iterations 5-15: GSM8K accuracy starts moving (+2-5%)
  - Iterations 15-30: continued improvement toward ~70-75%+ from 63.6% baseline

Usage:
    python scripts/run_grpo_training.py \\
        --base-model checkpoints/dual_task_v1 \\
        --gsm8k-data data/sft/gsm8k_sft.jsonl \\
        --num-iterations 30 \\
        --group-size 4 \\
        --questions-per-iter 16

    # Faster smoke test (no PRM, 3 iters):
    python scripts/run_grpo_training.py \\
        --base-model checkpoints/dual_task_v1 \\
        --num-iterations 3 --group-size 4 --questions-per-iter 8 \\
        --no-prm --skip-initial-eval --run-name smoke_grpo
"""

from __future__ import annotations

import argparse
import atexit
import copy
import csv
import json
import logging
import random
import re
import shutil
import sys
import time
import types
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.convert_gsm8k_to_sft import parse_gsm8k_answer
from scripts.eval_sft_inference import evaluate_gsm8k
from src.rl.prm_scorer import ProcessRewardScorer
from src.sft.solution_format import extract_final_answer_numeric_str
from src.utils.attn_backend import select_attn_implementation
from src.rl.math_environment_curriculum import CurriculumMathEnvironment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging infrastructure
# ---------------------------------------------------------------------------

class TeeStream:
    """Mirrors every write to a terminal stream into a log file.

    Wrapping sys.stdout and sys.stderr with this object ensures that *all*
    output — bare print() calls, tqdm bars, third-party library writes — lands
    in the run log file in addition to the terminal.

    A separate FileHandler on the root logger (see _add_file_logging) captures
    the Python logging subsystem independently, because logging.StreamHandler
    stores a reference to the stream at creation time and therefore bypasses
    any later sys.stderr reassignment.  Both mechanisms together guarantee that
    nothing escapes the log file.
    """

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


def _add_file_logging(log_path: Path) -> logging.FileHandler:
    """Attach a FileHandler to the root logger.

    Every logger.info / logger.warning / … call — from any module — will be
    written to ``log_path`` in addition to the terminal.  This complements
    TeeStream: TeeStream captures bare print() / sys.stderr writes; this
    handler captures the logging subsystem, which uses its own internal stream
    reference that TeeStream cannot intercept.
    """
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s - %(message)s"
    ))
    logging.getLogger().addHandler(fh)
    return fh


if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True   # auto-tune fastest conv algo per shape


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gsm8k(path: str) -> List[Dict[str, str]]:
    """Return list of {"question": ..., "gold_final": ...} from a JSONL file."""
    pairs: List[Dict[str, str]] = []
    p = Path(path)
    if not p.exists():
        logger.warning("GSM8K data not found at %s", path)
        return pairs
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            question = ""
            gold = ""
            if "question" in rec and "answer" in rec:
                question = rec["question"].strip()
                _, gold = parse_gsm8k_answer(str(rec["answer"]))
            elif "messages" in rec:
                user_text = ""
                asst_text = ""
                for msg in rec["messages"]:
                    if msg.get("role") == "user" and not user_text:
                        user_text = msg.get("content", "").strip()
                    elif msg.get("role") == "assistant" and not asst_text:
                        asst_text = msg.get("content", "")
                if "Problem:" in user_text:
                    question = user_text.split("Problem:", 1)[1].strip()
                else:
                    question = user_text
                answer_str = extract_final_answer_numeric_str(asst_text) or ""
                gold = answer_str.strip()

            if question and gold:
                pairs.append({"question": question, "gold_final": gold})
    logger.info("Loaded %d GSM8K QA pairs from %s", len(pairs), path)
    return pairs


# ---------------------------------------------------------------------------
# MATH harder dataset
# ---------------------------------------------------------------------------

def _extract_boxed(text: str) -> Optional[str]:
    r"""Extract the content of the first ``\boxed{...}`` in *text*."""
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    return m.group(1).strip() if m else None


def _boxed_to_numeric(answer: str) -> Optional[str]:
    """
    Convert a ``\\boxed{...}`` answer to a plain numeric string.

    Returns a string of the form ``"42"`` or ``"3.5000"`` when the answer
    is a recognisable integer, decimal, or simple fraction (``3/4`` or
    ``\\frac{3}{4}``).  Returns ``None`` for symbolic / multi-part answers
    like ``3\\sqrt{2}`` or ``(1, 2)``.
    """
    ans = answer.strip()
    # Direct integer
    try:
        return str(int(ans))
    except ValueError:
        pass
    # Direct float (includes "3.5", "0.75", etc.)
    try:
        v = float(ans)
        return str(int(v)) if v == int(v) else f"{v:.4f}"
    except ValueError:
        pass
    # LaTeX fraction  \frac{num}{den}
    m = re.fullmatch(r"\\frac\{(\d+)\}\{(\d+)\}", ans)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        if den:
            v = num / den
            return str(int(v)) if v == int(v) else f"{v:.4f}"
    # Plain fraction  num/den
    m = re.fullmatch(r"(\d+)/(\d+)", ans)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        if den:
            v = num / den
            return str(int(v)) if v == int(v) else f"{v:.4f}"
    return None


def load_math_dataset(
    local_path: Optional[str] = None,
    cache_path: str = "data/math/math_numeric.jsonl",
    max_difficulty: int = 3,
) -> List[Dict[str, str]]:
    """
    Load a subset of the MATH competition dataset filtered to problems with
    numerically-verifiable answers (integers, decimals, simple fractions).

    Loading order
    -------------
    1. ``local_path`` if provided and the file exists.
    2. ``cache_path`` if that file exists (written on first HF download).
    3. HuggingFace ``competition_math`` dataset; filtered + written to
       ``cache_path`` for subsequent runs.

    Only problems with ``Level ≤ max_difficulty`` are included.  Difficulty
    1-2 ≈ AMC-8 level (comparable to hard GSM8K); difficulty 3 ≈ AMC-10.
    Levels 4-5 are graduate-level and usually too hard for a 1.5B model to
    get any reward signal from (win_rate ≈ 0 → skipped groups every iter).
    """
    for candidate in filter(None, [local_path, cache_path]):
        p = Path(candidate)
        if p.exists():
            pairs: List[Dict[str, str]] = []
            with p.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            pairs.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            if pairs:
                logger.info("Loaded %d MATH pairs from %s", len(pairs), p)
                return pairs

    # Download from HuggingFace
    logger.info(
        "MATH dataset not found locally — downloading from HuggingFace "
        "(qwedsacf/competition_math, difficulty ≤ %d, numeric answers only)...",
        max_difficulty,
    )
    # Try HF sources in priority order.  Only keep sources confirmed reachable;
    # lighteval/MATH and hendrycks/competition_math have network/naming issues.
    _HF_SOURCES = [
        ("qwedsacf/competition_math", {}),           # reliable public mirror
        ("lighteval/MATH-Hard",       {"name": "default"}),  # hard subset
    ]
    ds = None
    for hf_name, hf_kwargs in _HF_SOURCES:
        try:
            from datasets import load_dataset  # type: ignore
            ds = load_dataset(hf_name, split="train", trust_remote_code=True, **hf_kwargs)
            logger.info("Loaded HuggingFace dataset: %s (%d items)", hf_name, len(ds))
            break
        except Exception as exc:
            logger.warning("Could not load %s: %s — trying next source.", hf_name, exc)
    if ds is None:
        logger.warning(
            "All MATH dataset sources failed. Proceeding with GSM8K only. "
            "To load offline: download from https://github.com/hendrycks/math "
            "and pass --math-data <path_to_jsonl>."
        )
        return []

    pairs = []
    for item in ds:
        level_str = item.get("level", "Level 5")
        try:
            level = int(level_str.split()[-1])
        except (ValueError, IndexError):
            level = 5
        if level > max_difficulty:
            continue

        question = item.get("problem", "").strip()
        solution = item.get("solution", "")
        boxed    = _extract_boxed(solution)
        if not boxed:
            continue
        numeric  = _boxed_to_numeric(boxed)
        if not numeric:
            continue
        pairs.append({"question": question, "gold_final": numeric})

    if pairs:
        out_p = Path(cache_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", encoding="utf-8") as f:
            for p_item in pairs:
                f.write(json.dumps(p_item) + "\n")
        logger.info("Cached %d MATH numeric pairs to %s", len(pairs), out_p)
    else:
        logger.warning("No MATH pairs passed the numeric filter — check the dataset.")

    return pairs


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

def compute_grounded_reward(
    question: str,
    solution: str,
    gold_final: str,
    math_env: CurriculumMathEnvironment,
) -> Dict[str, float]:
    """Score a solution against a known gold answer (grounded path).

    Returns a dict with:
      combined_score  – 0.50×correct + 0.40×process(prm_final,prm_mean) + 0.10×fmt
      step_accuracy   – fraction of PRM steps rated > 0.5 (the core process metric)
      prm_mean_score  – PRM mean across all steps
      prm_final_score – PRM score on the final reasoning step
      gt_match        – bool, whether pred matches gold
      format_score    – format compliance score
    """
    result = math_env.compute_grounded_reward(
        question=question,
        solution=solution,
        gold_final=gold_final,
    )
    return {
        "combined_score":  float(result.get("combined_score",  0.0)),
        "step_accuracy":   float(result.get("step_accuracy",   0.0)),
        "prm_mean_score":  float(result.get("prm_mean_score",  0.0)),
        "prm_final_score": float(result.get("prm_final_score", 0.0)),
        "gt_match":        bool(result.get("gt_match",         False)),
        "format_score":    float(result.get("format_score",    0.0)),
    }


def compute_self_play_reward(
    question: str,
    solution: str,
    target_topic: str,
    target_difficulty: float,
    math_env: CurriculumMathEnvironment,
) -> Tuple[float, float, float, Dict]:
    """Score a self-generated question + solution (self-play path).

    Returns (combined_reward, question_reward, solution_reward, q_metrics).

    Reward breakdown: R = 0.40×question_quality + 0.60×solution_quality,
    where question_quality captures topic match, difficulty fit, clarity,
    novelty, and solvability — completing the Theme #4 self-improvement loop
    where the model is rewarded for generating *good challenges*, not only
    for solving them.

    q_metrics contains the full question quality breakdown:
      topic_match, difficulty_fit, clarity, novelty, solvability, overall_score
    """
    result = math_env.compute_reward(
        question=question,
        solution=solution,
        target_topic=target_topic,
        target_difficulty=target_difficulty,
    )
    combined  = float(result["combined_score"])
    sol_score = result.get("solution_metrics", {})
    s_reward  = float(sol_score.get("overall_score", 0.0)) if isinstance(sol_score, dict) else 0.0

    # question_reward is NOT a top-level key in compute_reward()'s return dict.
    # The question quality score lives inside question_metrics["overall_score"].
    # Key mapping from QuestionEvalResult.to_dict():
    #   overall_score    → scalar  (overall question quality)
    #   topic_match      → scalar
    #   difficulty_score → scalar  (fit to target difficulty; named _score not _fit)
    #   clarity          → scalar
    #   solvability_score→ scalar  (the dict version is under "solvability" — don't use that)
    #   novelty_combined → scalar  (the dict version is under "novelty" — don't use that)
    q_metrics_raw = result.get("question_metrics", {}) or {}
    q_reward = float(q_metrics_raw.get("overall_score", 0.0))
    q_metrics: Dict = {
        "overall_score":  q_reward,
        "topic_match":    float(q_metrics_raw.get("topic_match",       0.0)),
        "difficulty_fit": float(q_metrics_raw.get("difficulty_score",  0.0)),
        "clarity":        float(q_metrics_raw.get("clarity",           0.0)),
        "novelty":        float(q_metrics_raw.get("novelty_combined",  0.0)),
        "solvability":    float(q_metrics_raw.get("solvability_score", 0.0)),
    }
    return combined, q_reward, s_reward, q_metrics


@torch.no_grad()
def generate_question(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    instruction: str,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    """Generate a math question from a curriculum instruction.

    The instruction (e.g. "Create a 2-step word problem about money in a
    shopping context") is formatted with the chat template so the SFT-trained
    model responds in its expected style.

    Returns the raw decoded question text (no special tokens).
    """
    system = (
        "You are a math problem creator. "
        "Given an instruction, write a clear, solvable math word problem. "
        "Write only the problem statement — no solution, no explanation."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": instruction},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt = f"{system}\n\n{instruction}\n"

    enc = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512
    ).to(device)
    prompt_len = enc["input_ids"].shape[1]

    stop_ids: List[int] = []
    if tokenizer.eos_token_id is not None:
        stop_ids.append(tokenizer.eos_token_id)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end not in stop_ids:
        stop_ids.append(im_end)

    out = model.generate(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.9,   # slightly higher than solutions for diversity
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=stop_ids or None,
        use_cache=True,
    )
    return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _build_stop_token_ids(tokenizer: AutoTokenizer) -> List[int]:
    """
    Return a list of token IDs that should stop generation.

    Qwen2.5-chat models end turns with <|im_end|> (ID 151645).  If that
    token is not the same as eos_token_id we include both so that .generate()
    halts cleanly instead of running to max_new_tokens and emitting repetitive
    garbage.
    """
    stop_ids: List[int] = []
    if tokenizer.eos_token_id is not None:
        stop_ids.append(tokenizer.eos_token_id)
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end_id, int) and im_end_id not in stop_ids:
        stop_ids.append(im_end_id)
    return stop_ids or None  # type: ignore[return-value]


def generate_solutions_batched(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    K: int,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> Tuple[List[str], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Generate K solutions for a prompt in a **single batched** model.generate() call.

    Batching all K sequences together achieves near-100% GPU utilisation vs
    the old sequential loop (which was <20% utilised).  On an A100 with K=8,
    this is typically 4-8× faster than K sequential calls.

    ``prompt`` must come from ``math_env.format_solution_prompt(question)``
    so the chat-template system/user wrapping exactly matches the SFT
    training format.

    Returns:
        solutions       : K decoded strings (prompt stripped, specials removed)
        input_ids_list  : K full (prompt+response) token ID tensors
        response_masks  : K bool masks (True = non-pad response token)
        old_log_probs   : K scalar tensors, sum(log π_old(token)) over response,
                          computed no_grad — used for IS clip ratio in the loss.
    """
    stop_ids = _build_stop_token_ids(tokenizer)
    pad_id: int = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=1024,
    ).to(device)
    prompt_len: int = enc["input_ids"].shape[1]

    # Expand prompt K times along the batch dimension (no data copy).
    input_ids_batch = enc["input_ids"].expand(K, -1).contiguous()
    attn_mask_batch = enc["attention_mask"].expand(K, -1).contiguous()

    model.eval()
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids_batch,
            attention_mask=attn_mask_batch,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=pad_id,
            eos_token_id=stop_ids,
            use_cache=True,
        )
        # out: [K, prompt_len + padded_response_len]

    # ── 1. Build masks and decode solutions ──────────────────────────────────
    solutions: List[str] = []
    input_ids_list: List[torch.Tensor] = []
    response_masks: List[torch.Tensor] = []

    pad_id_t = torch.tensor(pad_id, device=device, dtype=out.dtype)
    for i in range(K):
        full_ids = out[i]
        response_section = full_ids[prompt_len:]
        mask = torch.zeros(full_ids.shape[0], dtype=torch.bool, device=device)
        mask[prompt_len:] = response_section != pad_id_t
        solution = tokenizer.decode(response_section, skip_special_tokens=True)
        solutions.append(solution)
        input_ids_list.append(full_ids)
        response_masks.append(mask)

    # ── 2. Batched old_log_probs — ONE forward pass for all K sequences ───────
    # The old sequential approach called compute_sequence_log_prob K times
    # (K separate CPU→GPU round-trips + K forward passes).  A single batched
    # forward pass over out[K, total_len] gives the same result K× faster.
    #
    # Attention mask: always attend to prompt tokens; attend to response tokens
    # only where they are non-pad.  This matches what the model saw during
    # model.generate() and prevents padding from distorting log probs.
    old_log_probs: List[torch.Tensor] = []
    with torch.no_grad():
        attn_mask_lp = (out != pad_id_t)          # [K, total_len]
        attn_mask_lp[:, :prompt_len] = True        # prompt always attended

        batch_logits = model(
            input_ids=out,
            attention_mask=attn_mask_lp.long(),
            use_cache=False,
            return_dict=True,
        ).logits  # [K, total_len, vocab]

        for i in range(K):
            full_ids = out[i]
            mask = response_masks[i]

            shift_logits = batch_logits[i, :-1]      # [total_len-1, vocab]
            shift_labels  = full_ids[1:]              # [total_len-1]
            shift_mask    = mask[1:]                  # [total_len-1]

            lp_tokens = F.log_softmax(shift_logits, dim=-1)[
                torch.arange(shift_logits.size(0), device=device),
                shift_labels,
            ]  # [total_len-1]
            resp_lps = lp_tokens[shift_mask]
            old_log_probs.append(
                resp_lps.sum().detach() if resp_lps.numel() > 0
                else torch.tensor(0.0, device=device)
            )

    return solutions, input_ids_list, response_masks, old_log_probs


def compute_sequence_log_prob(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Forward pass through model to get sum of log probs for response tokens.

    Returns scalar tensor (differentiable).
    """
    # input_ids: [seq_len]  →  unsqueeze to [1, seq_len]
    ids = input_ids.unsqueeze(0)
    # Causal LM: logits[i] predicts token[i+1]
    outputs = model(input_ids=ids, use_cache=False, return_dict=True)
    logits = outputs.logits[0]  # [seq_len, vocab]

    # Shift: predict token t+1 from logit at position t
    shift_logits = logits[:-1]           # [seq_len-1, vocab]
    shift_labels = input_ids[1:]         # [seq_len-1]
    shift_mask = response_mask[1:]       # [seq_len-1]  (response tokens)

    log_probs = F.log_softmax(shift_logits, dim=-1)  # [seq_len-1, vocab]
    token_log_probs = log_probs[
        torch.arange(shift_logits.size(0), device=shift_logits.device),
        shift_labels,
    ]  # [seq_len-1]

    # Sum log probs over response tokens only
    response_log_probs = token_log_probs[shift_mask]
    if response_log_probs.numel() == 0:
        return torch.tensor(0.0, requires_grad=True, device=input_ids.device)
    return response_log_probs.sum()


# ---------------------------------------------------------------------------
# GRPO update for one question group
# ---------------------------------------------------------------------------

def grpo_loss_for_group(
    model: AutoModelForCausalLM,
    input_ids_list: List[torch.Tensor],
    response_masks: List[torch.Tensor],
    rewards: List[float],
    old_log_probs: List[torch.Tensor],
    clip_eps: float = 0.2,
    kl_coef: float = 0.0,
    ref_model: Optional[AutoModelForCausalLM] = None,
    eps: float = 1e-8,
) -> Optional[torch.Tensor]:
    """
    Compute GRPO loss for a group of K solutions to the same question.

    IS clip (``clip_eps > 0``):
        ratio  = π_θ(response) / π_old(response)   [sequence level]
        L_GRPO = -min(ratio × A, clip(ratio, 1-ε, 1+ε) × A) / T

    Reference-policy KL penalty (``kl_coef > 0``, ``ref_model`` required):
        KL(π_θ ‖ π_ref) ≈ (log π_θ − log π_ref) / T   per sequence
        L_total = L_GRPO + β × KL

    The KL term acts as an anchor: it prevents the policy from drifting so
    far from its starting point that it forgets the SFT knowledge baked in
    during dual_task_v1 fine-tuning.  β=0.04 is a conservative starting
    value (matches DeepSeekMath GRPO default).

    Returns None if all rewards are identical (zero gradient signal).
    """
    rewards_arr = np.array(rewards, dtype=np.float32)
    std_r = rewards_arr.std()
    if std_r < eps:
        return None

    mean_r = rewards_arr.mean()
    advantages = (rewards_arr - mean_r) / (std_r + eps)
    advantages = np.clip(advantages, -5.0, 5.0)

    _device = next(model.parameters()).device
    group_loss = torch.tensor(0.0, device=_device)
    n_valid = 0

    model.train()
    for ids, mask, adv, old_lp in zip(
        input_ids_list, response_masks, advantages, old_log_probs
    ):
        new_lp = compute_sequence_log_prob(model, ids, mask)  # differentiable
        n_response = int(mask[1:].sum().item())
        if n_response == 0:
            continue

        adv_t = torch.tensor(adv, dtype=new_lp.dtype, device=_device)

        # ── GRPO surrogate (with optional IS clip) ────────────────────────
        if clip_eps > 0:
            ratio = torch.exp(new_lp - old_lp.to(_device).detach())
            surr_unclipped = ratio * adv_t / n_response
            surr_clipped   = (
                torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
                * adv_t / n_response
            )
            loss_i = -torch.min(surr_unclipped, surr_clipped)
        else:
            loss_i = -(adv_t * new_lp / n_response)

        # ── Reference-policy KL penalty ───────────────────────────────────
        # KL(π_θ ‖ π_ref) = mean_token(log π_θ − log π_ref)
        # Adding +β×KL to the minimisation objective penalises drift from
        # the reference (frozen) checkpoint.  This is differentiable through
        # new_lp; ref_lp is always detached (no grad through frozen model).
        if kl_coef > 0.0 and ref_model is not None:
            with torch.no_grad():
                ref_lp = compute_sequence_log_prob(ref_model, ids, mask)
            kl_per_token = (new_lp - ref_lp.to(_device).detach()) / n_response
            loss_i = loss_i + kl_coef * kl_per_token

        group_loss = group_loss + loss_i
        n_valid += 1

    if n_valid == 0:
        return None
    return group_loss / n_valid


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _log_eval_result(label: str, res: Dict, best: Optional[float]) -> None:
    """Print a structured evaluation summary that mirrors the training objective."""
    cs      = float(res.get("combined_score",  0.0))
    cr      = float(res.get("correct_rate",    0.0))
    step_a  = float(res.get("step_accuracy",   0.0))
    prm     = float(res.get("prm_mean",        0.0))
    prm_fin = float(res.get("prm_final",       0.0))
    fmt     = float(res.get("format_mean",     0.0))
    n_sc    = int(res.get("n_scored", res.get("total", 0)))
    fa_acc  = float(res.get("final_answer_accuracy", cr))
    pak     = res.get("pass_at_k")
    pak_k   = int(res.get("pass_at_k_k", 4))

    best_str = f" (best={best:.4f})" if best is not None else ""
    logger.info(
        "Training Score  [%s]: %.4f%s  |  n=%d",
        label, cs, best_str, n_sc,
    )
    logger.info(
        "  Components    : 0.50×correct(%.1f%%) + 0.40×process + 0.10×fmt(%.3f)",
        100 * cr, fmt,
    )
    logger.info(
        "  Process score : prm_mean=%.3f  prm_final=%.3f  → weighted=%.3f",
        prm, prm_fin, 0.60 * prm_fin + 0.40 * prm,
    )
    logger.info(
        "  Step accuracy : %.1f%%  ← fraction of reasoning steps PRM rates correct (>0.5)",
        100 * step_a,
    )
    if pak is not None:
        logger.info(
            "  pass@%d (T=0.8): %.1f%%  |  greedy correct: %.1f%%  "
            "← ceiling vs floor gap",
            pak_k, 100 * pak, 100 * cr,
        )
    logger.info(
        "  (debug) final-answer accuracy: %.1f%%",
        100 * fa_acc,
    )


def evaluate_policy(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    eval_data_path: str,
    max_samples: int,
    max_new_tokens: int,
    math_env: Optional[Any] = None,
    pass_at_k: int = 4,
) -> Dict[str, object]:
    """Run GSM8K evaluation using the SAME reward formula as GRPO training.

    When *math_env* is supplied a ``reward_fn`` is constructed that calls
    ``math_env.compute_grounded_reward(question, solution, gold)``.  This
    returns ``combined_score = 0.60×correct + 0.15×PRM + 0.15×SymPy + 0.10×format``,
    making the eval metric IDENTICAL to the GRPO training objective.  Any
    improvement in step quality, SymPy verification or format compliance shows
    up immediately in the accuracy number instead of being hidden behind the
    coarse binary final-answer signal.
    """
    if not Path(eval_data_path).exists():
        return {"accuracy": 0.0, "combined_score": 0.0, "total": 0}
    model.eval()

    reward_fn = None
    if math_env is not None:
        import logging as _log_mod
        _mec_logger = _log_mod.getLogger("src.rl.math_environment_curriculum")

        def reward_fn(question: str, solution: str, gold: str) -> Dict:
            """Thin wrapper that silences per-sample debug logs during eval."""
            _old = _mec_logger.level
            _mec_logger.setLevel(_log_mod.WARNING)
            try:
                return math_env.compute_grounded_reward(question, solution, gold)
            finally:
                _mec_logger.setLevel(_old)

    results = evaluate_gsm8k(
        model=model,
        tokenizer=tokenizer,
        data_path=eval_data_path,
        max_samples=max_samples,
        max_new_tokens=max_new_tokens,
        reward_fn=reward_fn,
        pass_at_k=pass_at_k,
    )
    model.train()
    return results


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO training for self-improvement math")
    parser.add_argument("--base-model", default="checkpoints/dual_task_v1")
    parser.add_argument("--output-dir", default="checkpoints/grpo")
    parser.add_argument("--gsm8k-data", default="data/sft/gsm8k_sft.jsonl")
    parser.add_argument("--eval-data-path", default="data/sft/dual_task_val.jsonl")
    parser.add_argument("--num-iterations", type=int, default=30)
    parser.add_argument(
        "--group-size", type=int, default=4,
        help="K: number of solutions per question per GRPO group (default 4).",
    )
    parser.add_argument(
        "--questions-per-iter", type=int, default=16,
        help="Number of questions per training iteration (default 16).",
    )
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-max-samples", type=int, default=250)
    parser.add_argument("--eval-max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--eval-pass-at-k", type=int, default=0,
        help="Number of sampled solutions per eval problem for pass@k (0 to disable). "
             "Makes eval directly comparable to training batch_acc (both K samples at T=0.8). "
             "Disabled by default — enable with e.g. --eval-pass-at-k 4 for demo runs only "
             "(adds K×eval_samples extra forward passes).",
    )
    parser.add_argument("--use-prm", dest="use_prm", action="store_true", default=True)
    parser.add_argument("--no-prm", dest="use_prm", action="store_false")
    parser.add_argument("--prm-model", default="Qwen/Qwen2.5-Math-PRM-7B")
    parser.add_argument("--skip-initial-eval", action="store_true")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--kl-coef", type=float, default=0.04,
        help="Reference-policy KL penalty coefficient β. 0 = disabled. Default 0.04.",
    )
    parser.add_argument(
        "--math-data", type=str, default=None,
        help="Path to MATH dataset JSONL. If absent, downloads from HuggingFace "
             "(competition_math) and caches to data/math/math_numeric.jsonl.",
    )
    parser.add_argument(
        "--math-mix-ratio", type=float, default=0.3,
        help="Fraction of each question batch drawn from MATH (vs GSM8K). "
             "0 = GSM8K only, 1 = MATH only. Default 0.3.",
    )
    parser.add_argument(
        "--math-max-difficulty", type=int, default=3,
        help="Maximum MATH difficulty level to include (1-5). Default 3.",
    )
    parser.add_argument(
        "--clip-eps", type=float, default=0.2,
        help="Importance-sampling clip ratio ε (PPO-style clip applied inside GRPO). "
             "0 = disabled (plain GRPO). Default 0.2.",
    )
    parser.add_argument(
        "--warmup-iters", type=int, default=3,
        help="Number of linear LR warmup iterations before cosine decay. Default 3.",
    )
    parser.add_argument(
        "--min-lr-ratio", type=float, default=0.1,
        help="Cosine decay floor as a fraction of peak LR (default 0.1 = 10%%).",
    )
    parser.add_argument(
        "--difficulty-alpha", type=float, default=2.0,
        help="Sharpness of difficulty-weighted question sampling. "
             "Higher = stronger preference for on-the-margin questions (win_rate ≈ 0.5). "
             "0 = uniform random (default behaviour). Default 2.0.",
    )
    parser.add_argument(
        "--overlong-filter", dest="overlong_filter",
        action="store_true", default=True,
        help="Skip solutions that hit max-new-tokens (truncated = no Final Answer). Default on.",
    )
    parser.add_argument(
        "--no-overlong-filter", dest="overlong_filter", action="store_false",
        help="Disable overlong-response filtering.",
    )
    parser.add_argument(
        "--save-every", type=int, default=1,
        help="Save a full checkpoint every N iterations (default 1 = every iter). "
             "Best-policy is always saved when accuracy improves, independently of this flag.",
    )
    parser.add_argument(
        "--keep-last", type=int, default=0,
        help="Keep only the last K iter_* checkpoints on disk (0 = keep all). "
             "best_policy/ is never pruned.",
    )
    parser.add_argument(
        "--self-play-ratio", type=float, default=0.3,
        help="Fraction of each question batch that uses SELF-PLAY (model generates the "
             "question from a curriculum instruction, then solves it, rewarded on "
             "0.40 × question_quality + 0.60 × solution_quality). "
             "The remaining (1 - ratio) uses GROUNDED questions from GSM8K / MATH with "
             "gold-answer reward. "
             "0.0 = fully grounded (original behaviour), 1.0 = fully self-play. "
             "Default 0.3 — mirrors the PPO default of 30%% grounded / 70%% self-play "
             "(inverted here because grounded is our primary accuracy signal).",
    )
    args = parser.parse_args()

    # ── Run identity ─────────────────────────────────────────────────────────
    # Establish run_name first — everything that follows (including log paths)
    # derives from it.
    run_name = args.run_name or f"grpo_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Log directory ─────────────────────────────────────────────────────────
    # One canonical directory for ALL run artefacts that are not model weights:
    #   console_output.log  — full terminal mirror (logger.* + print + tqdm)
    #   config.json         — serialised CLI args for reproducibility
    #   metrics.csv         — one row per iteration, written live
    #   summary.json        — written at the end of training
    log_dir = Path("logs") / "grpo" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── Console log file ─────────────────────────────────────────────────────
    console_log_path = log_dir / "console_output.log"
    _console_log_file = console_log_path.open("a", encoding="utf-8", buffering=1)

    # 1) FileHandler on the root logger → every logger.*() call goes to file.
    #    This is necessary because logging.StreamHandler stores a reference to
    #    sys.stderr at *creation* time (inside logging.basicConfig above), so
    #    reassigning sys.stderr later has no effect on existing handlers.
    _file_handler = _add_file_logging(console_log_path)

    # 2) TeeStream on sys.stdout / sys.stderr → every print() / tqdm bar /
    #    library write also goes to file.  Both together cover 100% of output.
    _original_stdout = sys.stdout
    _original_stderr = sys.stderr
    sys.stdout = TeeStream(_original_stdout, _console_log_file)
    sys.stderr = TeeStream(_original_stderr, _console_log_file)

    logger.info("=" * 70)
    logger.info("GRPO run: %s", run_name)
    logger.info("Checkpoints : %s", out_dir)
    logger.info("Logs        : %s", log_dir)
    logger.info("Console log : %s", console_log_path)
    logger.info("=" * 70)

    # ── Persist config for reproducibility ───────────────────────────────────
    (log_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2, default=str), encoding="utf-8"
    )

    # ── Live CSV metrics writer ───────────────────────────────────────────────
    # Written one row per iteration so you can tail / open in Excel mid-run.
    _metrics_csv_path = log_dir / "metrics.csv"
    _csv_file: Optional[Any] = None
    _csv_writer: Optional[Any] = None

    def _append_metrics_csv(row: Dict[str, Any]) -> None:
        """Append one metrics row to metrics.csv; writes header on first call."""
        nonlocal _csv_file, _csv_writer
        # Normalise floats to fixed precision so the CSV is human-readable.
        flat = {
            k: (f"{v:.6f}" if isinstance(v, float) else v)
            for k, v in row.items()
        }
        if _csv_writer is None:
            _csv_file = _metrics_csv_path.open("w", newline="", encoding="utf-8")
            _csv_writer = csv.DictWriter(
                _csv_file,
                fieldnames=list(flat.keys()),
                extrasaction="ignore",
            )
            _csv_writer.writeheader()
        _csv_writer.writerow(flat)
        _csv_file.flush()  # type: ignore[union-attr]

    # ── Teardown: restore streams and close files on any exit path ───────────
    # atexit runs unconditionally — on normal completion, keyboard interrupt,
    # unhandled exception, or OOM crash.  This is equivalent to a finally block
    # without requiring the entire training body to be re-indented.
    def _teardown_logging() -> None:
        sys.stdout = _original_stdout
        sys.stderr = _original_stderr
        logging.getLogger().removeHandler(_file_handler)
        if not getattr(_file_handler.stream, "closed", False):
            _file_handler.close()
        if _csv_file is not None and not getattr(_csv_file, "closed", False):
            _csv_file.close()
        if not _console_log_file.closed:
            _console_log_file.close()

    atexit.register(_teardown_logging)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    attn_impl = select_attn_implementation()
    logger.info("Device: %s | attn: %s", device, attn_impl)
    if torch.cuda.is_available():
        _gpu = torch.cuda.get_device_properties(0)
        logger.info(
            "GPU: %s | %.1f GB VRAM | capability sm_%d%d",
            _gpu.name, _gpu.total_memory / 1e9, _gpu.major, _gpu.minor,
        )
    logger.info(
        "Run config: K=%d N=%d lr=%.1e T=%.2f max_new=%d | "
        "clip_eps=%.2f kl_coef=%.4f warmup=%d | diff_alpha=%.1f | "
        "self_play=%.0f%% grounded=%.0f%% | "
        "math_mix=%.0f%% math_maxdiff=%d | overlong_filter=%s | "
        "eval_every=%d eval_N=%d | grad_clip=%.2f save_every=%d keep_last=%d",
        args.group_size, args.questions_per_iter, args.learning_rate,
        args.temperature, args.max_new_tokens,
        args.clip_eps, args.kl_coef, args.warmup_iters,
        args.difficulty_alpha,
        100 * args.self_play_ratio, 100 * (1 - args.self_play_ratio),
        100 * args.math_mix_ratio, args.math_max_difficulty,
        args.overlong_filter,
        args.eval_every, args.eval_max_samples,
        args.max_grad_norm, args.save_every, args.keep_last,
    )

    # ── Load model ──────────────────────────────────────────────────────────
    logger.info("Loading model from %s ...", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # SFT adapter checkpoints often don't save chat_template, which causes
    # tokenizer.apply_chat_template() to raise an error inside evaluate_gsm8k
    # — silently swallowed there, giving 0% accuracy even for a capable model.
    # Mirror the fix from run_ppo_training_curriculum.py: load the template
    # from the base model when it's missing.
    if tokenizer.chat_template is None:
        _base_model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"
        _meta_file = Path(args.base_model) / "pipeline_meta.json"
        if _meta_file.exists():
            _meta = json.loads(_meta_file.read_text(encoding="utf-8"))
            _base_model_name = _meta.get("base_model", _base_model_name)
        logger.info(
            "Tokenizer has no chat_template; loading from base model %s", _base_model_name
        )
        try:
            _base_tok = AutoTokenizer.from_pretrained(_base_model_name, trust_remote_code=True)
            if _base_tok.chat_template is not None:
                tokenizer.chat_template = _base_tok.chat_template
                logger.info("Chat template loaded successfully.")
        except Exception as _e:
            logger.warning("Could not load chat template from base model: %s", _e)

    # PEFT <= 0.12 crashes inside merge_and_unload() when the
    # transformers.integrations.tensor_parallel module is missing.
    if "transformers.integrations.tensor_parallel" not in sys.modules:
        sys.modules["transformers.integrations.tensor_parallel"] = types.ModuleType(
            "tensor_parallel"
        )

    model_path = Path(args.base_model)
    is_adapter = (model_path / "adapter_config.json").exists()

    load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": device},
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )

    if is_adapter:
        # Determine actual base model from pipeline_meta.json (written by SFT pipeline).
        _meta_path = model_path / "pipeline_meta.json"
        _base_for_weights = "Qwen/Qwen2.5-Math-1.5B-Instruct"
        if _meta_path.exists():
            _base_for_weights = json.loads(
                _meta_path.read_text(encoding="utf-8")
            ).get("base_model", _base_for_weights)
        logger.info("Detected PEFT adapter — loading base %s then merging %s",
                    _base_for_weights, args.base_model)
        _base = AutoModelForCausalLM.from_pretrained(_base_for_weights, **load_kwargs)
        model = PeftModel.from_pretrained(_base, args.base_model).merge_and_unload()
        model = model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.base_model, **load_kwargs)

    # PEFT.merge_and_unload() leaves requires_grad=False on every param.
    # Re-enable unconditionally so GRPO's optimizer actually updates weights.
    params_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for p in model.parameters():
        p.requires_grad_(True)
    params_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if params_before == 0 and params_after > 0:
        logger.warning(
            "All parameters were frozen on load (PEFT merge_and_unload bug). "
            "Re-enabled requires_grad — any prior frozen runs were training nothing."
        )

    # Flash-Attn 2 turns attention memory from O(T²) to O(T), so gradient
    # checkpointing gives almost no extra saving while costing ~30% more
    # backward time.  Disable it when Flash is active (mirrors PPO runner).
    # gradient_checkpointing_enable requires use_reentrant=False on modern
    # PyTorch — the default True is deprecated and causes silent issues.
    # Also set use_cache=False: HF models can't use KV cache together with
    # gradient checkpointing (incompatible memory management).
    flash_active = attn_impl == "flash_attention_2"
    if not flash_active:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model, "config"):
            model.config.use_cache = False
        logger.info("Gradient checkpointing ENABLED (use_reentrant=False, use_cache=False).")
    else:
        logger.info(
            "Flash-Attn 2 active — gradient checkpointing OFF "
            "(Flash already gives O(T) attention memory)."
        )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable parameters: %s / %s (%.1f%%)",
        f"{n_params:,}", f"{n_total:,}", 100.0 * n_params / max(n_total, 1),
    )

    # ── Reference policy (frozen copy) ───────────────────────────────────────
    # A deep copy of the policy at t=0, kept frozen forever.  Used in the KL
    # penalty to anchor the policy against catastrophic forgetting of SFT
    # knowledge: L += β × (log π_θ - log π_ref) / T.
    # Memory cost: ~3 GB (1.5B × 2 bytes BF16) — negligible on 80 GB.
    ref_model: Optional[AutoModelForCausalLM] = None
    if args.kl_coef > 0.0:
        logger.info(
            "Creating frozen reference policy (kl_coef=%.4f, ~%.1f GB VRAM)...",
            args.kl_coef, sum(p.numel() for p in model.parameters()) * 2 / 1e9,
        )
        ref_model = copy.deepcopy(model)
        ref_model.requires_grad_(False)
        ref_model.eval()
        logger.info("Reference policy ready.")
    else:
        logger.info("KL coef = 0 — no reference policy created.")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        fused=torch.cuda.is_available(),
    )

    # ── LR schedule: linear warmup → cosine decay ────────────────────────────
    # Linear warmup avoids the large initial gradient spike when the policy
    # starts updating from an SFT checkpoint.  Cosine decay then smoothly
    # reduces LR toward min_lr as training progresses (standard in RLHF runs).
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    _n_warmup = max(1, args.warmup_iters)
    _n_total  = max(1, args.num_iterations)
    _n_decay  = max(1, _n_total - _n_warmup)
    _min_lr   = args.learning_rate * args.min_lr_ratio
    _warmup_sched = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=_n_warmup,
    )
    _cosine_sched = CosineAnnealingLR(
        optimizer,
        T_max=_n_decay,
        eta_min=_min_lr,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[_warmup_sched, _cosine_sched],
        milestones=[_n_warmup],
    )
    logger.info(
        "LR schedule: %.1e warmup(%d iters) → cosine decay(%d iters, min=%.1e)",
        args.learning_rate, _n_warmup, _n_decay, _min_lr,
    )

    # ── Load data ────────────────────────────────────────────────────────────
    gsm8k_pairs = load_gsm8k(args.gsm8k_data)
    if not gsm8k_pairs:
        logger.error("No GSM8K data found — cannot train. Exiting.")
        sys.exit(1)

    # Optional MATH dataset mixing
    math_pairs: List[Dict[str, str]] = []
    if args.math_mix_ratio > 0.0:
        math_pairs = load_math_dataset(
            local_path=args.math_data,
            max_difficulty=args.math_max_difficulty,
        )
        if math_pairs:
            logger.info(
                "MATH mixing: %.0f%% MATH (%d problems) + %.0f%% GSM8K (%d problems)",
                100 * args.math_mix_ratio, len(math_pairs),
                100 * (1 - args.math_mix_ratio), len(gsm8k_pairs),
            )
        else:
            logger.warning("No MATH pairs loaded — using GSM8K only.")

    # Combined pool used for difficulty sampling; kept separate for VRAM-aware
    # batch construction (sampler draws from each pool proportionally).
    qa_pairs = gsm8k_pairs  # for reward env (all GSM8K gold answers needed)

    # ── Load PRM (optional) ─────────────────────────────────────────────────
    prm: Optional[ProcessRewardScorer] = None
    if args.use_prm:
        try:
            prm = ProcessRewardScorer(
                model_name=args.prm_model,
                device=device,
                load_in_4bit=True,
            )
            logger.info("PRM loaded: %s (4-bit)", args.prm_model)
        except Exception as exc:
            logger.warning("PRM load failed (%s); running without PRM.", exc)

    # Build a minimal math_env just for its reward utilities (compute_grounded_reward).
    # value_model=None is safe: it's only stored as self.value and never invoked on
    # the grounded-reward path, so GRPO avoids the ~3 GB ValueHead backbone entirely.
    math_env = CurriculumMathEnvironment(
        policy_model=model,
        value_model=None,
        tokenizer=tokenizer,
        reference_questions=[],
        grounded_qa_pairs=qa_pairs,
        prm_scorer=prm,
        max_solution_tokens=args.max_new_tokens,
        device=device,
    )

    # ── Difficulty-adaptive sampling state ───────────────────────────────────
    # Track per-question win-rate.  Questions where the model scores correctly
    # 20-80% of the time are "on the margin" and provide the richest gradient
    # signal.  Questions it always gets right (win_rate≈1) or always gets wrong
    # (win_rate≈0) contribute little after the first few iterations.
    from collections import defaultdict
    _q_wins:     Dict[str, int] = defaultdict(int)
    _q_attempts: Dict[str, int] = defaultdict(int)

    def _question_key(q: str) -> str:
        """Short fingerprint — collision-safe enough for ~1k questions."""
        return q[:100]

    def _sample_by_difficulty(
        pool: List[Dict[str, str]], n: int, alpha: float
    ) -> List[Dict[str, str]]:
        """
        Sample ``n`` questions from ``pool``, weighting by how informative each is.

        Informativeness = 1 - |win_rate - 0.5| × 2   ∈ [0, 1]
          win_rate = 0.0 or 1.0  → informativeness = 0  (model already knows / lost cause)
          win_rate = 0.5         → informativeness = 1  (most uncertain = best signal)

        ``alpha`` sharpens the weighting (higher = stronger preference for win_rate≈0.5).
        Unseen questions get weight 0.75 to encourage exploration.
        A 5% floor prevents any question from being permanently excluded.
        """
        if alpha <= 0.0:
            return random.sample(pool, min(n, len(pool)))

        weights = []
        for qa in pool:
            key = _question_key(qa["question"])
            att = _q_attempts[key]
            if att == 0:
                w = 0.75
            else:
                win_rate = _q_wins[key] / att
                info = 1.0 - abs(win_rate - 0.5) * 2.0  # ∈ [0, 1]
                w = max(info ** alpha, 0.05)
            weights.append(w)

        total_w = sum(weights)
        probs = [w / total_w for w in weights]
        chosen = np.random.choice(
            len(pool), size=min(n, len(pool)), replace=False, p=probs
        )
        return [pool[i] for i in chosen]

    # ── Metrics log ─────────────────────────────────────────────────────────
    metrics_log: List[Dict] = []

    # ── Initial eval ─────────────────────────────────────────────────────────
    if not args.skip_initial_eval:
        logger.info("=" * 70)
        logger.info("INITIAL EVALUATION (Iteration 0)")
        logger.info("=" * 70)
        initial_eval = evaluate_policy(
            model, tokenizer,
            args.eval_data_path, args.eval_max_samples, args.eval_max_new_tokens,
            math_env=math_env,
            pass_at_k=args.eval_pass_at_k,
        )
        # accuracy == combined_score = 0.60×correct + 0.15×PRM + 0.15×SymPy + 0.10×format
        # This is identical to the GRPO training objective.
        _log_eval_result("INITIAL (iter 0)", initial_eval, best=None)
        metrics_log.append({"iteration": 0, **initial_eval})
        best_accuracy  = float(initial_eval.get("accuracy",     0.0))
        best_combined  = float(initial_eval.get("combined_score", 0.0))
        best_prm_mean  = float(initial_eval.get("prm_mean",     0.0))
    else:
        best_accuracy = 0.0
        best_combined = 0.0
        best_prm_mean = 0.0

    # ── Training ─────────────────────────────────────────────────────────────
    for iteration in range(1, args.num_iterations + 1):
        iter_start = time.perf_counter()
        logger.info("=" * 70)
        logger.info("GRPO ITERATION %d/%d", iteration, args.num_iterations)
        logger.info("=" * 70)

        # Sample questions — difficulty-weighted from the mixed pool.
        # When math_pairs is non-empty, draw proportionally: N*ratio from MATH
        # and N*(1-ratio) from GSM8K.  The difficulty sampler handles each pool
        # independently so MATH problems get their own win-rate tracking.
        if math_pairs and args.math_mix_ratio > 0.0:
            n_math  = max(1, round(args.questions_per_iter * args.math_mix_ratio))
            n_gsm8k = max(1, args.questions_per_iter - n_math)
            math_batch  = _sample_by_difficulty(math_pairs,  n_math,  alpha=args.difficulty_alpha)
            gsm8k_batch = _sample_by_difficulty(gsm8k_pairs, n_gsm8k, alpha=args.difficulty_alpha)
            questions_batch = math_batch + gsm8k_batch
            random.shuffle(questions_batch)
        else:
            questions_batch = _sample_by_difficulty(
                gsm8k_pairs, args.questions_per_iter, alpha=args.difficulty_alpha
            )
        cur_lr = optimizer.param_groups[0]["lr"]
        logger.info("LR this iteration: %.2e", cur_lr)

        all_rewards:   List[float] = []
        all_q_rewards: List[float] = []
        _grounded_rewards:   List[float] = []
        _sp_rewards:         List[float] = []
        _grounded_step_accs: List[float] = []   # step_accuracy from grounded rollouts
        # Per-component question quality accumulators
        _qc_topic:      List[float] = []
        _qc_diff:       List[float] = []
        _qc_clarity:    List[float] = []
        _qc_novelty:    List[float] = []
        _qc_solvability: List[float] = []

        skipped = 0
        n_groups = 0
        n_self_play = 0
        q_gen_attempts  = 0    # total generate_question() calls
        q_gen_valid     = 0    # non-empty questions produced (len > 10 chars)
        q_quality_good  = 0    # self-play groups where question_reward > 0.5
        total_loss_val = 0.0

        # Determine how many of this iteration's groups use self-play question
        # generation vs grounded (dataset) questions.
        n_self_play_target = int(round(len(questions_batch) * args.self_play_ratio))

        # Zero gradients once before the loop — we accumulate them via
        # per-group .backward() calls instead of building one giant graph.
        # Keeping all K*N forward passes alive until a single backward()
        # at the end would hold O(K*N) computation graphs in GPU memory
        # simultaneously (64 graphs at K=4, N=16), risking OOM.  Calling
        # .backward() immediately after each group frees that graph right
        # away; gradients accumulate in .grad tensors without extra memory.
        optimizer.zero_grad()

        pbar = tqdm(questions_batch, desc=f"Iter {iteration} GRPO groups", unit="q")
        for _group_idx, qa in enumerate(pbar):

            # ── Decide: self-play (model generates question) or grounded ─────
            # First n_self_play_target groups use self-play; the rest use the
            # grounded dataset question.  Interleaving is simple and stable.
            use_self_play = (
                args.self_play_ratio > 0.0
                and n_self_play < n_self_play_target
            )

            if use_self_play:
                # ── SELF-PLAY BRANCH ─────────────────────────────────────────
                # 1. Sample a curriculum instruction (topic + difficulty target)
                instruction, target_topic, target_difficulty = math_env.sample_instruction()

                # 2. Model generates the question from the instruction.
                #    This is the "proposer" role in Theme #4 self-improvement:
                #    the model creates its own challenge.
                q_gen_attempts += 1
                question = generate_question(
                    model=model,
                    tokenizer=tokenizer,
                    instruction=instruction,
                    max_new_tokens=128,   # questions are short
                    device=device,
                )
                # A valid question must have at least some substance.
                # Reject single-word, empty, or nonsensical outputs.
                if len(question.strip()) < 10:
                    logger.debug(
                        "Self-play: generated question too short (%d chars), skipping group.",
                        len(question.strip()),
                    )
                    skipped += 1
                    continue
                q_gen_valid += 1
                n_self_play += 1
                gold = None   # no gold answer — rewarded on question quality
            else:
                # ── GROUNDED BRANCH ──────────────────────────────────────────
                # Use pre-existing dataset question with known gold answer.
                question = qa["question"]
                gold = qa["gold_final"]
                target_topic = "grounded"
                target_difficulty = 0.5

            # --- Generate K solutions (batched — single model.generate call) ---
            solution_prompt = math_env.format_solution_prompt(question)
            solutions, input_ids_list, response_masks, old_log_probs_list = (
                generate_solutions_batched(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=solution_prompt,
                    K=args.group_size,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    device=device,
                )
            )

            # --- Overlong filter: drop truncated solutions (no Final Answer) ---
            # A response that hit max_new_tokens was cut off mid-generation;
            # it almost certainly didn't produce a valid "Final Answer: X" line,
            # so its reward is unreliable noise.  Dropping it keeps the group
            # advantage estimates clean.
            if args.overlong_filter:
                _valid = [
                    (sol, ids, mask, olp)
                    for sol, ids, mask, olp
                    in zip(solutions, input_ids_list, response_masks, old_log_probs_list)
                    if int(mask.sum().item()) < args.max_new_tokens
                ]
                if _valid:
                    solutions, input_ids_list, response_masks, old_log_probs_list = (
                        zip(*_valid)  # type: ignore[assignment]
                    )
                    solutions        = list(solutions)
                    input_ids_list   = list(input_ids_list)
                    response_masks   = list(response_masks)
                    old_log_probs_list = list(old_log_probs_list)
                else:
                    # All K solutions were truncated — skip group.
                    skipped += 1
                    continue

            # --- Score each solution (self-play: Q+S reward; grounded: S only) ---
            rewards = []
            _sp_q_rew_this_group: List[float] = []
            for sol in solutions:
                if use_self_play:
                    # compute_reward = 0.40×question_quality + 0.60×solution_quality
                    # This is the core Theme #4 signal: the model is rewarded
                    # for generating a well-formed, appropriately difficult,
                    # solvable question AND for solving it correctly.
                    r, q_rew, _, q_met = compute_self_play_reward(
                        question=question,
                        solution=sol,
                        target_topic=target_topic,
                        target_difficulty=target_difficulty,
                        math_env=math_env,
                    )
                    _sp_q_rew_this_group.append(q_rew)
                    all_q_rewards.append(q_rew)
                    # Collect per-component breakdown (same question, all K solutions
                    # get the same q_metrics — average to reduce noise).
                    _qc_topic.append(q_met["topic_match"])
                    _qc_diff.append(q_met["difficulty_fit"])
                    _qc_clarity.append(q_met["clarity"])
                    _qc_novelty.append(q_met["novelty"])
                    _qc_solvability.append(q_met["solvability"])
                else:
                    r_dict = compute_grounded_reward(
                        question=question,
                        solution=sol,
                        gold_final=gold,
                        math_env=math_env,
                    )
                    r = r_dict["combined_score"]
                    _grounded_step_accs.append(r_dict["step_accuracy"])
                rewards.append(r)
            all_rewards.extend(rewards)
            # Route to path-specific accumulators for separate batch_acc reporting
            if use_self_play:
                _sp_rewards.extend(rewards)
            else:
                _grounded_rewards.extend(rewards)

            # A self-play group is "accurate" if the question it generated scored
            # above 0.5 on question quality — meaning it was clear, on-topic,
            # appropriately difficult, and solvable.
            if use_self_play and _sp_q_rew_this_group:
                if float(np.mean(_sp_q_rew_this_group)) > 0.5:
                    q_quality_good += 1

            # --- Update difficulty stats (grounded questions only — self-play
            #     questions are ephemeral and have no stable key) ---
            if not use_self_play:
                _key = _question_key(question)
                _q_attempts[_key] += len(solutions)
                _q_wins[_key] += sum(1 for r in rewards if r > 0.5)

            # --- GRPO loss (IS clip + optional KL penalty) + immediate backward ---
            group_loss = grpo_loss_for_group(
                model=model,
                input_ids_list=input_ids_list,
                response_masks=response_masks,
                rewards=rewards,
                old_log_probs=old_log_probs_list,
                clip_eps=args.clip_eps,
                kl_coef=args.kl_coef,
                ref_model=ref_model,
            )

            if group_loss is None:
                skipped += 1
                _pf: Dict = dict(mean_r=f"{np.mean(rewards):.3f}", skip=skipped, loss="skip")
                if n_self_play > 0 and all_q_rewards:
                    _q_acc_pct = 100.0 * q_quality_good / max(1, n_self_play)
                    _pf["q_acc"] = f"{_q_acc_pct:.0f}%"
                pbar.set_postfix(**_pf)
                continue

            # Backprop immediately — frees this group's computation graph.
            # Gradients from all valid groups accumulate in param.grad.
            group_loss.backward()
            total_loss_val += group_loss.item()
            n_groups += 1
            _pf = dict(
                mean_r=f"{np.mean(rewards):.3f}",
                loss=f"{group_loss.item():.4f}",
                skip=skipped,
            )
            if n_self_play > 0 and all_q_rewards:
                # Show live question-gen accuracy in the tqdm bar.
                # q_acc = fraction of self-play groups whose generated question
                # scored > 0.5 on quality (clear, on-topic, solvable).
                _q_acc_pct = 100.0 * q_quality_good / max(1, n_self_play)
                _pf["q_acc"] = f"{_q_acc_pct:.0f}%"
                _pf["q_rew"]  = f"{float(np.mean(all_q_rewards)):.3f}"
            pbar.set_postfix(**_pf)

        # --- Gradient step: normalise accumulated grads then step ---
        if n_groups > 0:
            # Divide accumulated grads by n_groups to get the true average
            # (equivalent to averaging the group losses before backward).
            if n_groups > 1:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.div_(n_groups)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.max_grad_norm,
            )
            optimizer.step()
            loss_val = total_loss_val / n_groups
        else:
            loss_val = 0.0
        scheduler.step()

        iter_time = time.perf_counter() - iter_start
        mean_r   = float(np.mean(all_rewards))             if all_rewards   else 0.0
        std_r    = float(np.std(all_rewards))              if all_rewards   else 0.0
        acc_r    = float(np.mean([r > 0.5 for r in all_rewards])) if all_rewards else 0.0
        grounded_acc_r = (
            float(np.mean([r > 0.5 for r in _grounded_rewards]))
            if _grounded_rewards else 0.0
        )
        mean_step_acc = (
            float(np.mean(_grounded_step_accs))
            if _grounded_step_accs else 0.0
        )
        mean_q_r = float(np.mean(all_q_rewards)) if all_q_rewards else 0.0

        # Question generation accuracy metrics (self-play only)
        q_gen_valid_rate = (q_gen_valid   / q_gen_attempts)  if q_gen_attempts  > 0 else 0.0
        q_quality_rate   = (q_quality_good / n_self_play)    if n_self_play     > 0 else 0.0
        # Per-component averages (all non-empty across K solutions × groups)
        mean_q_topic     = float(np.mean(_qc_topic))       if _qc_topic      else 0.0
        mean_q_diff      = float(np.mean(_qc_diff))        if _qc_diff       else 0.0
        mean_q_clarity   = float(np.mean(_qc_clarity))     if _qc_clarity    else 0.0
        mean_q_novelty   = float(np.mean(_qc_novelty))     if _qc_novelty    else 0.0
        mean_q_solvab    = float(np.mean(_qc_solvability)) if _qc_solvability else 0.0

        _cur_lr = optimizer.param_groups[0]["lr"]

        # ── Primary summary line ─────────────────────────────────────────────
        logger.info(
            "Iter %d | loss=%.4f | reward mean=%.3f std=%.3f | "
            "grounded_acc=%.1f%% | step_acc=%.1f%% | batch_acc=%.1f%% | "
            "groups=%d skipped=%d | lr=%.2e | %.1fs",
            iteration, loss_val, mean_r, std_r,
            100 * grounded_acc_r,  # grounded final-answer pass rate
            100 * mean_step_acc,   # fraction of reasoning steps rated correct by PRM
            100 * acc_r,           # all rollouts incl. self-play
            n_groups, skipped, _cur_lr, iter_time,
        )

        # ── Question-generation accuracy line (only when self-play is active) ─
        if n_self_play > 0:
            logger.info(
                "  Question generation: %d/%d valid (%.0f%%) | "
                "q_reward=%.3f | q_acc=%.1f%% (>0.5 quality) | "
                "topic=%.2f diff=%.2f clarity=%.2f novelty=%.2f solvability=%.2f",
                q_gen_valid, q_gen_attempts, 100 * q_gen_valid_rate,
                mean_q_r, 100 * q_quality_rate,
                mean_q_topic, mean_q_diff, mean_q_clarity,
                mean_q_novelty, mean_q_solvab,
            )

        iter_metrics: Dict = {
            "iteration":             iteration,
            "loss":                  loss_val,
            "mean_reward":           mean_r,
            "std_reward":            std_r,
            "batch_accuracy":        acc_r,
            "grounded_accuracy":     grounded_acc_r,
            "step_accuracy":         mean_step_acc,
            "n_groups":              n_groups,
            "skipped_groups":        skipped,
            "learning_rate":         _cur_lr,
            "iter_time_s":           iter_time,
            # ── Question-generation metrics ─────────────────────────────────
            "n_self_play_groups":    n_self_play,
            "q_gen_attempts":        q_gen_attempts,
            "q_gen_valid":           q_gen_valid,
            "q_gen_valid_rate":      round(q_gen_valid_rate, 4),
            "mean_question_reward":  round(mean_q_r, 4),
            "q_quality_rate":        round(q_quality_rate, 4),
            "q_topic_match":         round(mean_q_topic,   4),
            "q_difficulty_fit":      round(mean_q_diff,    4),
            "q_clarity":             round(mean_q_clarity, 4),
            "q_novelty":             round(mean_q_novelty, 4),
            "q_solvability":         round(mean_q_solvab,  4),
        }

        # --- Eval ---
        if iteration % args.eval_every == 0:
            logger.info("Evaluating GSM8K (%d samples)...", args.eval_max_samples)
            eval_res = evaluate_policy(
                model, tokenizer,
                args.eval_data_path, args.eval_max_samples, args.eval_max_new_tokens,
                math_env=math_env,
                pass_at_k=args.eval_pass_at_k,
            )
            # accuracy == combined_score: 0.60×correct + 0.15×PRM + 0.15×SymPy + 0.10×format
            cur_combined = float(eval_res.get("combined_score", best_combined))
            cur_prm_mean = float(eval_res.get("prm_mean",       best_prm_mean))

            _log_eval_result(f"iter {iteration}", eval_res, best=best_combined)

            # ── Checkpoint: save when combined_score strictly improves ────────
            # combined_score is a continuous variable; any improvement in
            # correctness, PRM quality, SymPy, or format moves it.
            if cur_combined > best_combined + 1e-4:
                reason = f"combined {cur_combined:.4f} > {best_combined:.4f}"
                best_combined  = cur_combined
                best_prm_mean  = max(best_prm_mean, cur_prm_mean)
                best_accuracy  = best_combined
                best_path = out_dir / "best_policy"
                model.save_pretrained(str(best_path))
                tokenizer.save_pretrained(str(best_path))
                logger.info("New best saved → %s  (%s)", best_path, reason)

            iter_metrics.update(eval_res)

        # --- Save checkpoint (respect --save-every / --keep-last) ---
        is_last_iter = iteration == args.num_iterations
        should_save = is_last_iter or (
            args.save_every > 0 and iteration % args.save_every == 0
        )
        if should_save:
            ckpt_path = out_dir / f"iter_{iteration:04d}"
            ckpt_path.mkdir(exist_ok=True)
            model.save_pretrained(str(ckpt_path))
            tokenizer.save_pretrained(str(ckpt_path))

            # Prune older iter_* checkpoints beyond the rolling window.
            if args.keep_last and args.keep_last > 0:
                existing = sorted(
                    p for p in out_dir.iterdir()
                    if p.is_dir() and p.name.startswith("iter_")
                )
                to_remove = existing[: -args.keep_last]
                for old in to_remove:
                    try:
                        shutil.rmtree(old)
                        logger.info("Pruned old checkpoint: %s", old.name)
                    except OSError as exc:
                        logger.warning("Could not prune %s: %s", old.name, exc)

        # ── Write metrics to both JSONL (full history) and CSV (live row) ────
        metrics_log.append(iter_metrics)
        (out_dir / "metrics.jsonl").write_text(
            "\n".join(json.dumps(m) for m in metrics_log), encoding="utf-8"
        )
        # CSV: one row per iteration, flushed immediately so you can
        # `tail -f logs/grpo/<run>/metrics.csv` or open it in Excel mid-run.
        _append_metrics_csv({
            "iteration":      iter_metrics["iteration"],
            "timestamp":      datetime.now().isoformat(timespec="seconds"),
            "loss":           iter_metrics.get("loss", 0.0),
            "mean_reward":    iter_metrics.get("mean_reward", 0.0),
            "std_reward":     iter_metrics.get("std_reward", 0.0),
            "batch_accuracy": iter_metrics.get("batch_accuracy", 0.0),
            "n_groups":       iter_metrics.get("n_groups", 0),
            "skipped_groups": iter_metrics.get("skipped_groups", 0),
            "learning_rate":  iter_metrics.get("learning_rate", 0.0),
            "iter_time_s":    iter_metrics.get("iter_time_s", 0.0),
            # Training-objective eval (primary; same formula as GRPO reward)
            "gsm8k_combined":   iter_metrics.get("combined_score",        ""),
            "gsm8k_correct_rt": iter_metrics.get("correct_rate",          ""),
            "gsm8k_prm":        iter_metrics.get("prm_mean",              ""),
            "gsm8k_sympy":      iter_metrics.get("sympy_mean",            ""),
            "gsm8k_format":     iter_metrics.get("format_mean",           ""),
            "gsm8k_n_scored":   iter_metrics.get("n_scored",              ""),
            # Debug (not used for checkpointing)
            "gsm8k_final_ans":  iter_metrics.get("final_answer_accuracy", ""),
        })

    logger.info("=" * 70)
    logger.info("GRPO training complete.")
    logger.info(
        "Best training-objective score : %.4f  "
        "(0.60×correct + 0.15×PRM + 0.15×SymPy + 0.10×format)",
        best_combined,
    )
    logger.info("Best PRM component mean       : %.3f", best_prm_mean)
    logger.info("Checkpoints                   : %s", out_dir)
    logger.info("Logs                          : %s", log_dir)
    logger.info("Console log                   : %s", console_log_path)
    logger.info("=" * 70)

    # ── Final summary ─────────────────────────────────────────────────────────
    summary: Dict[str, Any] = {
        "run_name":          run_name,
        "best_accuracy":     best_combined,   # accuracy == combined_score
        "best_combined":     best_combined,
        "best_prm_mean":     best_prm_mean,
        "total_iterations":  args.num_iterations,
        "checkpoints_dir":   str(out_dir),
        "log_dir":           str(log_dir),
        "console_log":       str(console_log_path),
        "metrics_csv":       str(_metrics_csv_path),
        "metrics_jsonl":     str(out_dir / "metrics.jsonl"),
    }
    (log_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    logger.info("Summary written to %s", log_dir / "summary.json")

    # ── Auto-generate demo plots ───────────────────────────────────────────────
    _metrics_jsonl = out_dir / "metrics.jsonl"
    if _metrics_jsonl.exists():
        try:
            import importlib
            if importlib.util.find_spec("matplotlib") is None:
                logger.warning(
                    "matplotlib not installed — skipping auto-plot. "
                    "Install with: pip install matplotlib  then run: "
                    "python scripts/plot_grpo_run.py %s",
                    _metrics_jsonl,
                )
            else:
                from scripts.plot_grpo_run import generate_plots as _gen_plots
                _plot_dir = _gen_plots(_metrics_jsonl)
                logger.info("Plots saved → %s", _plot_dir)
        except Exception as _plot_exc:
            logger.warning(
                "Plot generation failed (%s: %s). "
                "Run manually: python scripts/plot_grpo_run.py %s",
                type(_plot_exc).__name__, _plot_exc, _metrics_jsonl,
            )

    # Explicit teardown (atexit is the safety net for crashes; calling here
    # ensures everything is flushed and closed before the process returns
    # normally — atexit won't double-close because _teardown_logging is
    # idempotent via the .closed checks).
    _teardown_logging()


if __name__ == "__main__":
    main()
