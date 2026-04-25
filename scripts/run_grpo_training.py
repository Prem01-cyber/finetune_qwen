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
from src.config.prompts import create_generator_messages

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
                pairs.append({"question": question, "gold_final": gold, "topic": "arithmetic", "difficulty": 2})
    logger.info("Loaded %d GSM8K QA pairs from %s", len(pairs), path)
    return pairs


# ---------------------------------------------------------------------------
# MATH harder dataset
# ---------------------------------------------------------------------------

def _extract_boxed(text: str) -> Optional[str]:
    r"""Extract the content of the first ``\boxed{...}`` in *text*.

    Handles one level of nested braces so that answers like ``\frac{3}{4}``
    and ``\sqrt{2}`` are captured correctly.  The previous ``[^}]*`` regex
    stopped at the first ``}`` and produced truncated strings like ``\frac{3``
    for ``\boxed{\frac{3}{4}}``, silently dropping the example.
    """
    start = text.find(r"\boxed{")
    if start == -1:
        return None
    i = start + len(r"\boxed{")
    depth = 1
    content_start = i
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return text[content_start : i - 1].strip()


def _boxed_to_numeric(answer: str) -> Optional[str]:
    """
    Convert a ``\\boxed{...}`` answer to a verifiable numeric string.

    Returns one of:
      - An integer string ``"42"`` for whole-number answers.
      - A decimal string ``"3.5"`` only when the value is exactly representable
        (i.e. terminates in decimal).  Non-terminating fractions (e.g. 1/3) are
        stored as ``"1/3"`` to avoid the precision loss that occurred when they
        were rounded to ``"0.3333"`` — SymPy's ``simplify(Rational(1,3) -
        Float(0.3333))`` is non-zero, causing false negatives in answer matching.
      - ``None`` for symbolic / multi-part answers (``3\\sqrt{2}``, ``(1,2)``).

    Handles both plain fractions (``3/4``) and LaTeX fractions
    (``\\frac{3}{4}``) now that ``_extract_boxed`` correctly extracts
    nested-brace content.
    """
    ans = answer.strip()
    # Direct integer
    try:
        return str(int(ans))
    except ValueError:
        pass
    # Direct float — only accept if it is exactly representable (terminates).
    try:
        v = float(ans)
        if v == int(v):
            return str(int(v))
        # Check for exact decimal representation by round-tripping.
        if float(f"{v}") == v:
            return str(v)
    except ValueError:
        pass
    # LaTeX fraction  \frac{num}{den}  — store as "num/den", not as float
    m = re.fullmatch(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", ans)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        if den:
            if num % den == 0:
                return str(num // den)
            return f"{num}/{den}"
    # Plain fraction  num/den
    m = re.fullmatch(r"(-?\d+)/(-?\d+)", ans)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        if den:
            if num % den == 0:
                return str(num // den)
            return f"{num}/{den}"
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
        
        # Determine topic — map all 7 MATH dataset categories rather than
        # collapsing Algebra/Geometry/Precalculus into a single "other" bucket
        # (which made curriculum win-rate tracking blind to these large categories).
        t_str = item.get("type", "")
        if "Counting & Probability" in t_str:
            topic = "combinatorics"
        elif "Number Theory" in t_str:
            topic = "number_theory"
        elif "Prealgebra" in t_str:
            topic = "arithmetic"
        elif "Algebra" in t_str:
            topic = "algebra"
        elif "Geometry" in t_str:
            topic = "geometry"
        elif "Precalculus" in t_str:
            topic = "precalculus"
        elif "Intermediate Algebra" in t_str:
            topic = "algebra"
        else:
            topic = "other"

        question = item.get("problem", "").strip()
        solution = item.get("solution", "")
        boxed    = _extract_boxed(solution)
        if not boxed:
            continue
        numeric  = _boxed_to_numeric(boxed)
        if not numeric:
            continue
        pairs.append({"question": question, "gold_final": numeric, "topic": topic, "difficulty": level})

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
# Big-Math dataset
# ---------------------------------------------------------------------------

def load_big_math(max_difficulty: int = 3) -> List[Dict[str, str]]:
    """Load Big-Math-RL-Verified dataset and convert to unified format."""
    pairs = []
    try:
        from datasets import load_dataset
        ds = load_dataset("SynthLabsAI/Big-Math-RL-Verified", split="train")
    except Exception as e:
        logger.warning(f"Big-Math unavailable ({e}). Skipping.")
        return pairs

    KEYWORDS = {
        "combinat": "combinatorics", "permut": "combinatorics",
        "sequen": "sequences", "series": "sequences",
        "probab": "probability",
    }

    def get_topic(subject):
        if not subject:
            return "other"
        s = subject.lower()
        for kw, topic in KEYWORDS.items():
            if kw in s:
                return topic
        return "other"

    skipped_symbolic = 0
    for row in ds:
        topic = get_topic(row.get("subject", ""))
        answer = str(row.get("answer", "")).strip()
        if not answer:
            continue
        diff = min(int(row.get("difficulty", 2)), 3)
        if diff > max_difficulty:
            continue
        # Validate that the answer is numerically verifiable via SymPy, the
        # same way load_math_dataset does.  Big-Math-RL-Verified contains
        # symbolic answers (\sqrt{3}, \pi/2, set-form, etc.) that will always
        # produce reward=0 in _answers_equivalent (parse failure), silently
        # wasting grounded training slots.
        #
        # Strategy: try _boxed_to_numeric first (handles integers, plain
        # fractions, simple decimals); if that fails, accept the raw string
        # only if SymPy can parse it without error (catches clean expressions
        # like "2/3" that _boxed_to_numeric also accepts, but rejects
        # '\sqrt{3}', 'x+1', etc.).
        numeric_answer = _boxed_to_numeric(answer)
        if numeric_answer is None:
            # Attempt a direct SymPy parse of the raw answer string.
            try:
                from sympy.parsing.sympy_parser import parse_expr as _pe
                from src.sft.sympy_normalize import normalize_for_parse_expr as _nfp
                _parsed = _pe(_nfp(answer))
                # Reject symbolic results (contain free symbols like x, sqrt, pi).
                if _parsed.free_symbols:
                    skipped_symbolic += 1
                    continue
                numeric_answer = answer   # keep raw string; SymPy will handle it at reward time
            except Exception:
                skipped_symbolic += 1
                continue
        pairs.append({
            "question": row["problem"],
            "gold_final": numeric_answer,
            "topic": topic,
            "difficulty": diff,
        })

    logger.info(
        "Loaded %d Big-Math pairs (%d symbolic/unparseable answers skipped)",
        len(pairs), skipped_symbolic,
    )
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

    Training reward is BINARY:
      combined_score = 1.0 if predicted final answer == gold_final, else 0.0

    Additional fields are diagnostic only (not part of the training signal):
      step_accuracy   – fraction of PRM steps rated > 0.5
      prm_mean_score  – PRM mean score across all steps
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
        "lccp":            float(result.get("lccp",            0.0)),
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
    # Use the gated question reward (zeroed when solution is invalid) — this is
    # what actually contributed to combined_score, not the raw overall_score.
    q_reward = float(result.get("effective_question_reward", q_metrics_raw.get("overall_score", 0.0)))
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
    temperature: float = 0.85,
) -> str:
    """Generate a math question from a curriculum instruction.

    Uses centralized prompts from src/config/prompts.py to ensure consistency
    across SFT training, GRPO, PPO, and inference.

    Returns the raw decoded question text (no special tokens).
    """
    # Use centralized prompt configuration
    messages = create_generator_messages(instruction)
    
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback if chat template is missing
        system = messages[0]["content"]
        user = messages[1]["content"]
        prompt = f"{system}\n\n{user}\n"

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
        temperature=temperature,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=stop_ids or None,
        use_cache=True,
    )
    return tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_questions_batched(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    instruction: str,
    K_q: int,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> Tuple[List[str], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Generate K_q question candidates from a single curriculum instruction in
    one batched model.generate() call.  Returns the same four-tuple as
    ``generate_solutions_batched`` so the question token IDs can be passed
    directly to ``grpo_loss_for_group`` for the question-level GRPO update.

    Uses the same centralized prompts (``create_generator_messages``) as
    ``generate_question()`` so the chat format is identical whether running
    single-question or batched two-phase generation.

    Returns:
        questions       : K_q decoded question strings
        input_ids_list  : K_q full (prompt+response) token ID tensors
        response_masks  : K_q bool masks (True = non-pad response token)
        old_log_probs   : K_q scalar tensors (sum log π_old over response),
                          no_grad — used as denominator in IS ratio.
    """
    messages = create_generator_messages(instruction)
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # Fallback: mirrors the identical block in generate_question().
        # 'system' was previously undefined here, causing a NameError crash
        # whenever apply_chat_template raised (e.g. missing chat template).
        _system = messages[0]["content"]
        prompt = f"{_system}\n\n{instruction}\n"

    stop_ids = _build_stop_token_ids(tokenizer)
    pad_id: int = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    enc = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512
    ).to(device)
    prompt_len: int = enc["input_ids"].shape[1]

    input_ids_batch = enc["input_ids"].expand(K_q, -1).contiguous()
    attn_mask_batch = enc["attention_mask"].expand(K_q, -1).contiguous()

    model.eval()
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids_batch,
            attention_mask=attn_mask_batch,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=pad_id,
            eos_token_id=stop_ids,
            use_cache=True,
        )

    questions: List[str] = []
    input_ids_list: List[torch.Tensor] = []
    response_masks: List[torch.Tensor] = []

    pad_id_t = torch.tensor(pad_id, device=device, dtype=out.dtype)
    for i in range(K_q):
        full_ids = out[i]
        response_section = full_ids[prompt_len:]
        mask = torch.zeros(full_ids.shape[0], dtype=torch.bool, device=device)
        mask[prompt_len:] = response_section != pad_id_t
        question = tokenizer.decode(response_section, skip_special_tokens=True).strip()
        questions.append(question)
        input_ids_list.append(full_ids)
        response_masks.append(mask)

    # Single batched forward pass for all K_q old log-probs (same trick as solutions).
    old_log_probs: List[torch.Tensor] = []
    with torch.no_grad():
        attn_mask_lp = (out != pad_id_t)
        attn_mask_lp[:, :prompt_len] = True
        batch_logits = model(
            input_ids=out,
            attention_mask=attn_mask_lp.long(),
            use_cache=False,
            return_dict=True,
        ).logits  # [K_q, total_len, vocab]

        for i in range(K_q):
            full_ids = out[i]
            mask = response_masks[i]
            shift_logits = batch_logits[i, :-1]
            shift_labels  = full_ids[1:]
            shift_mask    = mask[1:]
            lp_tokens = F.log_softmax(shift_logits, dim=-1)[
                torch.arange(shift_logits.size(0), device=device),
                shift_labels,
            ]
            resp_lps = lp_tokens[shift_mask]
            old_log_probs.append(
                resp_lps.sum().detach() if resp_lps.numel() > 0
                else torch.tensor(0.0, device=device)
            )

    return questions, input_ids_list, response_masks, old_log_probs

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
        old_log_probs   : K 1D tensors, log π_old(token) for each response token,
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
                resp_lps.detach() if resp_lps.numel() > 0
                else torch.empty(0, device=device)
            )

    return solutions, input_ids_list, response_masks, old_log_probs


def compute_sequence_log_prob(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Forward pass through model to get log probs for response tokens.

    Returns 1D tensor of log probs (differentiable).
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

    # Return log probs for response tokens only
    response_log_probs = token_log_probs[shift_mask]
    if response_log_probs.numel() == 0:
        return torch.empty(0, requires_grad=True, device=input_ids.device)
    return response_log_probs


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
        n_response = new_lp.numel()
        if n_response == 0:
            continue

        adv_t = torch.tensor(adv, dtype=new_lp.dtype, device=_device)
        old_lp_t = old_lp.to(_device).detach()

        # ── GRPO surrogate (with optional IS clip) ────────────────────────
        # MUST be computed per-token, not per-sequence!
        if clip_eps > 0:
            ratio = torch.exp(new_lp - old_lp_t)
            surr_unclipped = ratio * adv_t
            surr_clipped   = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t
            loss_i = -torch.min(surr_unclipped, surr_clipped).mean()
        else:
            loss_i = -(adv_t * new_lp).mean()

        # ── Reference-policy KL penalty ───────────────────────────────────
        # DeepSeekMath unbiased KL estimator: exp(ref_lp - new_lp) - (ref_lp - new_lp) - 1
        if kl_coef > 0.0 and ref_model is not None:
            with torch.no_grad():
                ref_lp = compute_sequence_log_prob(ref_model, ids, mask).to(_device).detach()
            
            diff = ref_lp - new_lp
            kl_per_token = torch.exp(diff) - diff - 1.0
            loss_i = loss_i + kl_coef * kl_per_token.mean()

        group_loss = group_loss + loss_i
        n_valid += 1

    if n_valid == 0:
        return None
    return group_loss / n_valid


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _log_eval_result(label: str, res: Dict, best: Optional[float]) -> None:
    """Print a structured evaluation summary.

    Training reward (grounded path) is binary: 1.0 if the final answer matches
    gold, 0.0 otherwise.  PRM, step accuracy, LCCP, and format are logged for
    diagnostics but do NOT contribute to the training signal.
    """
    cs      = float(res.get("combined_score",  0.0))
    cr      = float(res.get("correct_rate",    0.0))
    step_a  = float(res.get("step_accuracy",   0.0))
    lccp    = float(res.get("lccp",            0.0))
    prm     = float(res.get("prm_mean",        0.0))
    prm_fin = float(res.get("prm_final",       0.0))
    fmt     = float(res.get("format_mean",     0.0))
    n_sc    = int(res.get("n_scored", res.get("total", 0)))
    fa_acc  = float(res.get("final_answer_accuracy", cr))
    pak     = res.get("pass_at_k")
    pak_k   = int(res.get("pass_at_k_k", 4))

    best_str = f" (best={best:.4f})" if best is not None else ""
    logger.info(
        "Eval Score  [%s]: %.4f%s  |  n=%d",
        label, cs, best_str, n_sc,
    )
    logger.info(
        "  Training reward : BINARY (correct=1.0 / wrong=0.0)  "
        "→ correct_rate=%.1f%%",
        100 * cr,
    )
    logger.info(
        "  Diagnostics (not part of training reward):",
    )
    logger.info(
        "    PRM : prm_mean=%.3f  prm_final=%.3f  fmt=%.3f",
        prm, prm_fin, fmt,
    )
    logger.info(
        "    Step accuracy : %.1f%%  (fraction of PRM steps >0.5)",
        100 * step_a,
    )
    logger.info(
        "    Chain integrity (LCCP): %.1f%%  ← steps before first failure\n"
        "      [LCCP=100%% → all steps correct; LCCP=0%% → first step wrong]",
        100 * lccp,
    )
    if pak is not None:
        logger.info(
            "    pass@%d (T=0.8): %.1f%%  |  greedy correct: %.1f%%  "
            "← ceiling vs floor gap",
            pak_k, 100 * pak, 100 * cr,
        )
    logger.info(
        "  Final-answer accuracy (checkpoint metric): %.1f%%",
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
    """Run GSM8K evaluation aligned with the GRPO training signal.

    When *math_env* is supplied, ``math_env.compute_grounded_reward`` is used
    to score each prediction.  The grounded reward is BINARY: 1.0 if the
    predicted final answer is mathematically equivalent to the gold answer,
    0.0 otherwise.  PRM scores, step accuracy, LCCP, and format are recorded
    as diagnostic fields but do not affect the reward or the checkpoint metric.

    The checkpoint metric is ``final_answer_accuracy`` (exact binary match),
    which directly measures whether the model gets the math right — not whether
    it writes well-formatted wrong answers.
    """
    if not Path(eval_data_path).exists():
        return {"accuracy": 0.0, "combined_score": 0.0, "total": 0}
    model.eval()

    reward_fn = None
    if math_env is not None:
        import logging as _log_mod
        _mec_logger  = _log_mod.getLogger("src.rl.math_environment_curriculum")
        _prm_logger  = _log_mod.getLogger("src.rl.prm_scorer")

        def reward_fn(question: str, solution: str, gold: str) -> Dict:
            """Thin wrapper that silences per-sample INFO logs during eval."""
            _old_mec = _mec_logger.level
            _old_prm = _prm_logger.level
            _mec_logger.setLevel(_log_mod.WARNING)
            _prm_logger.setLevel(_log_mod.WARNING)
            try:
                return math_env.compute_grounded_reward(question, solution, gold)
            finally:
                _mec_logger.setLevel(_old_mec)
                _prm_logger.setLevel(_old_prm)

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
        "--q-group-size", type=int, default=1,
        help="K_q: question candidates per self-play group (default 1 = disabled). "
             "When ≥2, a second question-level GRPO update is added: K_q questions are "
             "sampled from the same instruction, each solved group-size times; the "
             "per-question reward (mean of its M solution rewards) drives a GRPO update "
             "on the question tokens.  Recommended: 2 with --group-size 4 to keep "
             "total self-play compute the same as K_q=1 with group-size 8.",
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
        "--math-mix-ratio-late", type=float, default=None,
        help="If set, ramp MATH fraction from --math-mix-ratio to this value "
             "starting at iter 15 (linear ramp over next 10 iters). "
             "Example: --math-mix-ratio 0.3 --math-mix-ratio-late 0.5 "
             "raises difficulty progressively once the policy is stable.",
    )
    parser.add_argument(
        "--math-ramp-start", type=int, default=15,
        help="Iteration at which to begin the MATH ratio ramp. Default 15.",
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
        "Run config: K=%d K_q=%d N=%d lr=%.1e T=%.2f max_new=%d | "
        "clip_eps=%.2f kl_coef=%.4f warmup=%d | diff_alpha=%.1f | "
        "self_play=%.0f%% grounded=%.0f%% | "
        "math_mix=%.0f%% math_maxdiff=%d | overlong_filter=%s | "
        "eval_every=%d eval_N=%d | grad_clip=%.2f save_every=%d keep_last=%d | "
        "question_GRPO=%s",
        args.group_size, args.q_group_size, args.questions_per_iter, args.learning_rate,
        args.temperature, args.max_new_tokens,
        args.clip_eps, args.kl_coef, args.warmup_iters,
        args.difficulty_alpha,
        100 * args.self_play_ratio, 100 * (1 - args.self_play_ratio),
        100 * args.math_mix_ratio, args.math_max_difficulty,
        args.overlong_filter,
        args.eval_every, args.eval_max_samples,
        args.max_grad_norm, args.save_every, args.keep_last,
        f"ENABLED (K_q={args.q_group_size})" if args.q_group_size > 1 else "disabled",
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

    # MATH dataset mixing
    math_pairs: List[Dict[str, str]] = load_math_dataset(
        local_path=args.math_data,
        max_difficulty=args.math_max_difficulty,
    )

    # Big-Math dataset mixing
    big_math_pairs: List[Dict[str, str]] = load_big_math(
        max_difficulty=args.math_max_difficulty,
    )

    # Combined pool used for difficulty sampling
    qa_pairs = gsm8k_pairs + math_pairs + big_math_pairs
    
    if not qa_pairs:
        logger.error("No data found — cannot train. Exiting.")
        sys.exit(1)

    logger.info(
        "Dataset built: %d total (GSM8K: %d, MATH: %d, Big-Math: %d)",
        len(qa_pairs), len(gsm8k_pairs), len(math_pairs), len(big_math_pairs)
    )
    
    # Print topic breakdown
    from collections import defaultdict
    topic_counts = defaultdict(int)
    for row in qa_pairs:
        topic_counts[row.get("topic", "other")] += 1
    for topic, count in sorted(topic_counts.items()):
        logger.info("  %s: %d", topic, count)

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
        # Feed all GSM8K training questions as the novelty reference set so
        # session_novelty is measured against the actual training distribution —
        # a self-play question that mimics a dataset question gets low novelty.
        reference_questions=[p["question"] for p in gsm8k_pairs],
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
    _q_wins:     Dict[str, int] = defaultdict(int)
    _q_attempts: Dict[str, int] = defaultdict(int)
    _topic_wins: Dict[str, int] = defaultdict(int)
    _topic_attempts: Dict[str, int] = defaultdict(int)

    def _question_key(q: str) -> str:
        """Stable hash fingerprint — collision-resistant for any pool size."""
        import hashlib
        return hashlib.md5(q.encode(), usedforsecurity=False).hexdigest()

    # Separate source pools for stratified sampling so --math-mix-ratio is
    # actually respected.  Previously everything was concatenated into qa_pairs
    # and sampled uniformly, which made effective MATH exposure ≈ 5% regardless
    # of the flag value.
    _gsm8k_pool = gsm8k_pairs
    _math_pool  = math_pairs + big_math_pairs   # MATH + Big-Math treated as one harder pool

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

        Issue #9 fix: emits a WARNING when the pool is smaller than ``n`` so
        callers are not silently surprised by a shrunken batch.
        """
        if not pool:
            return []

        actual_n = min(n, len(pool))
        if actual_n < n:
            logger.warning(
                "_sample_by_difficulty: requested %d questions but pool only has %d; "
                "batch will be smaller than --questions-per-iter.",
                n, len(pool),
            )

        if alpha <= 0.0:
            return random.sample(pool, actual_n)

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
            len(pool), size=actual_n, replace=False, p=probs
        )
        return [pool[i] for i in chosen]

    def _compute_math_ratio(iteration: int) -> float:
        """
        Return the effective MATH fraction for this iteration, applying the
        linear ramp from ``--math-mix-ratio`` to ``--math-mix-ratio-late``
        over the window [math_ramp_start, math_ramp_start + 10].

        Before ``math_ramp_start``:   ratio = math_mix_ratio
        During ramp (10 iters):       ratio linearly increases
        After ramp:                   ratio = math_mix_ratio_late
        """
        base = args.math_mix_ratio
        late = args.math_mix_ratio_late
        if late is None or late <= base:
            return base
        ramp_start = args.math_ramp_start
        ramp_end   = ramp_start + 10
        if iteration < ramp_start:
            return base
        if iteration >= ramp_end:
            return late
        frac = (iteration - ramp_start) / 10.0
        return base + frac * (late - base)

    def _sample_mixed_batch(n: int, iteration: int) -> List[Dict[str, str]]:
        """
        Sample ``n`` questions enforcing the configured MATH/GSM8K ratio.

        Stratification:
          math_n   = round(n × math_ratio)   drawn from MATH + Big-Math pool
          gsm8k_n  = n - math_n              drawn from GSM8K pool

        Both sub-samples use difficulty-weighted sampling independently so the
        win-rate statistics stay per-source rather than blending across pools of
        very different difficulty.

        If either sub-pool is empty the other pool absorbs its allocation.
        """
        math_ratio = _compute_math_ratio(iteration)
        math_n  = int(round(n * math_ratio))
        gsm8k_n = n - math_n

        # Fall back gracefully when one pool is empty
        if not _math_pool:
            math_n, gsm8k_n = 0, n
        if not _gsm8k_pool:
            gsm8k_n, math_n = 0, n

        sampled_math  = _sample_by_difficulty(_math_pool,  math_n,  args.difficulty_alpha)
        sampled_gsm8k = _sample_by_difficulty(_gsm8k_pool, gsm8k_n, args.difficulty_alpha)

        batch = sampled_math + sampled_gsm8k
        random.shuffle(batch)
        logger.debug(
            "Mixed batch iter %d: %d MATH + %d GSM8K = %d total (ratio=%.2f)",
            iteration, len(sampled_math), len(sampled_gsm8k), len(batch), math_ratio,
        )
        return batch

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
        # Training reward is binary (1.0 if correct, 0.0 otherwise).
        # PRM / format are diagnostic only.
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

        # Sample questions — stratified by source (MATH vs GSM8K) to honour
        # --math-mix-ratio, then difficulty-weighted within each sub-pool.
        questions_batch = _sample_mixed_batch(args.questions_per_iter, iteration)
        cur_lr = optimizer.param_groups[0]["lr"]
        # Temperature annealing: linearly decay T from peak → min_temp over the run.
        # Early iterations need high T for exploration; later ones need lower T
        # to consolidate learned strategies (and close the training/eval gap).
        _anneal_frac = min(1.0, (iteration - 1) / max(1, args.num_iterations - 1))
        _annealed_temp = args.temperature * (1.0 - 0.5 * _anneal_frac)  # 0.8 → 0.4
        logger.info(
            "LR this iteration: %.2e | T=%.3f",
            cur_lr, _annealed_temp,
        )

        all_rewards:   List[float] = []
        all_q_rewards: List[float] = []
        _grounded_rewards:   List[float] = []
        _sp_rewards:         List[float] = []
        _grounded_step_accs: List[float] = []
        _grounded_lccps:     List[float] = []
        _skipped_zero_var:   int = 0   # groups skipped due to zero reward variance
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

        # Randomly assign self-play slots before pre-sorting.
        n_self_play_target = int(round(len(questions_batch) * args.self_play_ratio))
        _all_indices = list(range(len(questions_batch)))
        random.shuffle(_all_indices)
        _self_play_indices = set(_all_indices[:n_self_play_target])

        # CHANGE 2: Pre-sort batch — grounded questions first, self-play second.
        # This lets us do a clean optimizer.step() on grounded gradients before
        # the noisier self-play gradients accumulate. Two separate weight updates
        # per iteration: one anchored to gold answers, one for self-play.
        _grounded_items = [
            (idx, qa) for idx, qa in enumerate(questions_batch)
            if idx not in _self_play_indices
        ]
        _selfplay_items = [
            (idx, qa) for idx, qa in enumerate(questions_batch)
            if idx in _self_play_indices
        ]
        _ordered_batch = _grounded_items + _selfplay_items
        _n_grounded = len(_grounded_items)
        _grounded_step_done = False   # flag: intermediate step already fired
        _grounded_n_groups  = 0       # groups counted in grounded phase
        _grounded_loss_val  = 0.0     # loss accumulated in grounded phase

        # Zero gradients once before the loop — we accumulate them via
        # per-group .backward() calls instead of building one giant graph.
        # Keeping all K*N forward passes alive until a single backward()
        # at the end would hold O(K*N) computation graphs in GPU memory
        # simultaneously (64 graphs at K=4, N=16), risking OOM.  Calling
        # .backward() immediately after each group frees that graph right
        # away; gradients accumulate in .grad tensors without extra memory.
        optimizer.zero_grad()

        pbar = tqdm(_ordered_batch, desc=f"Iter {iteration} GRPO groups", unit="q")
        for _group_idx, qa in pbar:

            # ── Decide: self-play (model generates question) or grounded ─────
            # Random interleaving: self-play slots chosen before the loop.
            use_self_play = _group_idx in _self_play_indices

            # CHANGE 2: At the first self-play group, step optimizer on
            # accumulated grounded gradients before processing any self-play.
            if use_self_play and not _grounded_step_done:
                if _grounded_n_groups > 0:
                    if _grounded_n_groups > 1:
                        for _p in model.parameters():
                            if _p.grad is not None:
                                _p.grad.div_(_grounded_n_groups)
                    torch.nn.utils.clip_grad_norm_(
                        [_p for _p in model.parameters() if _p.requires_grad],
                        args.max_grad_norm,
                    )
                    optimizer.step()
                    optimizer.zero_grad()
                    _grounded_loss_val_avg = _grounded_loss_val / _grounded_n_groups
                    logger.info(
                        "Grounded phase optimizer.step() | grounded_loss=%.4f | groups=%d",
                        _grounded_loss_val_avg, _grounded_n_groups,
                    )
                _grounded_step_done = True

            if use_self_play:
                # ── SELF-PLAY BRANCH ─────────────────────────────────────────
                # 1. Sample a curriculum instruction (topic + difficulty target)
                instruction, target_topic, target_difficulty = math_env.sample_instruction()

                # 2. Model generates the question from the instruction.
                #    This is the "proposer" role in Theme #4 self-improvement:
                #    the model creates its own challenge.
                q_gen_attempts += 1

                # ── TWO-PHASE QUESTION GRPO (when --q-group-size ≥ 2) ────────
                # Phase 1: sample K_q question candidates, store their token
                #   IDs for a question-level GRPO update.
                # Phase 2: for each candidate, generate M=group_size solutions,
                #   score them, and run a solution-level GRPO update.
                # The per-question reward (mean solution reward) is then used
                # to run GRPO on the question tokens — gradients flow back
                # through the question tokens for the first time.
                if args.q_group_size > 1:
                    _q_temp = min(0.90, _annealed_temp + 0.05)
                    q_cands, q_ids_all, q_masks_all, q_olps_all = generate_questions_batched(
                        model=model,
                        tokenizer=tokenizer,
                        instruction=instruction,
                        K_q=args.q_group_size,
                        max_new_tokens=128,
                        temperature=_q_temp,
                        device=device,
                    )
                    # Keep only candidates with enough substance
                    _valid_q = [
                        (q, ids, mask, olp)
                        for q, ids, mask, olp
                        in zip(q_cands, q_ids_all, q_masks_all, q_olps_all)
                        if len(q.strip()) >= 10
                    ]
                    if not _valid_q:
                        logger.debug("Two-phase SP: all %d question candidates too short, skipping.", args.q_group_size)
                        skipped += 1
                        continue
                    q_gen_valid += 1
                    n_self_play += 1

                    # Phase 2: score solutions for each valid question candidate
                    _question_agg_rewards: List[float] = []   # one per valid candidate
                    _q_total_loss_val: float = 0.0

                    for _q_text, _q_ids, _q_mask, _q_olp in _valid_q:
                        solution_prompt = math_env.format_solution_prompt(_q_text)
                        sols_q, ids_q, masks_q, olps_q = generate_solutions_batched(
                            model=model,
                            tokenizer=tokenizer,
                            prompt=solution_prompt,
                            K=args.group_size,
                            max_new_tokens=args.max_new_tokens,
                            temperature=_annealed_temp,
                            device=device,
                        )
                        # Overlong filter
                        if args.overlong_filter:
                            _vf = [
                                t for t in zip(sols_q, ids_q, masks_q, olps_q)
                                if int(t[2].sum().item()) < args.max_new_tokens
                            ]
                            if _vf:
                                sols_q, ids_q, masks_q, olps_q = map(list, zip(*_vf))  # type: ignore
                            else:
                                skipped += 1
                                _question_agg_rewards.append(0.0)
                                continue

                        # Score solutions
                        _sol_rewards: List[float] = []
                        for _sol in sols_q:
                            _r, _q_rew, _, _q_met = compute_self_play_reward(
                                question=_q_text,
                                solution=_sol,
                                target_topic=target_topic,
                                target_difficulty=target_difficulty,
                                math_env=math_env,
                            )
                            _sol_rewards.append(_r)
                            all_q_rewards.append(_q_rew)
                            _qc_topic.append(_q_met["topic_match"])
                            _qc_diff.append(_q_met["difficulty_fit"])
                            _qc_clarity.append(_q_met["clarity"])
                            _qc_novelty.append(_q_met["novelty"])
                            _qc_solvability.append(_q_met["solvability"])

                        all_rewards.extend(_sol_rewards)
                        _sp_rewards.extend(_sol_rewards)

                        # Aggregate question reward = mean of its solution rewards
                        _q_agg = float(np.mean(_sol_rewards))
                        _question_agg_rewards.append(_q_agg)
                        if _q_agg > 0.5:
                            q_quality_good += 1

                        # ── Solution-level GRPO update ───────────────────────
                        _sol_loss = grpo_loss_for_group(
                            model=model,
                            input_ids_list=ids_q,
                            response_masks=masks_q,
                            rewards=_sol_rewards,
                            old_log_probs=olps_q,
                            clip_eps=args.clip_eps,
                            kl_coef=args.kl_coef,
                            ref_model=ref_model,
                        )
                        if _sol_loss is not None:
                            _sol_loss.backward()
                            total_loss_val += _sol_loss.item()
                            _q_total_loss_val += _sol_loss.item()
                            n_groups += 1
                        else:
                            skipped += 1
                            _skipped_zero_var += 1

                    # ── Question-level GRPO update ───────────────────────────
                    # Advantages are computed over the K_q question-reward
                    # scalars.  The IS ratio is exp(new_lp_question - old_lp_question).
                    # kl_coef=0 here: there is no reference distribution for questions.
                    _q_ids_v   = [t[1] for t in _valid_q]
                    _q_masks_v = [t[2] for t in _valid_q]
                    _q_olps_v  = [t[3] for t in _valid_q]

                    _q_loss = grpo_loss_for_group(
                        model=model,
                        input_ids_list=_q_ids_v,
                        response_masks=_q_masks_v,
                        rewards=_question_agg_rewards,
                        old_log_probs=_q_olps_v,
                        clip_eps=args.clip_eps,
                        kl_coef=0.0,   # no ref model for question tokens
                        ref_model=None,
                    )
                    if _q_loss is not None:
                        _q_loss.backward()
                        logger.debug(
                            "Q-GRPO: loss=%.4f q_rewards=%s (variance=%.4f)",
                            _q_loss.item(),
                            [f"{r:.3f}" for r in _question_agg_rewards],
                            float(np.var(_question_agg_rewards)),
                        )

                    # pbar update then skip to next group (all done above)
                    _mean_r_sp = float(np.mean(all_rewards[-len(_valid_q)*args.group_size:])) if all_rewards else 0.0
                    _q_acc_pct = 100.0 * q_quality_good / max(1, n_self_play)
                    pbar.set_postfix(
                        loss=f"{_q_total_loss_val / max(1, len(_valid_q)):.4f}",
                        mean_r=f"{_mean_r_sp:.3f}",
                        q_acc=f"{_q_acc_pct:.0f}%",
                        q_rew=f"{float(np.mean(all_q_rewards)):.3f}" if all_q_rewards else "n/a",
                        skip=skipped,
                    )
                    continue  # ← everything handled above; jump to next group

                # ── K_q=1: original single-question path (no question GRPO) ──
                question = generate_question(
                    model=model,
                    tokenizer=tokenizer,
                    instruction=instruction,
                    max_new_tokens=128,   # questions are short
                    device=device,
                    # Slightly warmer than solution temperature for diversity,
                    # but anneals with the same schedule to stay consistent.
                    temperature=min(0.90, _annealed_temp + 0.05),
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
                    temperature=_annealed_temp,
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
                    # CHANGE 3: Use solution-quality-only reward for the GRPO update.
                    # The question reward is now adversarial (computed post-loop)
                    # rather than a heuristic quality score.
                    # compute_self_play_reward returns (combined, q_rew, sol_rew, q_met)
                    _, q_rew, r, q_met = compute_self_play_reward(
                        question=question,
                        solution=sol,
                        target_topic=target_topic,
                        target_difficulty=target_difficulty,
                        math_env=math_env,
                    )
                    # r is now the solution-only reward (PRM-based, no q blending)
                    _sp_q_rew_this_group.append(r)   # track sol scores for win-rate
                    all_q_rewards.append(q_rew)      # keep q_rew for logging
                    # Collect per-component breakdown
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
                    _grounded_lccps.append(r_dict["lccp"])
                rewards.append(r)
            all_rewards.extend(rewards)
            # Route to path-specific accumulators for separate batch_acc reporting
            if use_self_play:
                _sp_rewards.extend(rewards)
            else:
                _grounded_rewards.extend(rewards)

            # CHANGE 3: Adversarial win-rate question reward.
            # win_rate = fraction of K solutions with solution reward > 0.5
            # Generator earns reward for questions the Solver finds HARD (low win_rate).
            # Groups where win_rate==0.0 or 1.0 carry zero gradient variance — skip them.
            if use_self_play and _sp_q_rew_this_group:
                _win_rate = float(np.mean([r > 0.5 for r in _sp_q_rew_this_group]))
                if _win_rate == 0.0 or _win_rate == 1.0:
                    # No variance in rewards — no gradient signal, don't update q stats
                    logger.debug(
                        "Self-play group skipped for q_reward: win_rate=%.2f (zero variance)",
                        _win_rate,
                    )
                else:
                    # Adversarial: harder questions (low win_rate) → higher q_reward
                    _q_adv_reward = 1.0 - _win_rate
                    all_q_rewards.append(_q_adv_reward)
                    if _q_adv_reward > 0.5:   # win_rate < 0.5 → genuinely challenging
                        q_quality_good += 1
                    logger.debug(
                        "Self-play adversarial q_reward: win_rate=%.2f → q_reward=%.2f",
                        _win_rate, _q_adv_reward,
                    )

            # --- Update difficulty stats (grounded questions only — self-play
            #     questions are ephemeral and have no stable key) ---
            if not use_self_play:
                _key = _question_key(question)
                _topic = qa.get("topic", "other")
                _q_attempts[_key] += len(solutions)
                _topic_attempts[_topic] += len(solutions)
                # Win = reward in the top half of THIS group, not an absolute 0.5 threshold.
                # Using a relative threshold avoids the case where all solutions score 0.55
                # (all "wins" → easy) or all score 0.45 (all "losses" → impossible) when the
                # rewards are actually similar and carry no difficulty information.
                _group_median = float(np.median(rewards))
                _wins = sum(1 for r in rewards if r > _group_median)
                _q_wins[_key] += _wins
                _topic_wins[_topic] += _wins

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
                _skipped_zero_var += 1
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
            # CHANGE 2: Track per-phase group counts for per-phase normalisation.
            if not _grounded_step_done:
                _grounded_n_groups += 1
                _grounded_loss_val += group_loss.item()
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
        # CHANGE 2: If grounded step already fired (intermediate step done),
        # only self-play gradients remain in .grad — normalise by their count.
        # If no self-play groups exist, or self-play ratio=0, fall through as before.
        _selfplay_n_groups = n_groups - _grounded_n_groups
        _norm_groups = _selfplay_n_groups if _grounded_step_done else n_groups
        if _norm_groups > 0 or (not _grounded_step_done and n_groups > 0):
            _div = _norm_groups if _grounded_step_done else n_groups
            if _div > 1:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.div_(_div)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                args.max_grad_norm,
            )
            optimizer.step()
            loss_val = total_loss_val / max(1, n_groups)
        else:
            loss_val = 0.0
        scheduler.step()   # advance LR once per iteration (not per phase)

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
        mean_lccp = (
            float(np.mean(_grounded_lccps))
            if _grounded_lccps else 0.0
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
            "grounded_acc=%.1f%% | step_acc=%.1f%% | lccp=%.1f%% | batch_acc=%.1f%% | "
            "groups=%d skipped=%d(0var=%d) | lr=%.2e | %.1fs",
            iteration, loss_val, mean_r, std_r,
            100 * grounded_acc_r,
            100 * mean_step_acc,
            100 * mean_lccp,
            100 * acc_r,
            n_groups, skipped, _skipped_zero_var, _cur_lr, iter_time,
        )

        # Print topic win rates
        _topic_summary = {}
        for t, attempts in _topic_attempts.items():
            if attempts > 0:
                _topic_summary[t] = round(_topic_wins[t] / attempts, 3)
        if _topic_summary:
            logger.info("Topic win rates: %s", _topic_summary)
        # Starvation warning: if >30% of groups were skipped due to zero reward
        # variance (all K solutions same score), the curriculum difficulty is
        # mis-calibrated — either too easy (all correct) or too hard (all wrong).
        _total_attempted = n_groups + skipped
        if _total_attempted > 0 and _skipped_zero_var / _total_attempted > 0.30:
            logger.warning(
                "STARVATION: %.0f%% of groups skipped (zero variance). "
                "grounded_acc=%.1f%% suggests curriculum is %s. "
                "Consider adjusting --difficulty-alpha.",
                100 * _skipped_zero_var / _total_attempted,
                100 * grounded_acc_r,
                "too easy (raise alpha)" if grounded_acc_r > 0.75 else "too hard (lower alpha)",
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
            "lccp":                  mean_lccp,
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
            # CHANGE 4: Use final_answer_accuracy as the primary checkpoint metric.
            # combined_score blends PRM + format, so the model could get "better"
            # at formatting while getting worse at actual math. We now save
            # best_policy only when real math accuracy (exact-match %) improves.
            cur_combined = float(eval_res.get("combined_score", best_combined))
            cur_prm_mean = float(eval_res.get("prm_mean",       best_prm_mean))
            cur_fa_acc   = float(eval_res.get("final_answer_accuracy", 0.0))

            _log_eval_result(f"iter {iteration}", eval_res, best=best_combined)

            # ── Checkpoint: save when final-answer accuracy strictly improves ──
            if cur_fa_acc > best_accuracy + 1e-4:
                reason = f"final_answer_accuracy {cur_fa_acc:.4f} > {best_accuracy:.4f}"
                best_accuracy  = cur_fa_acc
                best_combined  = max(best_combined, cur_combined)
                best_prm_mean  = max(best_prm_mean, cur_prm_mean)
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
        "Best eval combined_score      : %.4f  "
        "(checkpoint metric: final_answer_accuracy — binary correct/wrong)",
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
