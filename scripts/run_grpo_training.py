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
import csv
import json
import logging
import random
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
# Reward
# ---------------------------------------------------------------------------

def compute_grounded_reward(
    question: str,
    solution: str,
    gold_final: str,
    math_env: CurriculumMathEnvironment,
) -> float:
    """Thin wrapper re-using the existing reward pipeline.
    PRM is handled inside math_env (loaded at startup); no need to pass it here.
    """
    result = math_env.compute_grounded_reward(
        question=question,
        solution=solution,
        gold_final=gold_final,
    )
    return float(result["combined_score"])


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


def generate_solutions(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    K: int,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> Tuple[List[str], List[torch.Tensor], List[torch.Tensor]]:
    """
    Generate K solutions for a pre-formatted prompt.

    ``prompt`` must come from ``math_env.format_solution_prompt(question)``
    so the chat-template system/user wrapping exactly matches the SFT
    training format.  Passing a raw-text prompt causes the model to ignore
    step-format instructions, produce empty Final-Answer lines, and
    generate repetitive filler tokens until max_new_tokens is exhausted.

    Returns:
        solutions     : list of K decoded solution strings (no prompt, no
                        special tokens — ready for compute_grounded_reward)
        input_ids_list: list of K full (prompt+response) token ID tensors
        response_masks: list of K bool masks (True = response token)
    """
    stop_ids = _build_stop_token_ids(tokenizer)

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=1024,   # generous for chat-template-wrapped prompts
    ).to(device)
    prompt_len = enc["input_ids"].shape[1]

    solutions: List[str] = []
    input_ids_list: List[torch.Tensor] = []
    response_masks: List[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for _ in range(K):
            out = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=stop_ids,
                use_cache=True,
            )
            full_ids = out[0]  # [prompt_len + response_len]
            response_ids = full_ids[prompt_len:]
            solution = tokenizer.decode(response_ids, skip_special_tokens=True)

            # Build mask: 0 for prompt tokens, 1 for response tokens
            mask = torch.zeros(full_ids.shape[0], dtype=torch.bool, device=device)
            mask[prompt_len:] = True

            solutions.append(solution)
            input_ids_list.append(full_ids)
            response_masks.append(mask)

    return solutions, input_ids_list, response_masks


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
    eps: float = 1e-8,
) -> Optional[torch.Tensor]:
    """
    Compute GRPO loss for a group of K solutions to the same question.

    Skips groups where all rewards are identical (zero gradient signal).
    Returns None if skipped.
    """
    rewards_arr = np.array(rewards, dtype=np.float32)
    std_r = rewards_arr.std()
    if std_r < eps:
        # All solutions got the same reward — skip (no learning signal)
        return None

    mean_r = rewards_arr.mean()
    advantages = (rewards_arr - mean_r) / (std_r + eps)
    # Clip to prevent extreme advantages from destabilising early training
    advantages = np.clip(advantages, -5.0, 5.0)

    group_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    n_valid = 0

    model.train()
    for ids, mask, adv in zip(input_ids_list, response_masks, advantages):
        log_prob_sum = compute_sequence_log_prob(model, ids, mask)
        n_response = mask[1:].sum().item()
        if n_response == 0:
            continue
        # Normalise by response length (mean log-prob, not sum)
        mean_log_prob = log_prob_sum / n_response
        group_loss = group_loss - adv * mean_log_prob
        n_valid += 1

    if n_valid == 0:
        return None
    return group_loss / n_valid


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_policy(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    eval_data_path: str,
    max_samples: int,
    max_new_tokens: int,
) -> Dict[str, object]:
    if not Path(eval_data_path).exists():
        return {"accuracy": 0.0, "correct": 0, "total": 0}
    model.eval()
    results = evaluate_gsm8k(
        model=model,
        tokenizer=tokenizer,
        data_path=eval_data_path,
        max_samples=max_samples,
        max_new_tokens=max_new_tokens,
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
    parser.add_argument("--use-prm", dest="use_prm", action="store_true", default=True)
    parser.add_argument("--no-prm", dest="use_prm", action="store_false")
    parser.add_argument("--prm-model", default="Qwen/Qwen2.5-Math-PRM-7B")
    parser.add_argument("--skip-initial-eval", action="store_true")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
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
        "Run config: K=%d N=%d lr=%.1e T=%.2f max_new=%d | eval_every=%d eval_N=%d | "
        "grad_clip=%.2f save_every=%d keep_last=%d",
        args.group_size, args.questions_per_iter, args.learning_rate,
        args.temperature, args.max_new_tokens,
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

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        fused=torch.cuda.is_available(),
    )

    # ── Load data ────────────────────────────────────────────────────────────
    qa_pairs = load_gsm8k(args.gsm8k_data)
    if not qa_pairs:
        logger.error("No GSM8K data found — cannot train. Exiting.")
        sys.exit(1)

    # ── Load PRM (optional) ─────────────────────────────────────────────────
    prm_scorer: Optional[ProcessRewardScorer] = None
    if args.use_prm:
        try:
            prm_scorer = ProcessRewardScorer(
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
        prm_scorer=prm_scorer,
        max_solution_tokens=args.max_new_tokens,
        device=device,
    )

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
        )
        logger.info("GSM8K Accuracy: %.2f%% (%d/%d)",
                    100 * initial_eval["accuracy"],
                    initial_eval["correct"], initial_eval["total"])
        metrics_log.append({"iteration": 0, **initial_eval})
        best_accuracy = float(initial_eval.get("accuracy", 0.0))
    else:
        best_accuracy = 0.0

    # ── Training ─────────────────────────────────────────────────────────────
    for iteration in range(1, args.num_iterations + 1):
        iter_start = time.perf_counter()
        logger.info("=" * 70)
        logger.info("GRPO ITERATION %d/%d", iteration, args.num_iterations)
        logger.info("=" * 70)

        # Sample questions for this iteration
        questions_batch = random.sample(
            qa_pairs, min(args.questions_per_iter, len(qa_pairs))
        )

        all_rewards: List[float] = []
        skipped = 0
        n_groups = 0
        total_loss_val = 0.0

        # Zero gradients once before the loop — we accumulate them via
        # per-group .backward() calls instead of building one giant graph.
        # Keeping all K*N forward passes alive until a single backward()
        # at the end would hold O(K*N) computation graphs in GPU memory
        # simultaneously (64 graphs at K=4, N=16), risking OOM.  Calling
        # .backward() immediately after each group frees that graph right
        # away; gradients accumulate in .grad tensors without extra memory.
        optimizer.zero_grad()

        pbar = tqdm(questions_batch, desc=f"Iter {iteration} GRPO groups", unit="q")
        for qa in pbar:
            question = qa["question"]
            gold = qa["gold_final"]

            # --- Generate K solutions ---
            # Use the exact same chat-template prompt format as PPO rollouts
            # (system prompt + user turn → add_generation_prompt=True).
            # A raw-text prompt causes empty Final-Answer lines and repetitive
            # filler tokens because the SFT-trained model expects the chat format.
            solution_prompt = math_env.format_solution_prompt(question)
            solutions, input_ids_list, response_masks = generate_solutions(
                model=model,
                tokenizer=tokenizer,
                prompt=solution_prompt,
                K=args.group_size,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                device=device,
            )

            # --- Score each solution ---
            rewards = []
            for sol in solutions:
                r = compute_grounded_reward(
                    question=question,
                    solution=sol,
                    gold_final=gold,
                    math_env=math_env,
                )
                rewards.append(r)
            all_rewards.extend(rewards)

            # --- GRPO loss + immediate backward ---
            group_loss = grpo_loss_for_group(
                model=model,
                input_ids_list=input_ids_list,
                response_masks=response_masks,
                rewards=rewards,
            )

            if group_loss is None:
                skipped += 1
                pbar.set_postfix(
                    mean_r=f"{np.mean(rewards):.3f}",
                    skip=skipped,
                    loss="skip",
                )
                continue

            # Backprop immediately — frees this group's computation graph.
            # Gradients from all valid groups accumulate in param.grad.
            group_loss.backward()
            total_loss_val += group_loss.item()
            n_groups += 1
            pbar.set_postfix(
                mean_r=f"{np.mean(rewards):.3f}",
                loss=f"{group_loss.item():.4f}",
                skip=skipped,
            )

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

        iter_time = time.perf_counter() - iter_start
        mean_r = float(np.mean(all_rewards)) if all_rewards else 0.0
        std_r  = float(np.std(all_rewards)) if all_rewards else 0.0
        acc_r  = float(np.mean([r > 0.5 for r in all_rewards])) if all_rewards else 0.0

        logger.info(
            "Iter %d | loss=%.4f | reward mean=%.3f std=%.3f | "
            "batch_acc=%.1f%% | groups=%d skipped=%d | %.1fs",
            iteration, loss_val, mean_r, std_r,
            100 * acc_r, n_groups, skipped, iter_time,
        )

        iter_metrics: Dict = {
            "iteration": iteration,
            "loss": loss_val,
            "mean_reward": mean_r,
            "std_reward": std_r,
            "batch_accuracy": acc_r,
            "n_groups": n_groups,
            "skipped_groups": skipped,
            "iter_time_s": iter_time,
        }

        # --- Eval ---
        if iteration % args.eval_every == 0:
            logger.info("Evaluating GSM8K (%d samples)...", args.eval_max_samples)
            eval_res = evaluate_policy(
                model, tokenizer,
                args.eval_data_path, args.eval_max_samples, args.eval_max_new_tokens,
            )
            gsm8k_acc = float(eval_res.get("accuracy", 0.0))
            logger.info(
                "GSM8K Accuracy: %.2f%% (%d/%d) | best=%.2f%%",
                100 * gsm8k_acc,
                eval_res["correct"], eval_res["total"],
                100 * best_accuracy,
            )
            if gsm8k_acc > best_accuracy:
                best_accuracy = gsm8k_acc
                best_path = out_dir / "best_policy"
                model.save_pretrained(str(best_path))
                tokenizer.save_pretrained(str(best_path))
                logger.info("New best saved to %s", best_path)
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
            "iter_time_s":    iter_metrics.get("iter_time_s", 0.0),
            # eval columns are empty ("") on non-eval iterations so the CSV
            # stays columnar without shifting anything.
            "gsm8k_accuracy": iter_metrics.get("accuracy", ""),
            "gsm8k_correct":  iter_metrics.get("correct", ""),
            "gsm8k_total":    iter_metrics.get("total", ""),
        })

    logger.info("=" * 70)
    logger.info("GRPO training complete.")
    logger.info("Best GSM8K accuracy : %.2f%%", 100 * best_accuracy)
    logger.info("Checkpoints         : %s", out_dir)
    logger.info("Logs                : %s", log_dir)
    logger.info("Console log         : %s", console_log_path)
    logger.info("=" * 70)

    # ── Final summary ─────────────────────────────────────────────────────────
    summary: Dict[str, Any] = {
        "run_name": run_name,
        "best_accuracy": best_accuracy,
        "total_iterations": args.num_iterations,
        "checkpoints_dir": str(out_dir),
        "log_dir": str(log_dir),
        "console_log": str(console_log_path),
        "metrics_csv": str(_metrics_csv_path),
        "metrics_jsonl": str(out_dir / "metrics.jsonl"),
    }
    (log_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    logger.info("Summary written to %s", log_dir / "summary.json")

    # Explicit teardown (atexit is the safety net for crashes; calling here
    # ensures everything is flushed and closed before the process returns
    # normally — atexit won't double-close because _teardown_logging is
    # idempotent via the .closed checks).
    _teardown_logging()


if __name__ == "__main__":
    main()
