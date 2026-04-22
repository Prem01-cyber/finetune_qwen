"""
Phase-1 benchmark: compare baseline vs vLLM batched rollout collection.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_ppo_training_curriculum import load_reference_questions
from src.rl.math_environment_curriculum import CurriculumMathEnvironment
from src.rl.value_network import ValueHead


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def initialize_models(base_model: str):
    model_path = Path(base_model)
    is_adapter = (model_path / "adapter_config.json").exists()

    if is_adapter:
        meta_file = model_path / "pipeline_meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            base_model_name = meta.get("base_model", "Qwen/Qwen2.5-Math-1.5B-Instruct")
        else:
            base_model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_lm = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        policy = PeftModel.from_pretrained(base_lm, base_model).merge_and_unload()
        value = ValueHead(base_model_name).to(policy.device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        policy = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        value = ValueHead(base_model).to(policy.device)
    return policy, value, tokenizer


def build_env(
    base_model: str,
    reference_questions: List[str],
    use_vllm: bool,
    rollout_batch_size: int,
    max_question_tokens: int,
    max_solution_tokens: int,
) -> CurriculumMathEnvironment:
    policy, value, tokenizer = initialize_models(base_model)
    return CurriculumMathEnvironment(
        policy_model=policy,
        value_model=value,
        tokenizer=tokenizer,
        reference_questions=reference_questions,
        curriculum_checkpoint_dir="checkpoints/ppo_training_curriculum/benchmark_curriculum",
        max_question_tokens=max_question_tokens,
        max_solution_tokens=max_solution_tokens,
        temperature=0.7,
        top_p=0.9,
        consensus_temperature=0.5,
        use_vllm=use_vllm,
        vllm_batch_size=max(1, int(rollout_batch_size)),
        vllm_tensor_parallel_size=1,
    )


def benchmark_once(
    env: CurriculumMathEnvironment,
    rollouts: int,
    batched: bool,
    rollout_batch_size: int,
) -> float:
    start = time.perf_counter()
    if batched:
        env.collect_rollouts_batched(
            num_trajectories=rollouts,
            batch_size=rollout_batch_size,
            verbose=False,
        )
    else:
        env.collect_rollouts(
            num_trajectories=rollouts,
            verbose=False,
        )
    return time.perf_counter() - start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark phase-1 rollout speed improvements")
    parser.add_argument("--base-model", type=str, default="checkpoints/dual_task_v1")
    parser.add_argument("--reference-data", type=str, default="data/sft/gsm8k_sft.jsonl")
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rollout-batch-size", type=int, default=4)
    parser.add_argument("--max-question-tokens", type=int, default=120)
    parser.add_argument("--max-solution-tokens", type=int, default=192)
    parser.add_argument("--output-json", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repeats = max(1, int(args.repeats))
    reference_questions = load_reference_questions(args.reference_data)

    baseline_env = build_env(
        base_model=args.base_model,
        reference_questions=reference_questions,
        use_vllm=False,
        rollout_batch_size=args.rollout_batch_size,
        max_question_tokens=args.max_question_tokens,
        max_solution_tokens=args.max_solution_tokens,
    )
    vllm_env = build_env(
        base_model=args.base_model,
        reference_questions=reference_questions,
        use_vllm=True,
        rollout_batch_size=args.rollout_batch_size,
        max_question_tokens=args.max_question_tokens,
        max_solution_tokens=args.max_solution_tokens,
    )

    baseline_times = [
        benchmark_once(
            env=baseline_env,
            rollouts=args.rollouts,
            batched=False,
            rollout_batch_size=args.rollout_batch_size,
        )
        for _ in range(repeats)
    ]
    vllm_times = [
        benchmark_once(
            env=vllm_env,
            rollouts=args.rollouts,
            batched=True,
            rollout_batch_size=args.rollout_batch_size,
        )
        for _ in range(repeats)
    ]

    baseline_mean = statistics.mean(baseline_times)
    vllm_mean = statistics.mean(vllm_times)
    speedup = baseline_mean / max(vllm_mean, 1e-9)

    report = {
        "rollouts": int(args.rollouts),
        "repeats": repeats,
        "baseline_seconds": baseline_times,
        "vllm_seconds": vllm_times,
        "baseline_mean_seconds": baseline_mean,
        "vllm_mean_seconds": vllm_mean,
        "speedup_x": speedup,
    }
    logger.info("Phase-1 benchmark report: %s", report)
    print(json.dumps(report, indent=2))

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

