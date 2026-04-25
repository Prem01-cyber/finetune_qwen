"""
Dual-task SFT pipeline: train model on both question generation and solution tasks.

This pipeline trains a single model that can:
1. Generate math questions when prompted with "### Task: Generate Question"
2. Solve math problems when prompted with "### Task: Solve Problem"

Examples
--------
  # Train dual-task model
  python scripts/dual_task_sft_pipeline.py train \\
      --data data/sft/dual_task_train.jsonl \\
      --output-dir checkpoints/dual_task_v1 \\
      --epochs 2

  # Infer - Question Generation
  python scripts/dual_task_sft_pipeline.py infer \\
      --adapter checkpoints/dual_task_v1 \\
      --task generate \\
      --prompt "Create a word problem about fractions and money requiring 3 steps."

  # Infer - Solution Generation
  python scripts/dual_task_sft_pipeline.py infer \\
      --adapter checkpoints/dual_task_v1 \\
      --task solve \\
      --problem "Janet has 16 eggs. She eats 3. How many are left?"

Dependencies: torch, transformers, peft, datasets, accelerate, bitsandbytes, trl
"""

from __future__ import annotations

import os

if "HF_HUB_DISABLE_XET" not in os.environ:
    os.environ["HF_HUB_DISABLE_XET"] = "1"

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config.prompts import (
    SOLVE_TASK_PREFIX,
    GENERATE_TASK_PREFIX,
    SOLVER_SYSTEM_PROMPT,
    GENERATOR_SYSTEM_PROMPT,
)


def _warmup_steps_from_ratio(
    num_examples: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: float,
    warmup_ratio: float,
) -> int:
    """Calculate warmup steps from ratio."""
    if warmup_ratio <= 0:
        return 0
    num_batches = max(
        1,
        (num_examples + per_device_train_batch_size - 1) // per_device_train_batch_size,
    )
    num_update_steps_per_epoch = max(1, num_batches // gradient_accumulation_steps)
    total_optimizer_steps = max(1, math.ceil(num_train_epochs * num_update_steps_per_epoch))
    return min(total_optimizer_steps, int(total_optimizer_steps * warmup_ratio))


def cmd_train(args: argparse.Namespace) -> None:
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        raise SystemExit(
            "Missing dependency for training. Install:\n"
            "  pip install torch transformers peft datasets accelerate bitsandbytes trl\n"
            f"Original error: {e}"
        ) from e

    data_path = Path(args.data)
    if not data_path.is_file():
        raise SystemExit(f"Data file not found: {data_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    compute_dtype = getattr(torch, args.bnb_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading model {args.model} …")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=compute_dtype,
    )
    model = prepare_model_for_kbit_training(model)
    
    peft = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(args.target_modules.split(",")),
    )
    model = get_peft_model(model, peft)
    model.config.use_cache = False
    model.print_trainable_parameters()

    print(f"Loading dual-task dataset from {data_path} …")
    ds = load_dataset("json", data_files=str(data_path), split="train")
    if args.max_samples and args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    
    task_counts = {"solve": 0, "generate": 0, "unknown": 0}
    for example in ds:
        task_type = example.get("task_type", "unknown")
        task_counts[task_type] = task_counts.get(task_type, 0) + 1
    
    print(f"Dataset composition:")
    print(f"  Total examples: {len(ds)}")
    print(f"  Solve tasks: {task_counts['solve']} ({task_counts['solve']/len(ds):.1%})")
    print(f"  Generate tasks: {task_counts['generate']} ({task_counts['generate']/len(ds):.1%})")
    if task_counts['unknown'] > 0:
        print(f"  Unknown tasks: {task_counts['unknown']}")

    def formatting_func(example):
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

    if args.warmup_steps is not None:
        warmup_steps = max(0, args.warmup_steps)
    else:
        warmup_steps = _warmup_steps_from_ratio(
            len(ds),
            args.batch_size,
            args.grad_accum,
            args.epochs,
            args.warmup_ratio,
        )

    sft_args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=args.bf16 and torch.cuda.is_available(),
        fp16=args.fp16 and torch.cuda.is_available() and not args.bf16,
        max_length=args.max_seq_length,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        report_to="none",
        gradient_checkpointing=True,
    )

    print("\nStarting dual-task training...")
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=ds,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )

    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    with (out_dir / "pipeline_meta.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "pipeline_type": "dual_task",
                "base_model": args.model,
                "data": str(data_path),
                "lora_rank": args.lora_rank,
                "epochs": args.epochs,
                "task_distribution": task_counts,
            },
            f,
            indent=2,
        )
    print(f"\nSaved dual-task adapter and tokenizer to {out_dir}")


def cmd_infer(args: argparse.Namespace) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    adapter = Path(args.adapter)
    meta_path = adapter / "pipeline_meta.json"
    base_model = args.base_model
    
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        base_model = meta.get("base_model", base_model)
        pipeline_type = meta.get("pipeline_type", "unknown")
        if pipeline_type != "dual_task":
            print(f"Warning: Adapter trained with pipeline_type='{pipeline_type}', expected 'dual_task'")

    compute_dtype = getattr(torch, args.bnb_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base {base_model} + adapter {adapter} …")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter))
    model.eval()

    if args.task == "solve":
        system_prompt = SOLVER_SYSTEM_PROMPT
        user_content = (
            f"{SOLVE_TASK_PREFIX}"
            "Solve the following problem. Show your reasoning as numbered steps, "
            "then give the final numeric answer on the last line.\n\n"
            f"Problem:\n{args.problem.strip()}"
        )
    elif args.task == "generate":
        system_prompt = GENERATOR_SYSTEM_PROMPT
        user_content = f"{GENERATE_TASK_PREFIX}{args.prompt.strip()}"
    else:
        raise ValueError(f"Unknown task: {args.task}. Must be 'solve' or 'generate'")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print(f"\nTask: {args.task}")
    print(f"Prompt length: {inputs['input_ids'].shape[1]} tokens")
    print("\nGenerating...")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=not args.greedy,
            pad_token_id=tokenizer.pad_token_id,
        )

    gen_ids = out[0, inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    
    print("\n" + "=" * 60)
    print("Generated Output")
    print("=" * 60)
    print(text)
    print("=" * 60)

    if args.task == "solve":
        print("\n--- Format Validation ---")
        from src.sft.solution_format import validate_sympy_solution_format
        r = validate_sympy_solution_format(text)
        print(json.dumps(r.__dict__, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dual-task SFT pipeline (train / infer)")
    sub = p.add_subparsers(dest="command", required=True)

    tr = sub.add_parser("train", help="Train dual-task model on mixed dataset")
    tr.add_argument("--data", type=str, required=True, help="Dual-task training JSONL")
    tr.add_argument("--output-dir", type=str, required=True, help="Output directory for adapter")
    tr.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct", help="Base model")
    tr.add_argument("--epochs", type=float, default=2.0, help="Training epochs (default: 2.0 for dual-task)")
    tr.add_argument("--batch-size", type=int, default=1)
    tr.add_argument("--grad-accum", type=int, default=8)
    tr.add_argument("--learning-rate", type=float, default=2e-4)
    tr.add_argument("--max-samples", type=int, default=0, help="0 = use full dataset")
    tr.add_argument("--lora-rank", type=int, default=16)
    tr.add_argument("--lora-alpha", type=int, default=32)
    tr.add_argument("--lora-dropout", type=float, default=0.05)
    tr.add_argument(
        "--target-modules",
        type=str,
        default="q_proj,v_proj,o_proj,gate_proj",
    )
    tr.add_argument("--max-seq-length", type=int, default=2048)
    tr.add_argument("--save-steps", type=int, default=200)
    tr.add_argument("--logging-steps", type=int, default=10)
    tr.add_argument("--warmup-ratio", type=float, default=0.03)
    tr.add_argument("--warmup-steps", type=int, default=None)
    tr.add_argument("--bf16", action="store_true", default=True)
    tr.add_argument("--no-bf16", dest="bf16", action="store_false")
    tr.add_argument("--fp16", action="store_true")
    tr.add_argument("--bnb-compute-dtype", type=str, default="bfloat16")
    tr.set_defaults(func=cmd_train)

    inf = sub.add_parser("infer", help="Generate with dual-task model")
    inf.add_argument("--adapter", type=str, required=True, help="Adapter directory")
    inf.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B-Instruct",
        help="Base model (auto-detected from pipeline_meta.json if present)",
    )
    inf.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["solve", "generate"],
        help="Task type: 'solve' for problem solving, 'generate' for question generation",
    )
    inf.add_argument(
        "--problem",
        type=str,
        default="",
        help="Math problem to solve (required if --task solve)",
    )
    inf.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Question generation prompt (required if --task generate)",
    )
    inf.add_argument("--max-new-tokens", type=int, default=1024)
    inf.add_argument("--temperature", type=float, default=0.7)
    inf.add_argument("--top-p", type=float, default=0.95)
    inf.add_argument("--greedy", action="store_true", help="Use greedy decoding")
    inf.add_argument("--bnb-compute-dtype", type=str, default="bfloat16")
    inf.set_defaults(func=cmd_infer)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    
    if args.command == "infer":
        if args.task == "solve" and not args.problem:
            raise SystemExit("Error: --problem is required when --task solve")
        if args.task == "generate" and not args.prompt:
            raise SystemExit("Error: --prompt is required when --task generate")
    
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    
    args.func(args)


if __name__ == "__main__":
    main()
