#!/usr/bin/env python3
"""
End-to-end GSM8K pipeline: prepare JSONL → QLoRA SFT → save adapter → inference.

The trained model follows ``Step N:`` / ``Final Answer:`` formatting with SymPy-friendly
expressions (see ``src.agent.math_agent.SOLVER_SYSTEM_PROMPT``).

Examples
--------
  # 1) Only build training JSONL from Hugging Face GSM8K
  python scripts/gsm8k_sft_pipeline.py prepare --output data/sft/gsm8k_sft.jsonl

  # 2) Fine-tune (requires GPU recommended)
  python scripts/gsm8k_sft_pipeline.py train \\
      --data data/sft/gsm8k_sft.jsonl \\
      --output-dir checkpoints/gsm8k_sft

  # 3) Run inference with saved adapter
  python scripts/gsm8k_sft_pipeline.py infer \\
      --adapter checkpoints/gsm8k_sft \\
      --problem \"Janet has 16 eggs. She eats 3. How many are left?\"

  # Full chain
  python scripts/gsm8k_sft_pipeline.py all --output-dir checkpoints/gsm8k_sft

Dependencies: torch, transformers, peft, datasets, accelerate, bitsandbytes, trl, sympy

Tip: if downloads fail with XET / "Background writer channel closed", export ``HF_HUB_DISABLE_XET=1``
before running (this script sets it by default unless already set).
"""

from __future__ import annotations

import os

# hf-xet can error or segfault on interrupted/large shards; classic HTTP download is more robust.
if "HF_HUB_DISABLE_XET" not in os.environ:
    os.environ["HF_HUB_DISABLE_XET"] = "1"

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

# Project root (…/Maths_LLM)
ROOT = Path(__file__).resolve().parents[1]


def cmd_prepare(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "convert_gsm8k_to_sft.py"),
        "--output",
        str(Path(args.output)),
        "--splits",
        *args.splits,
    ]
    if args.source == "jsonl":
        cmd.extend(["--source", "jsonl", "--input", str(args.input)])
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))
    if args.strip_scratchpads:
        _rewrite_jsonl_strip_scratchpads(Path(args.output))


def _rewrite_jsonl_strip_scratchpads(jsonl_path: Path) -> None:
    from src.sft.solution_format import strip_gsm8k_scratchpads

    tmp = jsonl_path.with_suffix(".jsonl.tmp")
    n = 0
    with jsonl_path.open(encoding="utf-8") as fin, tmp.open("w", encoding="utf-8") as fout:
        for line in fin:
            o = json.loads(line)
            for m in o.get("messages", []):
                if m.get("role") == "assistant":
                    m["content"] = strip_gsm8k_scratchpads(m["content"])
            if "text" in o:
                sys_p = next(x["content"] for x in o["messages"] if x["role"] == "system")
                usr = next(x["content"] for x in o["messages"] if x["role"] == "user")
                asst = next(x["content"] for x in o["messages"] if x["role"] == "assistant")
                o["text"] = (
                    f"<|system|>\n{sys_p}\n<|user|>\n{usr}\n<|assistant|>\n{asst}"
                )
            fout.write(json.dumps(o, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(jsonl_path)
    print(f"Stripped <<>> scratchpads in {n} records → {jsonl_path}")


def _warmup_steps_from_ratio(
    num_examples: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    num_train_epochs: float,
    warmup_ratio: float,
) -> int:
    """Approximate HF Trainer optimizer steps; used to map legacy warmup_ratio → warmup_steps."""
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
            "  pip install torch transformers peft datasets accelerate bitsandbytes trl sympy\n"
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

    ds = load_dataset("json", data_files=str(data_path), split="train")
    if args.max_samples and args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

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
                "base_model": args.model,
                "data": str(data_path),
                "lora_rank": args.lora_rank,
                "epochs": args.epochs,
            },
            f,
            indent=2,
        )
    print(f"Saved adapter and tokenizer to {out_dir}")


def cmd_infer(args: argparse.Namespace) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from src.agent.math_agent import SOLVER_SYSTEM_PROMPT

    adapter = Path(args.adapter)
    meta_path = adapter / "pipeline_meta.json"
    base_model = args.base_model
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        base_model = meta.get("base_model", base_model)

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

    user_content = (
        "Solve the following problem. Show your reasoning as numbered steps, "
        "then give the final numeric answer on the last line.\n\n"
        f"Problem:\n{args.problem.strip()}"
    )
    messages = [
        {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

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
    print("\n--- Generated ---\n")
    print(text)
    print("\n--- Format check ---")
    from src.sft.solution_format import validate_sympy_solution_format

    r = validate_sympy_solution_format(text)
    print(json.dumps(r.__dict__, indent=2))


def cmd_all(args: argparse.Namespace) -> None:
    out_jsonl = Path(args.data) if args.data else ROOT / "data" / "sft" / "gsm8k_sft.jsonl"
    ns = argparse.Namespace(
        output=out_jsonl,
        source=args.prepare_source,
        input=args.input,
        splits=args.splits,
        strip_scratchpads=args.strip_scratchpads,
    )
    cmd_prepare(ns)
    train_ns = argparse.Namespace(
        data=str(out_jsonl),
        output_dir=args.output_dir,
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        max_samples=args.max_samples,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        max_seq_length=args.max_seq_length,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        bnb_compute_dtype=args.bnb_compute_dtype,
    )
    cmd_train(train_ns)
    if args.problem:
        infer_ns = argparse.Namespace(
            adapter=Path(args.output_dir),
            base_model=args.model,
            problem=args.problem,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            greedy=args.greedy,
            bnb_compute_dtype=args.bnb_compute_dtype,
        )
        cmd_infer(infer_ns)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GSM8K SFT pipeline (prepare / train / infer / all)")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("prepare", help="Run convert_gsm8k_to_sft.py")
    pr.add_argument("--output", type=str, default=str(ROOT / "data" / "sft" / "gsm8k_sft.jsonl"))
    pr.add_argument("--source", choices=("hf", "jsonl"), default="hf")
    pr.add_argument("--input", type=str, help="JSONL path for --source jsonl")
    pr.add_argument("--splits", nargs="+", default=["train", "test"])
    pr.add_argument(
        "--strip-scratchpads",
        action="store_true",
        help="Remove GSM8K <<...>> traces from assistant text after conversion.",
    )
    pr.set_defaults(func=cmd_prepare)

    tr = sub.add_parser("train", help="QLoRA SFT on JSONL with messages field")
    tr.add_argument("--data", type=str, required=True, help="JSONL from prepare step")
    tr.add_argument("--output-dir", type=str, required=True)
    tr.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    tr.add_argument("--epochs", type=float, default=1.0)
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
    tr.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
        help="Used only if --warmup-steps is not set; converted to warmup_steps.",
    )
    tr.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="LR warmup steps; if set, overrides --warmup-ratio.",
    )
    tr.add_argument("--bf16", action="store_true", default=True)
    tr.add_argument("--no-bf16", dest="bf16", action="store_false")
    tr.add_argument("--fp16", action="store_true")
    tr.add_argument("--bnb-compute-dtype", type=str, default="bfloat16")
    tr.set_defaults(func=cmd_train)

    inf = sub.add_parser("infer", help="Generate with saved adapter")
    inf.add_argument("--adapter", type=str, required=True, help="Directory from train step")
    inf.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-Math-1.5B-Instruct",
        help="Must match base used in training if no pipeline_meta.json",
    )
    inf.add_argument("--problem", type=str, required=True)
    inf.add_argument("--max-new-tokens", type=int, default=1024)
    inf.add_argument("--temperature", type=float, default=0.7)
    inf.add_argument("--top-p", type=float, default=0.95)
    inf.add_argument("--greedy", action="store_true")
    inf.add_argument("--bnb-compute-dtype", type=str, default="bfloat16")
    inf.set_defaults(func=cmd_infer)

    al = sub.add_parser("all", help="prepare + train [+ infer if --problem]")
    al.add_argument("--data", type=str, default=None, help="Output JSONL path (default data/sft/gsm8k_sft.jsonl)")
    al.add_argument("--prepare-source", choices=("hf", "jsonl"), default="hf")
    al.add_argument("--input", type=str, help="For jsonl prepare")
    al.add_argument("--splits", nargs="+", default=["train", "test"])
    al.add_argument("--strip-scratchpads", action="store_true")
    al.add_argument("--output-dir", type=str, required=True)
    al.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    al.add_argument("--epochs", type=float, default=1.0)
    al.add_argument("--batch-size", type=int, default=1)
    al.add_argument("--grad-accum", type=int, default=8)
    al.add_argument("--learning-rate", type=float, default=2e-4)
    al.add_argument("--max-samples", type=int, default=0)
    al.add_argument("--lora-rank", type=int, default=16)
    al.add_argument("--lora-alpha", type=int, default=32)
    al.add_argument("--lora-dropout", type=float, default=0.05)
    al.add_argument("--target-modules", type=str, default="q_proj,v_proj,o_proj,gate_proj")
    al.add_argument("--max-seq-length", type=int, default=2048)
    al.add_argument("--save-steps", type=int, default=200)
    al.add_argument("--logging-steps", type=int, default=10)
    al.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
        help="Used only if --warmup-steps is not set; converted to warmup_steps.",
    )
    al.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="LR warmup steps; if set, overrides --warmup-ratio.",
    )
    al.add_argument("--bf16", action="store_true", default=True)
    al.add_argument("--no-bf16", dest="bf16", action="store_false")
    al.add_argument("--fp16", action="store_true")
    al.add_argument("--bnb-compute-dtype", type=str, default="bfloat16")
    al.add_argument("--problem", type=str, default="", help="If set, run infer after train")
    al.add_argument("--max-new-tokens", type=int, default=1024)
    al.add_argument("--temperature", type=float, default=0.7)
    al.add_argument("--top-p", type=float, default=0.95)
    al.add_argument("--greedy", action="store_true")
    al.set_defaults(func=cmd_all)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    args.func(args)


if __name__ == "__main__":
    main()
