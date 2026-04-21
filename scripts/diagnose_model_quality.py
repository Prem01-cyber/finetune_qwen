#!/usr/bin/env python3
"""
Diagnostic Script for SFT Model Quality and Consensus System

This script runs comprehensive diagnostics on your trained model:
1. Format compliance (Step N:, Final Answer:)
2. SymPy verification success rate
3. Consensus system functionality
4. Answer extraction reliability

Use this BEFORE starting PPO training to ensure your SFT model is ready.

Usage:
    python scripts/diagnose_model_quality.py \
        --adapter checkpoints/dual_task_v1 \
        --num-samples 20 \
        --wandb-project "math-diagnostics"
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import torch
import wandb
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sft.solution_format import validate_sympy_solution_format
from src.rl.triple_verifier import TripleVerifier


def load_model(adapter_path: str, base_model: str):
    """Load model with adapter."""
    print(f"Loading base model: {base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print(f"Loading adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base, adapter_path)
    model = model.merge_and_unload()
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def generate_question(model, tokenizer, prompt: str) -> str:
    """Generate a question from a prompt."""
    system_prompt = (
        "You are a math problem generator. "
        "Generate grade-school level math word problems that require 2-5 steps to solve."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"### Task: Generate Question\n{prompt}"},
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    gen_ids = outputs[0, inputs["input_ids"].shape[1]:]
    question = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return question


def generate_solution(model, tokenizer, question: str) -> str:
    """Generate a solution for a question."""
    system_prompt = (
        "You are a step-by-step math solver. "
        "Solve the given problem one step at a time. "
        "Each step must be on its own line, starting with 'Step N:'. "
        "End with a line starting with 'Final Answer:'. "
        "Write every mathematical expression in Python/SymPy syntax."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"### Task: Solve Problem\nProblem: {question}\nSolution:"},
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    gen_ids = outputs[0, inputs["input_ids"].shape[1]:]
    solution = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return solution


def test_format_compliance(model, tokenizer, num_samples: int = 20) -> Dict:
    """Test format compliance with question generation and solution."""
    print("\n" + "="*80)
    print("TEST 1: FORMAT COMPLIANCE")
    print("="*80)
    
    prompts = [
        "Create a word problem about addition with 2 steps.",
        "Create a problem about subtraction requiring 3 steps.",
        "Create a multiplication word problem with money.",
        "Create a division problem about sharing items.",
        "Create a problem combining addition and subtraction.",
    ]
    
    results = []
    format_ok = 0
    has_steps = 0
    has_final_answer = 0
    sympy_ok = 0
    
    for i in range(num_samples):
        prompt = prompts[i % len(prompts)]
        
        # Generate Q&A pair
        question = generate_question(model, tokenizer, prompt)
        solution = generate_solution(model, tokenizer, question)
        
        # Validate format
        validation = validate_sympy_solution_format(solution)
        
        if validation.ok:
            format_ok += 1
        if validation.step_count > 0:
            has_steps += 1
        if validation.has_final_line:
            has_final_answer += 1
        if validation.ok and validation.sympy_parseable_final:
            sympy_ok += 1
        
        result = {
            "sample_id": i,
            "prompt": prompt,
            "question": question,
            "solution": solution,
            "format_ok": validation.ok,
            "steps_found": validation.step_count > 0,
            "final_answer_found": validation.has_final_line,
            "sympy_parseable": validation.sympy_parseable_final,
            "num_steps": validation.step_count,
        }
        results.append(result)
        
        # Log to console
        status = "✓" if validation.ok else "✗"
        print(f"Sample {i+1:2d}/{num_samples}: {status} | Steps: {validation.step_count} | Answer: {validation.has_final_line} | SymPy: {validation.sympy_parseable_final}")
    
    summary = {
        "num_samples": num_samples,
        "format_ok_rate": format_ok / num_samples,
        "has_steps_rate": has_steps / num_samples,
        "has_final_answer_rate": has_final_answer / num_samples,
        "sympy_ok_rate": sympy_ok / num_samples,
    }
    
    print(f"\nSummary:")
    print(f"  Format OK: {format_ok}/{num_samples} ({summary['format_ok_rate']:.1%})")
    print(f"  Has Steps: {has_steps}/{num_samples} ({summary['has_steps_rate']:.1%})")
    print(f"  Has Final Answer: {has_final_answer}/{num_samples} ({summary['has_final_answer_rate']:.1%})")
    print(f"  SymPy OK: {sympy_ok}/{num_samples} ({summary['sympy_ok_rate']:.1%})")
    
    return {
        "summary": summary,
        "results": results,
    }


def test_consensus_system(model, tokenizer, num_samples: int = 10) -> Dict:
    """Test consensus verification system."""
    print("\n" + "="*80)
    print("TEST 2: CONSENSUS SYSTEM")
    print("="*80)
    
    verifier = TripleVerifier(
        model=model,
        tokenizer=tokenizer,
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
    )
    
    # Test questions with known answers
    test_cases = [
        ("A store has 100 apples. They sell 40. How many remain?", 60),
        ("Janet has 16 eggs. She eats 3. How many are left?", 13),
        ("There are 25 students. 10 leave. How many stay?", 15),
        ("A baker makes 50 cookies. He sells 20. How many left?", 30),
        ("Tom has 8 toys. He gets 5 more. How many total?", 13),
    ]
    
    results = []
    consensus_found = 0
    primary_matches = 0
    correct_answers = 0
    avg_diversity = 0
    
    for i in range(min(num_samples, len(test_cases))):
        question, expected = test_cases[i]
        
        # Generate primary solution
        primary = generate_solution(model, tokenizer, question)
        
        # Run triple verification
        print(f"\nTest {i+1}: {question}")
        verification = verifier.verify_with_triple_check(question, primary)
        
        consensus = verification["consensus"]
        has_majority = consensus["has_majority"]
        primary_matches_maj = consensus["primary_matches_majority"]
        majority_answer = consensus["majority_answer"]
        answer_diversity = consensus["answer_diversity"]
        
        if has_majority:
            consensus_found += 1
        if primary_matches_maj:
            primary_matches += 1
        if majority_answer is not None and abs(majority_answer - expected) < 1e-6:
            correct_answers += 1
        avg_diversity += answer_diversity
        
        result = {
            "sample_id": i,
            "question": question,
            "expected_answer": expected,
            "majority_answer": majority_answer,
            "primary_answer": consensus["primary_answer"],
            "has_consensus": has_majority,
            "primary_matches": primary_matches_maj,
            "answer_diversity": answer_diversity,
            "consensus_strength": consensus["consensus_strength"],
            "is_correct": majority_answer is not None and abs(majority_answer - expected) < 1e-6,
        }
        results.append(result)
        
        # Log to console
        status = "✓" if has_majority else "✗"
        match_status = "✓" if primary_matches_maj else "✗"
        correct_status = "✓" if result["is_correct"] else "✗"
        print(f"  Consensus: {status} | Primary Matches: {match_status} | Correct: {correct_status}")
        print(f"  Answers: {consensus['answers']} → Majority: {majority_answer} (expected: {expected})")
        print(f"  Diversity: {answer_diversity}, Strength: {consensus['consensus_strength']:.2f}")
    
    n = min(num_samples, len(test_cases))
    avg_diversity /= n
    
    summary = {
        "num_samples": n,
        "consensus_rate": consensus_found / n,
        "primary_match_rate": primary_matches / n,
        "accuracy": correct_answers / n,
        "avg_diversity": avg_diversity,
    }
    
    print(f"\nSummary:")
    print(f"  Consensus Found: {consensus_found}/{n} ({summary['consensus_rate']:.1%})")
    print(f"  Primary Matches Majority: {primary_matches}/{n} ({summary['primary_match_rate']:.1%})")
    print(f"  Correct Answers: {correct_answers}/{n} ({summary['accuracy']:.1%})")
    print(f"  Avg Answer Diversity: {avg_diversity:.2f}")
    
    return {
        "summary": summary,
        "results": results,
    }


def generate_recommendations(format_test: Dict, consensus_test: Dict) -> Dict:
    """Generate recommendations based on diagnostic results."""
    format_summary = format_test["summary"]
    consensus_summary = consensus_test["summary"]
    
    recommendations = {
        "status": "unknown",
        "ready_for_ppo": False,
        "issues": [],
        "suggested_actions": [],
    }
    
    # Check format compliance
    if format_summary["format_ok_rate"] < 0.7:
        recommendations["issues"].append("❌ Format compliance too low (<70%)")
        recommendations["suggested_actions"].append(
            "CRITICAL: Retrain SFT model with more epochs (3-4) or better data quality"
        )
    elif format_summary["format_ok_rate"] < 0.85:
        recommendations["issues"].append("⚠️ Format compliance moderate (70-85%)")
        recommendations["suggested_actions"].append(
            "RECOMMENDED: Retrain SFT for better format compliance, or proceed with caution"
        )
    else:
        recommendations["issues"].append("✓ Format compliance good (≥85%)")
    
    # Check consensus functionality
    if consensus_summary["consensus_rate"] < 0.3:
        recommendations["issues"].append("❌ Consensus system not working (<30% consensus)")
        recommendations["suggested_actions"].append(
            "CRITICAL: Check answer extraction - model may not be producing valid 'Final Answer:' lines"
        )
    elif consensus_summary["consensus_rate"] < 0.6:
        recommendations["issues"].append("⚠️ Consensus rate moderate (30-60%)")
        recommendations["suggested_actions"].append(
            "RECOMMENDED: Improve solution quality or adjust consensus temperature"
        )
    else:
        recommendations["issues"].append("✓ Consensus system working (≥60%)")
    
    # Check accuracy
    if consensus_summary["accuracy"] < 0.5:
        recommendations["issues"].append("❌ Answer accuracy too low (<50%)")
        recommendations["suggested_actions"].append(
            "CRITICAL: Model is not solving problems correctly - retrain SFT"
        )
    elif consensus_summary["accuracy"] < 0.7:
        recommendations["issues"].append("⚠️ Answer accuracy moderate (50-70%)")
    else:
        recommendations["issues"].append("✓ Answer accuracy good (≥70%)")
    
    # Overall status
    if (format_summary["format_ok_rate"] >= 0.8 and 
        consensus_summary["consensus_rate"] >= 0.5 and 
        consensus_summary["accuracy"] >= 0.6):
        recommendations["status"] = "READY"
        recommendations["ready_for_ppo"] = True
        recommendations["suggested_actions"].insert(0, 
            "✅ MODEL IS READY FOR PPO TRAINING"
        )
    elif (format_summary["format_ok_rate"] >= 0.7 and 
          consensus_summary["consensus_rate"] >= 0.3):
        recommendations["status"] = "MARGINAL"
        recommendations["ready_for_ppo"] = False
        recommendations["suggested_actions"].insert(0,
            "⚠️ MODEL IS MARGINAL - Recommend retraining SFT before PPO"
        )
    else:
        recommendations["status"] = "NOT_READY"
        recommendations["ready_for_ppo"] = False
        recommendations["suggested_actions"].insert(0,
            "❌ MODEL NOT READY - Must retrain SFT before attempting PPO"
        )
    
    return recommendations


def main():
    parser = argparse.ArgumentParser(description="Diagnose SFT model quality")
    parser.add_argument("--adapter", type=str, required=True, help="Path to adapter")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of format test samples")
    parser.add_argument("--num-consensus", type=int, default=10, help="Number of consensus test samples")
    parser.add_argument("--wandb-project", type=str, default="math-diagnostics")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--output-json", type=str, help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"diagnostics_{Path(args.adapter).name}_{datetime.now():%Y%m%d_%H%M%S}",
            config={
                "adapter": args.adapter,
                "base_model": args.base_model,
                "num_samples": args.num_samples,
                "num_consensus": args.num_consensus,
            },
            tags=["diagnostics", "pre-ppo"],
        )
    
    print("="*80)
    print("MODEL QUALITY DIAGNOSTICS")
    print("="*80)
    print(f"Adapter: {args.adapter}")
    print(f"Base Model: {args.base_model}")
    print(f"Samples: {args.num_samples} (format), {args.num_consensus} (consensus)")
    
    # Load model
    model, tokenizer = load_model(args.adapter, args.base_model)
    
    # Test 1: Format compliance
    format_test = test_format_compliance(model, tokenizer, args.num_samples)
    
    # Test 2: Consensus system
    consensus_test = test_consensus_system(model, tokenizer, args.num_consensus)
    
    # Generate recommendations
    recommendations = generate_recommendations(format_test, consensus_test)
    
    # Print recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print(f"Status: {recommendations['status']}")
    print(f"Ready for PPO: {'YES ✓' if recommendations['ready_for_ppo'] else 'NO ✗'}")
    print("\nIssues:")
    for issue in recommendations["issues"]:
        print(f"  {issue}")
    print("\nSuggested Actions:")
    for i, action in enumerate(recommendations["suggested_actions"], 1):
        print(f"  {i}. {action}")
    
    # Log to wandb
    if not args.no_wandb:
        wandb.log({
            "format/format_ok_rate": format_test["summary"]["format_ok_rate"],
            "format/has_steps_rate": format_test["summary"]["has_steps_rate"],
            "format/has_final_answer_rate": format_test["summary"]["has_final_answer_rate"],
            "format/sympy_ok_rate": format_test["summary"]["sympy_ok_rate"],
            "consensus/consensus_rate": consensus_test["summary"]["consensus_rate"],
            "consensus/primary_match_rate": consensus_test["summary"]["primary_match_rate"],
            "consensus/accuracy": consensus_test["summary"]["accuracy"],
            "consensus/avg_diversity": consensus_test["summary"]["avg_diversity"],
            "recommendations/ready_for_ppo": 1.0 if recommendations["ready_for_ppo"] else 0.0,
        })
        
        # Create summary table
        table = wandb.Table(
            columns=["Metric", "Value", "Status"],
            data=[
                ["Format OK Rate", f"{format_test['summary']['format_ok_rate']:.1%}", 
                 "✓" if format_test['summary']['format_ok_rate'] >= 0.8 else "✗"],
                ["Consensus Rate", f"{consensus_test['summary']['consensus_rate']:.1%}",
                 "✓" if consensus_test['summary']['consensus_rate'] >= 0.5 else "✗"],
                ["Answer Accuracy", f"{consensus_test['summary']['accuracy']:.1%}",
                 "✓" if consensus_test['summary']['accuracy'] >= 0.6 else "✗"],
                ["Ready for PPO", "Yes" if recommendations["ready_for_ppo"] else "No",
                 "✓" if recommendations["ready_for_ppo"] else "✗"],
            ]
        )
        wandb.log({"diagnostic_summary": table})
        
        wandb.finish()
    
    # Save to JSON
    if args.output_json:
        output = {
            "config": vars(args),
            "format_test": format_test,
            "consensus_test": consensus_test,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat(),
        }
        
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    # Exit code based on readiness
    if recommendations["ready_for_ppo"]:
        print("\n✅ All diagnostics passed! Ready to proceed with PPO training.")
        sys.exit(0)
    else:
        print(f"\n❌ Diagnostics failed. Status: {recommendations['status']}")
        print("Please address the issues above before starting PPO training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
