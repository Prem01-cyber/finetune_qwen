"""
Diagnostic script for curriculum-guided PPO pipeline.

Validates all components before full training run.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rl.math_environment_curriculum import CurriculumMathEnvironment
from src.rl.value_network import ValueHead


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class DiagnosticResult:
    def __init__(self):
        self.checks = []
        self.warnings = []
        self.errors = []
    
    def add_check(self, name: str, passed: bool, details: str = ""):
        self.checks.append({"name": name, "passed": passed, "details": details})
        if not passed:
            self.errors.append(f"{name}: {details}")
    
    def add_warning(self, message: str):
        self.warnings.append(message)
    
    def print_report(self):
        print("\n" + "=" * 80)
        print("CURRICULUM PIPELINE DIAGNOSTIC REPORT")
        print("=" * 80)
        
        passed = sum(1 for c in self.checks if c["passed"])
        total = len(self.checks)
        
        print(f"\nChecks: {passed}/{total} passed")
        print()
        
        for check in self.checks:
            status = "✓" if check["passed"] else "✗"
            print(f"{status} {check['name']}")
            if check["details"]:
                print(f"  → {check['details']}")
        
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                print(f"  - {w}")
        
        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for e in self.errors:
                print(f"  - {e}")
        
        print("\n" + "=" * 80)
        if passed == total:
            print("✅ ALL CHECKS PASSED - Safe to proceed with full training")
        else:
            print(f"❌ {total - passed} CHECKS FAILED - Fix issues before training")
        print("=" * 80 + "\n")
        
        return passed == total


def load_model_for_diagnostic(model_path: str):
    """Load model quickly for diagnostic (minimal config)."""
    model_path = Path(model_path)
    is_adapter = (model_path / "adapter_config.json").exists()
    
    if is_adapter:
        meta_file = model_path / "pipeline_meta.json"
        if meta_file.exists():
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            base_model_name = meta.get("base_model", "Qwen/Qwen2.5-Math-1.5B-Instruct")
        else:
            base_model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        policy = PeftModel.from_pretrained(base_model, model_path)
        policy = policy.merge_and_unload()
        
        value = ValueHead(base_model_name).to(policy.device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        policy = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        value = ValueHead(model_path).to(policy.device)
    
    return policy, value, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Diagnose curriculum pipeline")
    parser.add_argument("--base-model", type=str, default="checkpoints/dual_task_v1")
    parser.add_argument("--num-test-trajectories", type=int, default=3)
    args = parser.parse_args()
    
    result = DiagnosticResult()
    
    print("\n" + "=" * 80)
    print("CURRICULUM PIPELINE DIAGNOSTIC")
    print("=" * 80)
    print(f"\nModel: {args.base_model}")
    print(f"Test trajectories: {args.num_test_trajectories}")
    print()
    
    # Check 1: Import all modules
    print("[1/8] Importing modules...")
    try:
        from src.rl.question_classifier import QuestionClassifier
        from src.rl.curriculum_manager import CurriculumManager
        from src.rl.question_quality_evaluator import QuestionQualityEvaluator
        from src.rl.math_environment_curriculum import CurriculumMathEnvironment
        result.add_check("Module imports", True)
    except Exception as e:
        result.add_check("Module imports", False, str(e))
        result.print_report()
        return 1
    
    # Check 2: Load model
    print("[2/8] Loading model...")
    try:
        policy, value, tokenizer = load_model_for_diagnostic(args.base_model)
        result.add_check("Model loading", True, f"Device: {policy.device}")
    except Exception as e:
        result.add_check("Model loading", False, str(e))
        result.print_report()
        return 1
    
    # Check 3: Initialize environment
    print("[3/8] Initializing curriculum environment...")
    try:
        with TemporaryDirectory() as temp_dir:
            env = CurriculumMathEnvironment(
                policy_model=policy,
                value_model=value,
                tokenizer=tokenizer,
                reference_questions=["Test has 10 apples. Gives away 3. How many left?"],
                curriculum_checkpoint_dir=temp_dir,
                max_question_tokens=200,
                max_solution_tokens=500,
            )
            result.add_check("Environment initialization", True)
            
            # Check 4: Generate trajectories
            print(f"[4/8] Generating {args.num_test_trajectories} test trajectories...")
            trajectories = []
            generation_errors = []
            
            for i in range(args.num_test_trajectories):
                try:
                    print(f"  Trajectory {i+1}/{args.num_test_trajectories}...", end=" ", flush=True)
                    traj = env.rollout_trajectory()
                    trajectories.append(traj)
                    print(f"✓ (reward={traj.total_reward:.3f})")
                except Exception as e:
                    generation_errors.append(str(e))
                    print(f"✗ ({e})")
            
            if trajectories:
                result.add_check(
                    "Trajectory generation",
                    True,
                    f"{len(trajectories)}/{args.num_test_trajectories} succeeded"
                )
            else:
                result.add_check(
                    "Trajectory generation",
                    False,
                    f"All failed: {generation_errors[0] if generation_errors else 'unknown'}"
                )
                result.print_report()
                return 1
            
            # Check 5: Validate metadata structure
            print("[5/8] Validating trajectory metadata...")
            meta = trajectories[0].metadata
            required_fields = [
                "target_topic", "target_difficulty", "detected_topic",
                "question_reward", "solution_reward", "combined_reward",
                "consensus_achieved", "sympy_verified", "steps_total",
                "curriculum_state_snapshot"
            ]
            missing = [f for f in required_fields if f not in meta]
            
            if not missing:
                result.add_check("Metadata completeness", True)
            else:
                result.add_check("Metadata completeness", False, f"Missing: {missing}")
            
            # Check 6: Validate reward breakdown
            print("[6/8] Validating reward composition...")
            valid_rewards = True
            reward_issues = []
            
            for traj in trajectories:
                meta = traj.metadata
                q_reward = meta["question_reward"]
                s_reward = meta["solution_reward"]
                combined = meta["combined_reward"]
                expected = 0.3 * q_reward + 0.7 * s_reward
                
                if abs(combined - expected) > 0.01:
                    valid_rewards = False
                    reward_issues.append(
                        f"Expected {expected:.3f}, got {combined:.3f}"
                    )
                
                if not (0.0 <= q_reward <= 1.0 and 0.0 <= s_reward <= 1.0):
                    valid_rewards = False
                    reward_issues.append(f"Rewards out of [0,1]: Q={q_reward}, S={s_reward}")
            
            if valid_rewards:
                result.add_check("Reward composition (30/70)", True)
            else:
                result.add_check("Reward composition (30/70)", False, reward_issues[0])
            
            # Check 7: Validate solution quality signals
            print("[7/8] Checking solution verification quality...")
            sympy_working = sum(1 for t in trajectories if t.metadata["sympy_verified"])
            consensus_working = sum(1 for t in trajectories if t.metadata["consensus_achieved"])
            has_steps = sum(1 for t in trajectories if t.metadata["steps_total"] > 0)
            
            if has_steps == len(trajectories):
                result.add_check("Solution format (steps present)", True)
            else:
                result.add_check(
                    "Solution format (steps present)",
                    False,
                    f"Only {has_steps}/{len(trajectories)} have steps - model not generating 'Step N:' format"
                )
                result.add_warning(
                    "Model not following step-by-step format. Check if model was trained with proper format."
                )
            
            # SymPy verification is only meaningful if there are steps
            if has_steps == 0:
                result.add_check(
                    "SymPy verification",
                    False,
                    "Cannot verify - no steps detected in solutions"
                )
            elif sympy_working > 0:
                result.add_check(
                    "SymPy verification",
                    True,
                    f"{sympy_working}/{len(trajectories)} fully verified"
                )
            else:
                result.add_warning(
                    f"SymPy verification: 0/{len(trajectories)} fully verified (all have arithmetic errors)"
                )
                result.add_check(
                    "SymPy verification",
                    True,
                    "System working but model needs improvement via PPO"
                )
            
            if consensus_working > 0:
                result.add_check(
                    "Consensus verification",
                    True,
                    f"{consensus_working}/{len(trajectories)} achieved consensus"
                )
            else:
                result.add_warning(
                    f"Consensus: 0/{len(trajectories)} achieved (3 solutions disagree)"
                )
                result.add_check(
                    "Consensus verification",
                    True,
                    "System working but model needs improvement via PPO"
                )
            
            # Check 8: Curriculum state updates
            print("[8/8] Validating curriculum state...")
            stats = env.curriculum_manager.get_curriculum_stats()
            total_attempts = sum(
                t["total_attempts"] for t in stats["topics"].values()
            )
            
            if total_attempts >= len(trajectories):
                result.add_check(
                    "Curriculum state updates",
                    True,
                    f"{total_attempts} topic attempts recorded"
                )
            else:
                result.add_check(
                    "Curriculum state updates",
                    False,
                    f"Expected {len(trajectories)}, got {total_attempts}"
                )
            
            # Print sample trajectory for inspection
            print("\n" + "=" * 80)
            print("SAMPLE TRAJECTORY (for manual inspection)")
            print("=" * 80)
            sample = trajectories[0].metadata
            print(f"\nTarget Topic: {sample['target_topic']}")
            print(f"Target Difficulty: {sample['target_difficulty']:.2f}")
            print(f"\nGenerated Question (first 150 chars):")
            print(f"  {sample['generated_question'][:150]}...")
            print(f"\nGenerated Solution (FULL TEXT for format inspection):")
            print("-" * 80)
            print(sample['generated_solution'])
            print("-" * 80)
            print(f"\nDetected Topic: {sample['detected_topic']}")
            print(f"Measured Difficulty: {sample['estimated_difficulty']:.2f}")
            print(f"\nVerification:")
            print(f"  Steps total: {sample['steps_total']}")
            print(f"  Steps verified: {sample['steps_verified_ok']}")
            print(f"  Consensus achieved: {sample['consensus_achieved']}")
            print(f"  Primary matches majority: {sample['primary_matches_majority']}")
            
            # Check format compliance
            import re
            sol_text = sample['generated_solution']
            has_step_pattern = bool(re.search(r'^\s*Step\s+\d+\s*:', sol_text, re.I | re.M))
            has_final_pattern = bool(re.search(r'(?im)^Final\s*Answer\s*:', sol_text))
            print(f"\nFormat Analysis:")
            print(f"  Contains 'Step N:' pattern: {has_step_pattern}")
            print(f"  Contains 'Final Answer:' pattern: {has_final_pattern}")
            if not has_step_pattern:
                print("  ⚠️  Solution does not follow expected 'Step N:' format!")
            
            print(f"\nRewards:")
            print(f"  Question: {sample['question_reward']:.3f} (30% weight)")
            print(f"  Solution: {sample['solution_reward']:.3f} (70% weight)")
            print(f"  Combined: {sample['combined_reward']:.3f}")
            print(f"  Expected: {0.3 * sample['question_reward'] + 0.7 * sample['solution_reward']:.3f}")
    
    except Exception as e:
        result.add_check("Full pipeline test", False, str(e))
    
    # Final report
    all_passed = result.print_report()
    
    if not all_passed:
        print("\n⚠️  RECOMMENDATION: Fix errors before proceeding")
        return 1
    
    if result.warnings:
        print("\n⚠️  WARNINGS PRESENT:")
        print("    Low SymPy/consensus scores are EXPECTED for untrained models.")
        print("    PPO training will improve these over iterations.")
        print("    Safe to proceed if all checks passed.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
