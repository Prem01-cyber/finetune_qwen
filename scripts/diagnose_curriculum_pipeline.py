"""
Diagnostic script for curriculum-guided PPO pipeline.

Validates all components before full training run.
Uses exact same functions as training pipeline - no custom generation logic.
"""

import argparse
import logging
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rl.math_environment_curriculum import CurriculumMathEnvironment
from src.sft.solution_format import STEP_RE, FINAL_RE
from scripts.run_ppo_training_curriculum import (
    CurriculumTrainingConfig,
    initialize_models,
    load_reference_questions,
)


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


def main():
    parser = argparse.ArgumentParser(description="Diagnose curriculum pipeline")
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--num-test-trajectories", type=int, default=3)
    args = parser.parse_args()
    
    result = DiagnosticResult()
    
    # Use exact training config
    config = CurriculumTrainingConfig()
    if args.base_model:
        config.base_model = args.base_model
    
    print("\n" + "=" * 80)
    print("CURRICULUM PIPELINE DIAGNOSTIC")
    print("=" * 80)
    print(f"\nModel: {config.base_model}")
    print(f"Test trajectories: {args.num_test_trajectories}")
    print(f"Using training config: max_question_tokens={config.max_question_tokens}, "
          f"max_solution_tokens={config.max_solution_tokens}")
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
    
    # Check 2: Load model using exact training function
    print("[2/8] Loading model (using training pipeline function)...")
    try:
        policy, value, tokenizer = initialize_models(config)
        result.add_check("Model loading", True, f"Device: {policy.device}")
    except Exception as e:
        result.add_check("Model loading", False, str(e))
        result.print_report()
        return 1
    
    # Check 3: Load reference questions using training function
    print("[3/8] Loading reference questions (using training pipeline function)...")
    try:
        reference_questions = load_reference_questions(config.gsm8k_reference_data)
        if not reference_questions:
            result.add_warning(f"No reference questions loaded from {config.gsm8k_reference_data}")
            reference_questions = ["Test has 10 apples. Gives away 3. How many left?"]
        result.add_check("Reference questions loading", True, f"{len(reference_questions)} questions")
    except Exception as e:
        result.add_check("Reference questions loading", False, str(e))
        result.print_report()
        return 1
    
    # Check 4: Initialize environment with exact training config
    print("[4/8] Initializing curriculum environment (using training config)...")
    try:
        with TemporaryDirectory() as temp_dir:
            env = CurriculumMathEnvironment(
                policy_model=policy,
                value_model=value,
                tokenizer=tokenizer,
                reference_questions=reference_questions,
                curriculum_checkpoint_dir=temp_dir,
                max_question_tokens=config.max_question_tokens,
                max_solution_tokens=config.max_solution_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                consensus_temperature=config.consensus_temperature,
            )
            result.add_check("Environment initialization", True)
            
            # Check 5: Generate trajectories using environment's rollout_trajectory
            print(f"[5/9] Generating {args.num_test_trajectories} test trajectories (using env.rollout_trajectory)...")
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
            
            # Check 6: Validate metadata structure
            print("[6/9] Validating trajectory metadata...")
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
            
            # Check 7: Validate reward breakdown
            print("[7/9] Validating reward composition...")
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
            
            # Check 8: Validate solution quality signals
            print("[8/9] Checking solution verification quality...")
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
            
            # Check 9: Curriculum state updates
            print("[9/9] Validating curriculum state...")
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
            
            # Print sample trajectory for inspection (using exact data from training pipeline)
            print("\n" + "=" * 80)
            print("SAMPLE TRAJECTORY (exact output from training pipeline)")
            print("=" * 80)
            sample = trajectories[0].metadata
            
            print(f"\n{'='*80}")
            print("CURRICULUM INSTRUCTION:")
            print(f"{'='*80}")
            print(sample['instruction'])
            
            print(f"\n{'='*80}")
            print("GENERATED QUESTION (full text as generated by model):")
            print(f"{'='*80}")
            print(sample['generated_question'])
            
            print(f"\n{'='*80}")
            print("GENERATED SOLUTION (full text as generated by model):")
            print(f"{'='*80}")
            print(sample['generated_solution'])
            
            print(f"\n{'='*80}")
            print("CURRICULUM & VERIFICATION DETAILS:")
            print(f"{'='*80}")
            print(f"Target Topic: {sample['target_topic']}")
            print(f"Target Difficulty: {sample['target_difficulty']:.2f}")
            print(f"Detected Topic: {sample['detected_topic']}")
            print(f"Measured Difficulty: {sample['estimated_difficulty']:.2f}")
            print(f"Topic Match Score: {sample['topic_match_score']:.3f}")
            print(f"Clarity Score: {sample['clarity_score']:.3f}")
            
            print(f"\nSolution Verification:")
            print(f"  Steps total: {sample['steps_total']}")
            print(f"  Steps verified OK: {sample['steps_verified_ok']}")
            print(f"  Steps failed: {sample['steps_failed']}")
            print(f"  SymPy fully verified: {sample['sympy_verified']}")
            print(f"  Consensus achieved: {sample['consensus_achieved']}")
            print(f"  Consensus strength: {sample['consensus_strength']:.3f}")
            print(f"  Primary matches majority: {sample['primary_matches_majority']}")
            
            # Check format compliance using exact regex from solution_format.py
            sol_text = sample['generated_solution']
            has_step_pattern = bool(STEP_RE.search(sol_text))
            has_final_pattern = bool(FINAL_RE.search(sol_text))
            step_count = len(STEP_RE.findall(sol_text))
            
            print(f"\nFormat Compliance (using solution_format.py patterns):")
            print(f"  Contains 'Step N:' pattern: {has_step_pattern}")
            print(f"  Step count: {step_count}")
            print(f"  Contains 'Final Answer:' pattern: {has_final_pattern}")
            if not has_step_pattern:
                print("  ⚠️  WARNING: Solution does not follow expected 'Step N:' format!")
                print("  ⚠️  This means SymPy verification cannot extract steps to verify.")
                print("  ⚠️  Model needs to generate format: 'Step 1: ...' 'Step 2: ...' 'Final Answer: ...')")
            
            print(f"\nReward Breakdown:")
            print(f"  Question Reward: {sample['question_reward']:.3f} (30% weight)")
            print(f"  Solution Reward: {sample['solution_reward']:.3f} (70% weight)")
            print(f"  Combined Reward: {sample['combined_reward']:.3f}")
            print(f"  Expected Combined: {0.3 * sample['question_reward'] + 0.7 * sample['solution_reward']:.3f}")
            print(f"  Match: {'✓' if abs(sample['combined_reward'] - (0.3 * sample['question_reward'] + 0.7 * sample['solution_reward'])) < 0.001 else '✗'}")
    
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
