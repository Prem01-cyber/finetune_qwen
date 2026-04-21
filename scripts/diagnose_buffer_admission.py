#!/usr/bin/env python3
"""
Diagnostic script to analyze replay buffer admission bottlenecks.

Run this after each training iteration to understand why trajectories
are not entering the replay buffer.

Usage:
    python scripts/diagnose_buffer_admission.py checkpoints/ppo_training_curriculum/iteration_003/trajectories.jsonl
"""

import json
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, List


def analyze_quality_filter_bottleneck(trajectories: List[Dict]) -> None:
    """Diagnose which quality filter criteria are blocking buffer admission."""
    
    total = len(trajectories)
    
    # Initialize counters
    stats = {
        'total': total,
        'reward_ok': 0,
        'reward_silver_ok': 0,
        'reward_platinum_ok': 0,
        'sympy_ok': 0,
        'sympy_perfect': 0,
        'consensus_ok': 0,
        'primary_matches_ok': 0,
        'consensus_strong': 0,
        'topic_match_ok': 0,
        'gold_eligible': 0,
        'silver_eligible': 0,
        'platinum_eligible': 0,
        'replay_added': 0,
    }
    
    reject_reasons = Counter()
    
    for traj in trajectories:
        meta = traj.get('metadata', {})
        
        # Reward tiers
        reward = float(meta.get('combined_reward', 0.0))
        if reward >= 0.7:
            stats['reward_ok'] += 1
        if reward >= 0.75:
            stats['reward_silver_ok'] += 1
        if reward >= 0.95:
            stats['reward_platinum_ok'] += 1
        
        # SymPy verification
        if meta.get('sympy_verified'):
            stats['sympy_ok'] += 1
        sympy_score = float(meta.get('reward_breakdown', {}).get('solution_metrics', {}).get('sympy_score', 0.0))
        if sympy_score >= 0.95:
            stats['sympy_perfect'] += 1
        
        # Consensus
        if meta.get('consensus_achieved'):
            stats['consensus_ok'] += 1
        if meta.get('primary_matches_majority'):
            stats['primary_matches_ok'] += 1
        consensus_strength = float(meta.get('consensus_strength', 0.0))
        if consensus_strength >= 0.8:
            stats['consensus_strong'] += 1
        
        # Topic match
        if float(meta.get('topic_match_score', 0.0)) >= 0.6:
            stats['topic_match_ok'] += 1
        
        # Tier eligibility
        # Platinum: reward >= 0.95
        if reward >= 0.95:
            stats['platinum_eligible'] += 1
        
        # Gold: reward >= 0.7 + consensus + sympy + topic
        if (reward >= 0.7 and
            meta.get('consensus_achieved') and
            meta.get('primary_matches_majority') and
            meta.get('sympy_verified') and
            float(meta.get('topic_match_score', 0.0)) >= 0.6):
            stats['gold_eligible'] += 1
        
        # Silver: reward >= 0.75 + (perfect_sympy OR strong_consensus) + topic
        if (reward >= 0.75 and
            (sympy_score >= 0.95 or 
             (meta.get('consensus_achieved') and consensus_strength >= 0.8)) and
            float(meta.get('topic_match_score', 0.0)) >= 0.6):
            stats['silver_eligible'] += 1
        
        # Track actual additions
        if meta.get('replay_added'):
            stats['replay_added'] += 1
        
        # Track reject reasons
        if not meta.get('replay_added'):
            reason = meta.get('replay_reject_reason', 'unknown')
            reject_reasons[reason] += 1
    
    # Print report
    print("\n" + "="*80)
    print("REPLAY BUFFER ADMISSION DIAGNOSTIC")
    print("="*80)
    
    print(f"\n📊 OVERVIEW")
    print(f"  Total trajectories: {total}")
    print(f"  Added to buffer: {stats['replay_added']} ({100*stats['replay_added']/total:.1f}%)")
    print(f"  Rejected: {total - stats['replay_added']} ({100*(total-stats['replay_added'])/total:.1f}%)")
    
    print(f"\n💎 TIER ELIGIBILITY")
    print(f"  Platinum tier (≥0.95 reward): {stats['platinum_eligible']} ({100*stats['platinum_eligible']/total:.1f}%)")
    print(f"  Gold tier (0.7+ reward + both signals): {stats['gold_eligible']} ({100*stats['gold_eligible']/total:.1f}%)")
    print(f"  Silver tier (0.75+ reward + one strong signal): {stats['silver_eligible']} ({100*stats['silver_eligible']/total:.1f}%)")
    total_eligible = stats['platinum_eligible'] + stats['gold_eligible'] + stats['silver_eligible']
    print(f"  Total eligible (any tier): {total_eligible} ({100*total_eligible/total:.1f}%)")
    
    print(f"\n🎯 REWARD DISTRIBUTION")
    print(f"  Reward ≥ 0.70 (gold threshold): {stats['reward_ok']} ({100*stats['reward_ok']/total:.1f}%)")
    print(f"  Reward ≥ 0.75 (silver threshold): {stats['reward_silver_ok']} ({100*stats['reward_silver_ok']/total:.1f}%)")
    print(f"  Reward ≥ 0.95 (platinum threshold): {stats['reward_platinum_ok']} ({100*stats['reward_platinum_ok']/total:.1f}%)")
    
    print(f"\n✅ SYMPY VERIFICATION")
    print(f"  SymPy verified (no failures): {stats['sympy_ok']} ({100*stats['sympy_ok']/total:.1f}%)")
    print(f"  SymPy perfect score (≥0.95): {stats['sympy_perfect']} ({100*stats['sympy_perfect']/total:.1f}%)")
    
    print(f"\n🤝 CONSENSUS STATUS")
    print(f"  Has majority (≥2/3 agree): {stats['consensus_ok']} ({100*stats['consensus_ok']/total:.1f}%)")
    print(f"  Primary matches majority: {stats['primary_matches_ok']} ({100*stats['primary_matches_ok']/total:.1f}%)")
    print(f"  Strong consensus (≥0.8): {stats['consensus_strong']} ({100*stats['consensus_strong']/total:.1f}%)")
    print(f"  Both consensus checks pass: {stats['primary_matches_ok']} ({100*stats['primary_matches_ok']/total:.1f}%)")
    
    print(f"\n🎓 TOPIC MATCH")
    print(f"  Topic match ≥ 0.6: {stats['topic_match_ok']} ({100*stats['topic_match_ok']/total:.1f}%)")
    
    print(f"\n❌ REJECTION REASONS")
    for reason, count in reject_reasons.most_common(10):
        print(f"  {reason}: {count} ({100*count/total:.1f}%)")
    
    # Bottleneck analysis
    print(f"\n🔍 BOTTLENECK ANALYSIS")
    
    # Check consensus bottleneck
    consensus_pass_rate = 100 * stats['primary_matches_ok'] / total
    if consensus_pass_rate < 30:
        print(f"  ⚠️  CRITICAL: Consensus rate is {consensus_pass_rate:.1f}% (target: 80%+)")
        print(f"      → Consider lowering consensus temperature further (current: 0.5)")
        print(f"      → Or relax silver tier to accept weaker consensus")
    elif consensus_pass_rate < 50:
        print(f"  ⚠️  WARNING: Consensus rate is {consensus_pass_rate:.1f}% (target: 80%+)")
        print(f"      → Should improve with more training iterations")
    else:
        print(f"  ✓ Consensus rate is healthy: {consensus_pass_rate:.1f}%")
    
    # Check reward bottleneck
    if stats['reward_ok'] < total * 0.3:
        print(f"  ⚠️  CRITICAL: Only {100*stats['reward_ok']/total:.1f}% of trajectories exceed reward threshold")
        print(f"      → Training may need more iterations to improve base quality")
    
    # Check eligibility vs admission gap
    if total_eligible > stats['replay_added'] * 1.5:
        gap = total_eligible - stats['replay_added']
        print(f"  ⚠️  WARNING: {gap} eligible trajectories didn't make it to buffer")
        print(f"      → Likely blocked by novelty filter")
        print(f"      → Consider lowering novelty_threshold (current: 0.5)")
    
    print("\n" + "="*80)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/diagnose_buffer_admission.py <trajectories.jsonl>")
        print("\nExample:")
        print("  python scripts/diagnose_buffer_admission.py \\")
        print("    checkpoints/ppo_training_curriculum/iteration_003/trajectories.jsonl")
        sys.exit(1)
    
    trajectory_file = Path(sys.argv[1])
    
    if not trajectory_file.exists():
        print(f"Error: File not found: {trajectory_file}")
        sys.exit(1)
    
    # Load trajectories
    trajectories = []
    with trajectory_file.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                trajectories.append(json.loads(line))
    
    if not trajectories:
        print("Error: No trajectories found in file")
        sys.exit(1)
    
    print(f"\nLoaded {len(trajectories)} trajectories from {trajectory_file}")
    
    # Run analysis
    analyze_quality_filter_bottleneck(trajectories)


if __name__ == '__main__':
    main()
