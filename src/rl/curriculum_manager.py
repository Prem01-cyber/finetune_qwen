"""
Adaptive curriculum manager for dual-task math training.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.rl.question_classifier import QuestionClassifier, TOPIC_LIST

logger = logging.getLogger(__name__)

# Maps dataset skill_id prefixes → canonical curriculum topic names.
# Used during bootstrap so questions from NuminaMath / OpenMathInstruct
# are credited to the right ZPD bucket instead of being misclassified.
SKILL_ID_TO_TOPIC: dict[str, str] = {
    # NuminaMath-CoT
    "numina_algebra":        "algebra",
    "numina_prealgebra":     "algebra",
    "numina_number_theory":  "number_theory",
    "numina_geometry":       "geometry",
    "numina_combinatorics":  "combinatorics",
    "numina_calculus":       "calculus",
    "numina_statistics":     "statistics",
    "numina_synthetic":      "multi_step_reasoning",
    "numina_olympiad":       "competition_math",
    "numina_competition":    "competition_math",
    "numina_general":        "multi_step_reasoning",
    # OpenMathInstruct-2
    "openmath_algebra":      "algebra",
    "openmath_prealgebra":   "algebra",
    "openmath_number_theory":"number_theory",
    "openmath_geometry":     "geometry",
    "openmath_combinatorics":"combinatorics",
    "openmath_calculus":     "calculus",
    "openmath_competition":  "competition_math",
    "openmath_synthetic":    "multi_step_reasoning",
    "openmath_general":      "multi_step_reasoning",
    # Legacy
    "gsm8k_grade_school":    "basic_arithmetic",
    "aqua_rat_algebra":      "algebra",
    "question_generation":   "multi_step_reasoning",
}


@dataclass
class TopicState:
    topic_name: str
    total_attempts: int = 0
    successes: int = 0
    success_rate: float = 0.0
    difficulty_target: float = 0.5
    difficulty_history: List[float] = field(default_factory=list)
    last_practiced: int = 0
    first_attempted: int = 0
    status: str = "untested"
    mastered_at_iteration: Optional[int] = None
    retention_tests_passed: int = 0
    last_retention_score: float = 0.0
    consecutive_failures: int = 0
    failure_count_total: int = 0
    history: List[Dict[str, float]] = field(default_factory=list)
    current_iteration_attempts: int = 0  # Track attempts within current iteration


class CurriculumManager:
    """Goldilocks curriculum with adaptive topic selection."""

    SWEET_SPOT_MIN = 0.4
    SWEET_SPOT_MAX = 0.7
    TARGET_SUCCESS = 0.55

    CONTEXTS = ["bakery", "shopping", "school", "sports", "gardening", "travel"]
    ACTIONS = ["uses", "sells", "shares", "loses", "earns", "mixes"]

    TOPIC_TEMPLATES = {
        "basic_arithmetic": [
            "Generate a {context} word problem using addition/subtraction.",
        ],
        "single_step_word_problems": [
            "Generate a simple one-idea word problem in a {context} setting.",
        ],
        "fractions": [
            "Generate a fractions word problem where someone {action} part of a quantity in a {context} scenario.",
            "Create a problem involving fraction operations in {context}.",
        ],
        "percentages": [
            "Generate a percentage change or discount problem in {context}.",
        ],
        "ratios": [
            "Generate a ratios/proportions word problem in {context}.",
        ],
        "money_problems": [
            "Create a money and pricing problem in {context}.",
        ],
        "time_distance": [
            "Generate a time/speed/distance problem in {context}.",
        ],
        "multi_step_reasoning": [
            "Generate a multi-step reasoning problem in {context}.",
        ],
        "algebra": [
            "Generate an algebra problem that solves for a variable in {context}.",
        ],
        "mixed_operations": [
            "Generate a problem requiring mixed operations in {context}.",
        ],
        "comparison_problems": [
            "Generate a comparison problem in {context} ('more than'/'less than').",
        ],
        "optimization_problems": [
            "Generate a constrained optimization style word problem in {context}.",
        ],
        # ── AQuA-RAT additions ────────────────────────────────────────────
        "number_theory": [
            "Generate a number theory problem about divisibility, remainders, or prime factors in a {context} setting.",
            "Create a problem involving multiples and factors where someone in {context} {action} items in groups.",
        ],
        "profit_loss": [
            "Generate a profit and loss problem where someone in {context} {action} goods at cost price and selling price.",
            "Create a problem about percentage profit or loss on a transaction in {context}.",
        ],
        "interest": [
            "Generate a simple or compound interest problem involving a loan or investment in {context}.",
            "Create a problem where someone in {context} {action} money at a given annual interest rate.",
        ],
        "sets": [
            "Generate a set theory or Venn diagram problem where people in {context} belong to overlapping groups.",
            "Create a problem using union and intersection of two groups in {context}.",
        ],
        "combinatorics": [
            "Generate a combinatorics problem about arrangements or selections of objects in {context}.",
            "Create a problem involving permutations or combinations where someone in {context} {action} items.",
        ],
        "sequences": [
            "Generate an arithmetic or geometric sequence problem in {context}.",
            "Create a problem where someone in {context} follows a pattern and must find the nth term.",
        ],
        "probability": [
            "Generate a probability problem involving random selection or chance events in {context}.",
            "Create a problem where someone in {context} {action} items from a group and asks for probability.",
        ],
        "work_time": [
            "Generate a work-rate problem where two people in {context} complete a task together or alone.",
            "Create a problem where workers in {context} {action} a job at different rates.",
        ],
        # ── NuminaMath / OpenMathInstruct additions ───────────────────────
        "geometry": [
            "Generate a geometry problem about area or perimeter of a shape encountered in {context}.",
            "Create a problem involving triangles or circles where someone in {context} needs to find a missing length or angle.",
            "Generate a coordinate geometry problem where points in a {context} layout form a geometric figure.",
            "Create a problem involving volume or surface area of a 3D shape relevant to {context}.",
        ],
        "calculus": [
            "Generate a rate-of-change problem where a quantity in {context} grows or shrinks over time.",
            "Create an optimization problem where someone in {context} wants to maximise profit or minimise cost using calculus.",
            "Generate a problem involving a function whose minimum or maximum value must be found in a {context} scenario.",
        ],
        "statistics": [
            "Generate a statistics problem where someone in {context} collects data and must find the mean, median, or mode.",
            "Create a problem involving standard deviation or variance of measurements taken in {context}.",
            "Generate a problem where data from {context} is summarised and an outlier or expected value must be identified.",
        ],
        "competition_math": [
            "Generate a number theory problem asking how many positive integers satisfy a divisibility condition.",
            "Create a competition-style problem: find all integer solutions to an equation involving remainders or modular arithmetic.",
            "Generate a counting problem asking in how many ways objects can be arranged or selected under a constraint.",
            "Create a problem: given two integers relatively prime to each other, find their least common multiple or sum of digits.",
        ],
    }

    TOPIC_PREREQUISITES = {
        "fractions": ["basic_arithmetic"],
        "percentages": ["fractions"],
        "ratios": ["basic_arithmetic"],
        "algebra": ["basic_arithmetic", "comparison_problems"],
        "mixed_operations": ["basic_arithmetic", "fractions"],
        "optimization_problems": ["comparison_problems", "algebra"],
        # AQuA-RAT additions
        "number_theory": ["basic_arithmetic"],
        "profit_loss": ["percentages", "money_problems"],
        "interest": ["percentages"],
        "sets": ["basic_arithmetic"],
        "combinatorics": ["basic_arithmetic"],
        "sequences": ["basic_arithmetic", "algebra"],
        "probability": ["fractions", "ratios"],
        "work_time": ["ratios", "multi_step_reasoning"],
        # NuminaMath / OpenMathInstruct additions
        "geometry": ["basic_arithmetic"],
        "calculus": ["algebra", "sequences"],
        "statistics": ["ratios", "fractions"],
        "competition_math": ["number_theory", "combinatorics", "algebra"],
    }

    def __init__(self, checkpoint_dir: str | Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.classifier = QuestionClassifier()

        self.current_iteration = 0
        self.recent_combined_rewards: List[float] = []
        self.topics: Dict[str, TopicState] = {
            topic: TopicState(topic_name=topic) for topic in TOPIC_LIST
        }
        self.current_focus_topics: List[str] = []
        self.hyperparams = {
            "sweet_spot_min": self.SWEET_SPOT_MIN,
            "sweet_spot_max": self.SWEET_SPOT_MAX,
            "target_success": self.TARGET_SUCCESS,
        }

    def initialize(self, bootstrap_questions: Optional[List[str]] = None) -> None:
        """Initialize topic priors, optionally from GSM8K-style questions."""
        if bootstrap_questions:
            counts = {topic: 0 for topic in TOPIC_LIST}
            for question in bootstrap_questions:
                detected = self.classifier.classify_topic(question)
                topic = str(detected["primary_topic"])
                if topic in counts:
                    counts[topic] += 1

            total = max(1, sum(counts.values()))
            for topic, state in self.topics.items():
                prevalence = counts[topic] / total
                state.difficulty_target = float(max(0.3, min(0.75, 0.35 + 0.8 * prevalence)))
        else:
            for state in self.topics.values():
                state.difficulty_target = 0.5

    def initialize_from_dataset(
        self,
        records: List[Dict],
        difficulty_field: str = "difficulty",
    ) -> None:
        """
        Bootstrap curriculum topic priors directly from dataset skill_ids.

        Much faster than the question-classifier bootstrap path — reads
        skill_id and difficulty from each JSONL record rather than running
        the keyword classifier on every question.

        Args:
            records:          List of dataset records with 'skill_id' and
                              optionally 'difficulty' fields.
            difficulty_field: Name of the difficulty field (default 'difficulty').
                              Values: 1=easy, 2=medium, 3=hard.
        """
        counts: Dict[str, int] = {topic: 0 for topic in TOPIC_LIST}
        # Map difficulty 1/2/3 → difficulty_target 0.35/0.55/0.75
        _diff_map = {1: 0.35, 2: 0.55, 3: 0.75}
        topic_difficulties: Dict[str, List[float]] = {t: [] for t in TOPIC_LIST}

        for rec in records:
            skill_id = rec.get("skill_id", "")
            topic = SKILL_ID_TO_TOPIC.get(skill_id)
            if topic is None:
                # Fall back to keyword classifier on the question text
                msgs = rec.get("messages", [])
                question = next(
                    (m.get("content", "") for m in msgs if m.get("role") == "user"), ""
                )
                if question:
                    detected = self.classifier.classify_topic(question)
                    topic = str(detected["primary_topic"])
                else:
                    continue
            if topic not in counts:
                continue
            counts[topic] += 1
            raw_diff = rec.get(difficulty_field, 2)
            topic_difficulties[topic].append(_diff_map.get(int(raw_diff), 0.55))

        total = max(1, sum(counts.values()))
        for topic, state in self.topics.items():
            prevalence = counts[topic] / total
            # Difficulty target: average of observed difficulties, biased up
            # for rare topics (less data → start harder to find signal faster)
            diffs = topic_difficulties[topic]
            if diffs:
                avg_diff = sum(diffs) / len(diffs)
            else:
                avg_diff = 0.50

            # Blend prevalence-based prior with observed average difficulty
            state.difficulty_target = float(
                max(0.30, min(0.80, 0.4 * avg_diff + 0.6 * (0.35 + 0.8 * prevalence)))
            )

        logger.info(
            "Curriculum bootstrapped from %d records across %d topics",
            len(records),
            sum(1 for c in counts.values() if c > 0),
        )
        for topic, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            if cnt > 0:
                logger.debug(
                    "  %-30s  %4d samples  target_difficulty=%.2f",
                    topic, cnt, self.topics[topic].difficulty_target,
                )

    def select_topic_and_difficulty(self) -> Tuple[str, float]:
        probs = self._compute_topic_probabilities()
        names = list(probs.keys())
        dist = np.array([probs[name] for name in names], dtype=np.float64)
        dist = dist / dist.sum()
        
        # Log topic distribution at start of each iteration (rollout 0, 10, 20, etc.)
        total_attempts = sum(t.current_iteration_attempts for t in self.topics.values())
        if total_attempts % 20 == 0:  # Every 20 rollouts
            top_5 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info(f"Topic probabilities (rollout {total_attempts}): {[(t, f'{p:.3f}') for t, p in top_5]}")
        
        topic = str(np.random.choice(names, p=dist))
        difficulty = self._get_difficulty_for_topic(topic)
        self.current_focus_topics = [topic]
        return topic, difficulty

    def update_from_trajectory(
        self,
        topic: str,
        question_reward: float,
        solution_success: bool,
        combined_reward: Optional[float] = None,
        measured_difficulty: Optional[float] = None,
    ) -> None:
        # Grounded rollouts tag themselves with a synthetic topic name
        # (``grounded_gsm8k``) that isn't part of the curriculum ontology.
        # They must not pollute per-topic statistics — silently skip the
        # update for any unknown topic instead of crashing.  The combined
        # reward is still recorded for plateau detection.
        state = self.topics.get(topic)
        if state is None:
            logger.debug(
                "Skipping curriculum update for out-of-ontology topic %r", topic
            )
            if combined_reward is not None:
                self.recent_combined_rewards.append(float(combined_reward))
                self.recent_combined_rewards = self.recent_combined_rewards[-30:]
            return
        state.total_attempts += 1
        state.current_iteration_attempts += 1
        state.successes += int(solution_success)
        state.success_rate = state.successes / max(1, state.total_attempts)
        state.last_practiced = self.current_iteration
        if state.first_attempted == 0:
            state.first_attempted = self.current_iteration

        success_value = 1.0 if solution_success else 0.0
        if solution_success:
            state.consecutive_failures = 0
        else:
            state.consecutive_failures += 1
            state.failure_count_total += 1

        target = state.difficulty_target
        # Only adjust difficulty if we have sufficient data
        if state.total_attempts >= 5:
            if state.success_rate > self.SWEET_SPOT_MAX:
                # Increase difficulty gradually
                state.difficulty_target = min(0.95, target + 0.03)
                if state.status != "mastered" and state.success_rate >= 0.75:
                    state.status = "mastered"
                    state.mastered_at_iteration = self.current_iteration
            elif state.success_rate < self.SWEET_SPOT_MIN:
                # Decrease difficulty more conservatively to avoid getting stuck too low
                state.difficulty_target = max(0.2, target - 0.04)
                state.status = "active"
            else:
                state.status = "active"

        if measured_difficulty is not None:
            state.difficulty_history.append(float(measured_difficulty))
        else:
            state.difficulty_history.append(state.difficulty_target)

        state.history.append(
            {
                "iteration": float(self.current_iteration),
                "question_reward": float(question_reward),
                "solution_success": float(success_value),
                "success_rate": float(state.success_rate),
                "difficulty_target": float(state.difficulty_target),
            }
        )

        if combined_reward is not None:
            self.recent_combined_rewards.append(float(combined_reward))
            self.recent_combined_rewards = self.recent_combined_rewards[-30:]

        self.handle_persistent_failure(topic)

    def increment_iteration(self) -> None:
        self.current_iteration += 1
        # Reset within-iteration counters
        for state in self.topics.values():
            state.current_iteration_attempts = 0
        self._run_retention_tests_if_due()

    def generate_instruction(self, topic: str, target_difficulty: float) -> str:
        templates = self.TOPIC_TEMPLATES.get(topic, self.TOPIC_TEMPLATES["multi_step_reasoning"])
        template = random.choice(templates)
        # Note: {steps} placeholder removed from templates to let model decide complexity
        return template.format(
            context=random.choice(self.CONTEXTS),
            action=random.choice(self.ACTIONS),
        )

    def get_curriculum_stats(self) -> Dict[str, object]:
        return {
            "iteration": self.current_iteration,
            "topics": {topic: asdict(state) for topic, state in self.topics.items()},
            "sweet_spot_topics": self.get_sweet_spot_topics(),
            "current_focus_topics": self.get_current_focus(),
            "avg_recent_reward": float(np.mean(self.recent_combined_rewards)) if self.recent_combined_rewards else 0.0,
        }

    def get_sweet_spot_topics(self) -> List[str]:
        return [
            topic
            for topic, state in self.topics.items()
            if state.total_attempts > 0 and self.SWEET_SPOT_MIN <= state.success_rate <= self.SWEET_SPOT_MAX
        ]

    def get_current_focus(self) -> List[str]:
        return list(self.current_focus_topics)

    def save_state(self, iteration: int, rollout: Optional[int] = None) -> None:
        if rollout is not None and rollout % 10 != 0:
            return
        filename = (
            f"iteration_{iteration:03d}_final.json"
            if rollout is None
            else f"iteration_{iteration:03d}_rollout_{rollout:03d}.json"
        )
        path = self.checkpoint_dir / filename
        state = {
            "version": "1.0",
            "timestamp": datetime.utcnow().isoformat(),
            "iteration": iteration,
            "rollout": rollout,
            "current_iteration": self.current_iteration,
            "recent_combined_rewards": self.recent_combined_rewards,
            "topics": {topic: asdict(topic_state) for topic, topic_state in self.topics.items()},
            "hyperparams": self.hyperparams,
        }
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def load_checkpoint_safe(self) -> bool:
        checkpoints = sorted(self.checkpoint_dir.glob("iteration_*_final.json"), reverse=True)
        for checkpoint in checkpoints:
            try:
                data = json.loads(checkpoint.read_text(encoding="utf-8"))
                topics = data["topics"]
                if not isinstance(topics, dict):
                    raise ValueError("Invalid topics section")
                rebuilt = {}
                for topic in TOPIC_LIST:
                    values = topics.get(topic)
                    if values is None or "success_rate" not in values:
                        raise ValueError(f"Topic {topic} missing or malformed")
                    rebuilt[topic] = TopicState(**values)

                self.topics = rebuilt
                self.current_iteration = int(data.get("current_iteration", data.get("iteration", 0)))
                self.recent_combined_rewards = list(data.get("recent_combined_rewards", []))
                logger.info("Loaded curriculum state from %s", checkpoint)
                return True
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to load curriculum checkpoint %s: %s", checkpoint, exc)
        return False

    def _compute_topic_probabilities(self) -> Dict[str, float]:
        all_states = list(self.topics.values())
        sweet_spot = [t for t in all_states if self.SWEET_SPOT_MIN <= t.success_rate <= self.SWEET_SPOT_MAX and t.total_attempts > 0]
        untested = [t for t in all_states if t.total_attempts == 0]
        mastered = [t for t in all_states if t.status == "mastered"]

        if self.current_iteration <= 3:
            weights = {"sweet": 0.0, "explore": 1.0, "retention": 0.0}
        elif self.current_iteration <= 10:
            weights = {"sweet": 0.50, "explore": 0.35, "retention": 0.15}
        else:
            weights = {"sweet": 0.60, "explore": 0.25, "retention": 0.15}

        if self._detect_plateau():
            weights["explore"] = min(0.50, weights["explore"] + 0.2)
            weights["sweet"] = max(0.2, weights["sweet"] - 0.2)

        # Start with minimum allocation for ALL topics (5% split)
        MIN_ALLOCATION = 0.05
        probs: Dict[str, float] = {t.topic_name: MIN_ALLOCATION / len(all_states) for t in all_states}
        remaining_mass = 1.0 - MIN_ALLOCATION
        
        bonus_probs: Dict[str, float] = {}
        
        # Allocate sweet spot budget with within-iteration diversity penalty
        if sweet_spot:
            # Apply diversity penalty based on current iteration attempts
            staleness = {}
            for t in sweet_spot:
                # Strong penalty for topics sampled many times in current iteration
                if t.current_iteration_attempts == 0:
                    # Not yet sampled this iteration - highest priority
                    staleness[t.topic_name] = 10.0
                elif t.current_iteration_attempts <= 3:
                    # Sampled 1-3 times - moderate priority
                    staleness[t.topic_name] = 5.0 / t.current_iteration_attempts
                else:
                    # Sampled 4+ times - heavily penalized
                    staleness[t.topic_name] = 1.0 / (t.current_iteration_attempts ** 1.5)
            
            total_stale = sum(staleness.values())
            if total_stale > 0:
                for t in sweet_spot:
                    bonus_probs[t.topic_name] = bonus_probs.get(t.topic_name, 0.0) + (
                        remaining_mass * weights["sweet"] * (staleness[t.topic_name] / total_stale)
                    )

        # Allocate explore budget - ensure we always explore something
        explore_pool = untested if untested else self._get_diverse_exploration_pool(sweet_spot)
        if explore_pool:
            each = remaining_mass * weights["explore"] / len(explore_pool)
            for t in explore_pool:
                bonus_probs[t.topic_name] = bonus_probs.get(t.topic_name, 0.0) + each

        # Allocate retention budget
        retention_due = [t for t in mastered if self._schedule_retention_test(t) <= self.current_iteration]
        if retention_due:
            each = remaining_mass * weights["retention"] / len(retention_due)
            for t in retention_due:
                bonus_probs[t.topic_name] = bonus_probs.get(t.topic_name, 0.0) + each

        # Add bonus to base minimum allocation
        for topic, bonus in bonus_probs.items():
            probs[topic] = probs.get(topic, 0.0) + bonus
        
        # Normalize to ensure sum = 1.0
        total = sum(probs.values())
        if total <= 0:
            each = 1.0 / len(all_states)
            return {t.topic_name: each for t in all_states}
        
        normalized = {topic: value / total for topic, value in probs.items()}
        
        # Apply topic probability floor to prevent mode collapse
        MIN_TOPIC_PROB = 0.02  # Every topic gets at least 2% chance
        for topic in normalized:
            if normalized[topic] < MIN_TOPIC_PROB:
                normalized[topic] = MIN_TOPIC_PROB
        
        # Re-normalize after applying floor
        total = sum(normalized.values())
        normalized = {topic: value / total for topic, value in normalized.items()}
        
        # Log top 5 topics for debugging
        top_topics = sorted(normalized.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.debug(f"Topic probabilities: {top_topics}")
        
        return normalized

    def _get_boundary_topics(self) -> List[TopicState]:
        result = []
        for state in self.topics.values():
            if state.total_attempts == 0:
                continue
            near_low = abs(state.success_rate - self.SWEET_SPOT_MIN) <= 0.08
            near_high = abs(state.success_rate - self.SWEET_SPOT_MAX) <= 0.08
            if near_low or near_high:
                result.append(state)
        if not result:
            result = sorted(self.topics.values(), key=lambda t: abs(t.success_rate - self.TARGET_SUCCESS))[:4]
        return result
    
    def _get_diverse_exploration_pool(self, exclude_sweet_spot: List[TopicState]) -> List[TopicState]:
        """
        Get topics for exploration that are NOT in sweet spot.
        
        Prioritizes:
        1. Under-practiced topics (low attempt count)
        2. Topics with potential (success rate 0.2-0.4 or 0.7-0.9)
        3. Topics not recently attempted
        
        Args:
            exclude_sweet_spot: Topics already in sweet spot to exclude
        
        Returns:
            List of 3-5 topics for exploration
        """
        sweet_spot_names = {t.topic_name for t in exclude_sweet_spot}
        candidates = [t for t in self.topics.values() if t.topic_name not in sweet_spot_names]
        
        if not candidates:
            # Fallback if somehow all topics are in sweet spot
            return list(self.topics.values())[:3]
        
        # Score each candidate
        scored = []
        for state in candidates:
            # Factor 1: Under-practiced (inverse of attempts)
            attempt_score = 1.0 / (1.0 + state.total_attempts / 10.0)
            
            # Factor 2: Near sweet spot boundaries (could improve into sweet spot)
            if 0.2 <= state.success_rate < self.SWEET_SPOT_MIN:
                potential_score = 2.0  # Just below sweet spot - high potential
            elif self.SWEET_SPOT_MAX < state.success_rate <= 0.9:
                potential_score = 1.5  # Just above sweet spot - could be challenged more
            elif state.total_attempts == 0:
                potential_score = 3.0  # Untested - highest priority
            else:
                potential_score = 0.5  # Far from sweet spot
            
            # Factor 3: Staleness (not practiced recently)
            staleness_score = max(1, self.current_iteration - state.last_practiced) / 5.0
            
            # Combined score
            total_score = attempt_score + potential_score + staleness_score
            scored.append((state, total_score))
        
        # Return top 3-5 topics by score
        scored.sort(key=lambda x: x[1], reverse=True)
        num_explore = min(5, max(3, len(candidates) // 3))
        return [state for state, _ in scored[:num_explore]]

    def _get_difficulty_for_topic(self, topic: str) -> float:
        state = self.topics[topic]
        noise = random.uniform(-0.04, 0.04)
        return max(0.1, min(0.95, state.difficulty_target + noise))

    def _difficulty_to_step_range(self, difficulty: float) -> str:
        if difficulty < 0.3:
            return "1-2"
        if difficulty < 0.6:
            return "2-3"
        return "3-4"

    def _schedule_retention_test(self, state: TopicState) -> int:
        if state.mastered_at_iteration is None:
            return 10 ** 9
        interval = min(2 ** max(0, state.retention_tests_passed), 32)
        return state.mastered_at_iteration + interval

    def _run_retention_tests_if_due(self) -> None:
        for state in self.topics.values():
            if state.status != "mastered":
                continue
            if self._schedule_retention_test(state) <= self.current_iteration:
                # Retention test scheduling is represented by increasing pressure
                # during topic selection rather than explicit immediate update.
                logger.info("Topic %s is due for retention test", state.topic_name)

    def handle_retention_test_result(self, topic: str, success_rate: float) -> None:
        state = self.topics[topic]
        if success_rate >= 0.7:
            state.retention_tests_passed += 1
            state.last_retention_score = success_rate
            state.status = "mastered"
        elif success_rate >= 0.4:
            state.retention_tests_passed = 0
            state.last_retention_score = success_rate
            state.status = "active"
        else:
            state.retention_tests_passed = 0
            state.last_retention_score = success_rate
            state.status = "forgotten"
            state.difficulty_target = max(0.15, state.difficulty_target * 0.7)

    def handle_persistent_failure(self, topic: str) -> None:
        state = self.topics[topic]
        failures = state.consecutive_failures
        if failures >= 3:
            state.difficulty_target = max(0.1, state.difficulty_target * 0.6)
        if failures >= 5:
            state.status = "paused"
        if failures >= 10:
            hard_topics = [
                t
                for t in self.topics.values()
                if t.total_attempts >= 10 and t.success_rate < 0.3
            ]
            if len(hard_topics) >= 3:
                self._emergency_reset()

    def _emergency_reset(self) -> None:
        logger.warning("Emergency curriculum reset triggered")
        for state in self.topics.values():
            state.status = "active"
            state.difficulty_target = min(0.45, max(0.2, state.difficulty_target))
            state.consecutive_failures = 0

    def _detect_plateau(self) -> bool:
        if len(self.recent_combined_rewards) < 10:
            return False
        window = self.recent_combined_rewards[-10:]
        return float(np.std(window)) < 0.05
