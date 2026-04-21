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


class CurriculumManager:
    """Goldilocks curriculum with adaptive topic selection."""

    SWEET_SPOT_MIN = 0.4
    SWEET_SPOT_MAX = 0.7
    TARGET_SUCCESS = 0.55

    CONTEXTS = ["bakery", "shopping", "school", "sports", "gardening", "travel"]
    ACTIONS = ["uses", "sells", "shares", "loses", "earns", "mixes"]

    TOPIC_TEMPLATES = {
        "basic_arithmetic": [
            "Generate a {context} word problem using addition/subtraction with {steps} steps.",
        ],
        "single_step_word_problems": [
            "Generate a simple one-idea word problem in a {context} setting with {steps} steps.",
        ],
        "fractions": [
            "Generate a fractions word problem where someone {action} part of a quantity in a {context} scenario with {steps} steps.",
            "Create a problem involving fraction operations in {context} with {steps} steps.",
        ],
        "percentages": [
            "Generate a percentage change or discount problem in {context} with {steps} steps.",
        ],
        "ratios": [
            "Generate a ratios/proportions word problem in {context} with {steps} steps.",
        ],
        "money_problems": [
            "Create a money and pricing problem in {context} with {steps} reasoning steps.",
        ],
        "time_distance": [
            "Generate a time/speed/distance problem in {context} with {steps} steps.",
        ],
        "multi_step_reasoning": [
            "Generate a multi-step reasoning problem in {context} requiring {steps} steps.",
        ],
        "algebra": [
            "Generate an algebra problem that solves for a variable in {context} with {steps} steps.",
        ],
        "mixed_operations": [
            "Generate a problem requiring mixed operations in {context} with {steps} steps.",
        ],
        "comparison_problems": [
            "Generate a comparison problem in {context} ('more than'/'less than') with {steps} steps.",
        ],
        "optimization_problems": [
            "Generate a constrained optimization style word problem in {context} with {steps} steps.",
        ],
    }

    TOPIC_PREREQUISITES = {
        "fractions": ["basic_arithmetic"],
        "percentages": ["fractions"],
        "ratios": ["basic_arithmetic"],
        "algebra": ["basic_arithmetic", "comparison_problems"],
        "mixed_operations": ["basic_arithmetic", "fractions"],
        "optimization_problems": ["comparison_problems", "algebra"],
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

    def select_topic_and_difficulty(self) -> Tuple[str, float]:
        probs = self._compute_topic_probabilities()
        names = list(probs.keys())
        dist = np.array([probs[name] for name in names], dtype=np.float64)
        dist = dist / dist.sum()
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
        state = self.topics[topic]
        state.total_attempts += 1
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
        if state.success_rate > self.SWEET_SPOT_MAX:
            state.difficulty_target = min(0.95, target + 0.05)
            if state.status != "mastered":
                state.status = "mastered"
                state.mastered_at_iteration = self.current_iteration
        elif state.success_rate < self.SWEET_SPOT_MIN:
            state.difficulty_target = max(0.15, target - 0.07)
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
        self._run_retention_tests_if_due()

    def generate_instruction(self, topic: str, target_difficulty: float) -> str:
        templates = self.TOPIC_TEMPLATES.get(topic, self.TOPIC_TEMPLATES["multi_step_reasoning"])
        template = random.choice(templates)
        steps = self._difficulty_to_step_range(target_difficulty)
        return template.format(
            context=random.choice(self.CONTEXTS),
            action=random.choice(self.ACTIONS),
            steps=steps,
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
            weights = {"sweet": 0.60, "explore": 0.25, "retention": 0.15}
        else:
            weights = {"sweet": 0.70, "explore": 0.15, "retention": 0.15}

        if self._detect_plateau():
            weights["explore"] = min(0.45, weights["explore"] + 0.2)
            weights["sweet"] = max(0.2, weights["sweet"] - 0.2)

        probs: Dict[str, float] = {}
        if sweet_spot:
            staleness = {t.topic_name: max(1, self.current_iteration - t.last_practiced) for t in sweet_spot}
            total_stale = sum(staleness.values())
            for t in sweet_spot:
                probs[t.topic_name] = weights["sweet"] * (staleness[t.topic_name] / total_stale)

        explore_pool = untested if untested else self._get_boundary_topics()
        if explore_pool:
            each = weights["explore"] / len(explore_pool)
            for t in explore_pool:
                probs[t.topic_name] = probs.get(t.topic_name, 0.0) + each

        retention_due = [t for t in mastered if self._schedule_retention_test(t) <= self.current_iteration]
        if retention_due:
            each = weights["retention"] / len(retention_due)
            for t in retention_due:
                probs[t.topic_name] = probs.get(t.topic_name, 0.0) + each

        if not probs:
            each = 1.0 / len(all_states)
            return {t.topic_name: each for t in all_states}

        total = sum(probs.values())
        if total <= 0:
            each = 1.0 / len(all_states)
            return {t.topic_name: each for t in all_states}
        return {topic: value / total for topic, value in probs.items()}

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
