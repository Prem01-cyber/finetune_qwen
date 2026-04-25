"""
Curriculum-aware math environment with dual reward signals.

This file is deliberately minimal: a single ``collect_rollouts`` method is all
the training loop needs.  Rollouts and PPO updates run in the same process on
a single GPU — no subprocesses, no RPC, no vLLM colocation.
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from sympy import simplify
from sympy.parsing.sympy_parser import parse_expr
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config.prompts import create_solver_messages
from src.rl.curriculum_manager import CurriculumManager
from src.rl.expert_panel import SimulatedExpertPanel
from src.rl.mdp_components import Action, State, Trajectory, Transition
from src.rl.prm_scorer import ProcessRewardScorer
from src.rl.quality_filter import QualityFilter
from src.rl.question_quality_evaluator import QuestionQualityEvaluator
from src.rl.replay_buffer import GenerationalReplayBuffer
from src.rl.value_network import ValueHead
from src.sft.solution_format import extract_final_answer_numeric_str
from src.sft.sympy_normalize import normalize_for_parse_expr

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryMetadata:
    curriculum_iteration: int
    target_topic: str
    target_difficulty: float
    instruction: str
    generated_question: str
    generated_solution: str
    question_length: int
    solution_length: int
    detected_topic: str
    detected_secondary_topics: List[str]
    topic_match_score: float
    estimated_difficulty: float
    clarity_score: float
    novelty_scores: Dict[str, float]
    consensus_achieved: bool
    consensus_strength: float
    answer_diversity: int
    majority_answer: Optional[float]
    primary_matches_majority: bool
    sympy_verified: bool
    steps_total: int
    steps_verified_ok: int
    steps_failed: int
    final_answer_ok: bool
    question_reward: float
    solution_reward: float
    pre_expert_reward: float
    expert_reward_modifier: float
    expert_phase: str
    expert_feedback: str
    replay_candidate: bool
    replay_novelty: float
    replay_added: bool
    combined_reward: float
    reward_breakdown: Dict[str, object]
    topics_in_sweet_spot: List[str]
    current_focus_topics: List[str]
    curriculum_state_snapshot: Dict[str, object]


class CurriculumMathEnvironment:
    """Standalone curriculum environment with PRM-based rewards and GRPO training support."""

    def __init__(
        self,
        policy_model: AutoModelForCausalLM,
        value_model: Optional[ValueHead],
        tokenizer: AutoTokenizer,
        reference_questions: Optional[List[str]] = None,
        grounded_qa_pairs: Optional[List[Dict[str, str]]] = None,
        prm_scorer: Optional[ProcessRewardScorer] = None,
        curriculum_checkpoint_dir: str = "checkpoints/curriculum",
        max_question_tokens: int = 200,
        max_solution_tokens: int = 500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        consensus_temperature: float = 0.7,
        device: Optional[torch.device] = None,
        unified_accuracy_calc: Optional[Any] = None,
    ):
        # ── Core model attributes (used by generation helpers) ───────────
        self.policy = policy_model
        self.value = value_model
        self.tokenizer = tokenizer
        self.max_question_tokens = max_question_tokens
        self.max_solution_tokens = max_solution_tokens
        self.temperature = temperature
        self.top_p = top_p

        if device is not None:
            self.device = torch.device(device)
        else:
            try:
                self.device = next(policy_model.parameters()).device
            except StopIteration:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.reference_questions = reference_questions or []
        self.grounded_qa_pairs: List[Dict[str, str]] = [
            qa for qa in (grounded_qa_pairs or [])
            if qa.get("question") and qa.get("gold_final")
        ]
        self.consensus_temperature = consensus_temperature
        self.curriculum_manager = CurriculumManager(checkpoint_dir=curriculum_checkpoint_dir)
        self.curriculum_manager.initialize(bootstrap_questions=self.reference_questions)
        self.curriculum_manager.load_checkpoint_safe()
        self.question_evaluator = QuestionQualityEvaluator(
            reference_questions=self.reference_questions
        )
        # PRM is the sole process-quality signal.  Passing prm_scorer=None
        # will cause compute_reward/compute_grounded_reward to raise at
        # call time — GRPO training always supplies the PRM.
        self.prm_scorer = prm_scorer
        # Unified accuracy calculator — activated on Phase 2+ transition.
        # When use_chain_scoring is True, chain_integrity_score from this
        # calculator replaces PRM-based process_score in both grounded and
        # self-play reward paths.
        self.unified_accuracy_calc: Optional[Any] = unified_accuracy_calc
        self.use_chain_scoring: bool = False
        self.expert_panel = SimulatedExpertPanel()
        self.replay_buffer = GenerationalReplayBuffer(max_size=500)
        self.quality_filter = QualityFilter(novelty_threshold=0.5)
        self.last_replay_ratio: float = 0.0
        self.last_rollout_mix: Dict[str, int] = {
            "fresh": 0,
            "replay": 0,
            "grounded": 0,
        }
        # Running counts for the most recent grounded batch, so the training
        # script can log grounded accuracy per iteration without re-parsing
        # trajectory metadata.
        self.last_grounded_stats: Dict[str, float] = {
            "count": 0,
            "correct": 0,
            "accuracy": 0.0,
            "mean_reward": 0.0,
        }

    def sample_instruction(self) -> Tuple[str, str, float]:
        topic, difficulty = self.curriculum_manager.select_topic_and_difficulty()
        instruction = self.curriculum_manager.generate_instruction(
            topic=topic, target_difficulty=difficulty
        )
        return instruction, topic, difficulty

    def format_solution_prompt(self, question: str) -> str:
        """Format a question into a chat-templated solver prompt."""
        messages = create_solver_messages(question)
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate_with_logging(
        self,
        initial_prompt: str,
        max_tokens: int,
        phase: str,
    ) -> Tuple[str, List[Transition]]:
        """
        Generate text with per-step PPO-grade transition logging.

        Used by the PPO-compatible rollout methods (``collect_rollouts``,
        ``rollout_trajectory``, ``rollout_grounded_trajectory``).  The GRPO
        training loop uses ``generate_solutions_batched`` instead.
        """
        import torch.nn.functional as F  # local import to keep top-level clean

        prompt_ids = self.tokenizer.encode(
            initial_prompt, return_tensors="pt"
        ).to(self.device)
        prompt_length = prompt_ids.shape[1]
        prompt_attn = torch.ones_like(prompt_ids)

        temperature = float(self.temperature)
        do_sample = temperature > 1e-4
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id or eos_id

        gen_kwargs: Dict[str, Any] = dict(
            input_ids=prompt_ids,
            attention_mask=prompt_attn,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            use_cache=True,
            output_logits=True,
            return_dict_in_generate=True,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = max(temperature, 1e-6)
            gen_kwargs["top_p"] = float(self.top_p)

        with torch.no_grad():
            gen_out = self.policy.generate(**gen_kwargs)

        full_ids = gen_out.sequences  # [1, P + T]
        T_gen = int(full_ids.shape[1] - prompt_length)
        if T_gen <= 0:
            return "", []

        raw_logits = torch.stack([lg[0] for lg in gen_out.logits], dim=0).float()
        raw_log_probs = F.log_softmax(raw_logits, dim=-1)
        sampled_tokens = full_ids[0, prompt_length:]
        chosen_log_probs = raw_log_probs.gather(
            1, sampled_tokens.unsqueeze(1)
        ).squeeze(1)
        entropies = -(raw_log_probs.exp() * raw_log_probs).sum(dim=-1)

        positions = torch.arange(
            prompt_length - 1, prompt_length + T_gen - 1, device=self.device
        )
        full_attn = torch.ones_like(full_ids)
        if self.value is not None:
            values = self.value.values_at_positions(
                input_ids=full_ids, positions=positions, attention_mask=full_attn
            )
        else:
            values = torch.zeros(T_gen, device=self.device)

        piece_by_piece: List[str] = self.tokenizer.batch_decode(
            [[tok.item()] for tok in sampled_tokens], skip_special_tokens=False
        )

        transitions: List[Transition] = []
        running_text = initial_prompt
        for t in range(T_gen):
            state_input_ids = full_ids[0, : prompt_length + t]
            current_state = State(
                text=running_text,
                input_ids=state_input_ids,
                attention_mask=torch.ones_like(state_input_ids),
                phase=phase,
            )
            action_token = int(sampled_tokens[t].item())
            action = Action(
                token_id=action_token,
                log_prob=float(chosen_log_probs[t].item()),
                entropy=float(entropies[t].item()),
            )
            next_text = running_text + piece_by_piece[t]
            next_input_ids = full_ids[0, : prompt_length + t + 1]
            next_state = State(
                text=next_text,
                input_ids=next_input_ids,
                attention_mask=torch.ones_like(next_input_ids),
                phase=phase,
            )
            is_done = eos_id is not None and action_token == eos_id
            transitions.append(
                Transition(
                    state=current_state,
                    action=action,
                    reward=0.0,
                    next_state=next_state,
                    value=float(values[t].item()),
                    done=is_done,
                )
            )
            running_text = next_text
            if is_done:
                break

        generated_ids = full_ids[0, prompt_length : prompt_length + len(transitions)]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return generated_text, transitions

    def _compute_format_score(self, solution: str) -> float:
        """
        Structural format score based purely on text patterns — no SymPy.

        Checks:
          - Presence of 'Step N:' lines (multi-step structure)
          - Presence of 'Final Answer:' line (correct termination)
          - Length: ≥2 step lines scores highest

        Returns a score in [0, 1].
        """
        lines = solution.splitlines()
        step_lines  = [l for l in lines if re.match(r"^\s*Step\s+\d+\s*:", l)]
        has_final   = any(re.match(r"^\s*Final Answer\s*:", l, re.IGNORECASE) for l in lines)

        n_steps = len(step_lines)
        if n_steps >= 2:
            length_bonus = 1.0
        elif n_steps == 1:
            length_bonus = 0.5
        else:
            length_bonus = 0.0

        final_ok = 1.0 if has_final else 0.0
        # 0.7 × step-structure + 0.3 × final-answer presence
        return max(0.0, min(1.0, 0.7 * length_bonus + 0.3 * final_ok))

    def compute_reward(
        self,
        question: str,
        solution: str,
        target_topic: str,
        target_difficulty: float,
    ) -> Dict[str, object]:
        # With a PRM scorer plugged in we skip the expensive (and noisy)
        # TripleVerifier consensus step.  PRM gives per-step correctness
        # against the actual question semantics, which is strictly better
        # than "do 3 independent samples agree?"
        if self.prm_scorer is not None:
            return self._compute_reward_with_prm(
                question=question,
                solution=solution,
                target_topic=target_topic,
                target_difficulty=target_difficulty,
            )

        raise RuntimeError(
            "compute_reward called without a PRM scorer. "
            "CurriculumMathEnvironment requires prm_scorer to be set. "
            "Pass prm_scorer=ProcessRewardScorer(...) at construction time."
        )

    def _compute_reward_with_prm(
        self,
        question: str,
        solution: str,
        target_topic: str,
        target_difficulty: float,
    ) -> Dict[str, object]:
        """
        Self-play reward using Qwen2.5-Math-PRM as the semantic-correctness
        signal.  PRM gives per-step probabilities that each reasoning step
        is correct *given the question* — exactly the signal consensus
        voting was supposed to approximate but couldn't (three samples
        from the same policy agree on wrong answers).

        Solution reward (PRM path):
            R_sol = 0.45·prm_final + 0.35·prm_mean + 0.20·lccp
            R     = 0.4·R_q + 0.6·R_sol      (then expert-panel modifier)

        * ``prm_final`` (final step score) is the strongest predictor of
          overall answer correctness.
        * ``prm_mean`` provides a smooth gradient over all steps.
        * ``lccp`` (Longest Correct Consecutive Prefix) rewards chain
          integrity — consecutive correct steps before the first failure.
        * The 0.4/0.6 Q/Sol split boosts gradient to question-generation
          without starving the solution-correctness signal.
        """
        assert self.prm_scorer is not None, "caller must check self.prm_scorer"

        prm_result = self.prm_scorer.score_solution(
            question=question, solution=solution
        )
        format_score = self._compute_format_score(solution)

        prm_mean = float(prm_result.get("mean_score", 0.0))
        prm_min = float(prm_result.get("min_score", 0.0))
        prm_final = float(prm_result.get("final_score", 0.0))
        prm_num_steps = int(prm_result.get("num_steps", 0))
        prm_degraded = bool(prm_result.get("degraded", False))

        # If the PRM degraded (empty solution, tokeniser mismatch, truncation),
        # the output is effectively unparseable.  Prior behavior was to fall
        # back on SymPy+format, but the upstream ``base_combined_score`` also
        # blends in the question reward — so the policy got a positive signal
        # for producing a broken solution as long as the *question* looked
        # fine.  We now treat a degraded PRM as a hard zero on the solution
        # reward; the question reward is gated below so the full combined
        # score also collapses.
        if prm_degraded or prm_num_steps == 0:
            solution_reward = 0.0
            _sp_lccp = 0.0
            sol_valid = False
            _sp_chain_integrity: Optional[float] = None
            logger.info(
                "PRM degraded (%s); sol_reward set to 0.0 (format=%.2f).",
                prm_result.get("degraded_reason", "unknown"),
                format_score,
            )
        else:
            # LCCP for self-play: same chain-integrity measure as grounded path
            _sp_step_scores = prm_result.get("step_scores", []) or []
            if _sp_step_scores:
                _first_fail = next(
                    (i for i, s in enumerate(_sp_step_scores) if s <= 0.5),
                    len(_sp_step_scores),
                )
                _sp_lccp = _first_fail / len(_sp_step_scores)
            else:
                _sp_lccp = 0.0

            # Self-play solution: PRM-only reward blending mean, final & chain integrity.
            # LCCP anchors the grade to *consecutive* correctness, not just bag-of-steps.
            solution_reward = (
                0.45 * prm_final
                + 0.35 * prm_mean
                + 0.20 * _sp_lccp
            )
            # Phase 2+ chain scoring: replace PRM solution blend with unified
            # chain integrity + dependency consistency.  This also populates the
            # question_score from the unified calculator so the Q/Sol weighting
            # below uses chain-verified signals instead of PRM proxies.
            _sp_chain_integrity = None
            if self.use_chain_scoring and self.unified_accuracy_calc is not None:
                try:
                    _sp_report = self.unified_accuracy_calc.compute(
                        solution=solution,
                        gold_answer=None,
                        question=question,
                        topic=target_topic,
                        phase="selfplay",
                    )
                    solution_reward = _sp_report.composite_accuracy
                    _sp_chain_integrity = _sp_report.chain_integrity_score
                except Exception as _sp_exc:
                    logger.debug("Unified accuracy calc (self-play) failed: %s", _sp_exc)
            sol_valid = True
        solution_reward = max(0.0, min(1.0, solution_reward))

        question_result = self.question_evaluator.evaluate(
            question=question,
            solution=solution,
            # Synthesize a "consensus-equivalent" dict so the question
            # evaluator keeps working unchanged.  PRM mean score stands
            # in for consensus strength since both are correctness proxies.
            consensus_result={
                "has_majority": prm_mean >= 0.5,
                "consensus_strength": prm_mean,
                "primary_matches_majority": prm_mean >= 0.5,
                "answer_diversity": 0,
                "majority_answer": None,
                "primary_answer": None,
            },
            target_topic=target_topic,
            target_difficulty=target_difficulty,
        )
        question_reward = float(question_result["overall_score"])

        # Gate the question-quality bonus on having a parseable solution.
        # A great-looking question with a broken solution is not progress
        # toward self-improvement — it's the policy gaming whichever
        # signal is easier to produce.
        effective_question_reward = question_reward if sol_valid else 0.0

        # Q/Sol = 0.4/0.6 — see note in compute_reward (non-PRM path).
        base_combined_score = (
            0.4 * effective_question_reward + 0.6 * solution_reward
        )

        # Format floor: if the solution structure is broken (<0.5 format),
        # cap the overall reward at 0.3 regardless of how much the PRM
        # likes the prose.  Previously we saw combined=0.83 with
        # Format=0.30, i.e. the PRM "approved" an output that didn't have
        # parseable Step/Final Answer lines — pure reward hacking.
        format_floor_active = format_score < 0.5
        format_cap = 0.3 if format_floor_active else 1.0
        base_combined_score = min(base_combined_score, format_cap)

        # Novelty gate: prevent template-copying reward hacking.
        # If the model just generates "John has X apples..." with different numbers,
        # n-gram similarity to the reference corpus is high → dataset_novelty is LOW.
        # We cap the reward to discourage this without penalising genuinely novel questions.
        #   < 0.20: near-copy of a training question (template + new variables) → cap 0.35
        #   > 0.85: completely off-domain (not a real math problem style)       → cap 0.55
        #   [0.20, 0.85]: Goldilocks zone → full reward (novelty_cap = 1.0)
        _dataset_novelty = float(
            question_result.get("novelty", {}).get("dataset_novelty", 0.5)
            if isinstance(question_result.get("novelty"), dict)
            else 0.5
        )
        if _dataset_novelty < 0.20:
            _novelty_cap = 0.35
        elif _dataset_novelty > 0.85:
            _novelty_cap = 0.55
        else:
            _novelty_cap = 1.0
        if _novelty_cap < 1.0:
            base_combined_score = min(base_combined_score, _novelty_cap)
            logger.debug(
                "Novelty gate: dataset_novelty=%.2f → cap=%.2f (was %.3f → now %.3f)",
                _dataset_novelty, _novelty_cap,
                base_combined_score / _novelty_cap if _novelty_cap > 0 else 0,
                base_combined_score,
            )

        expert_adjustment = self.expert_panel.apply_expert_preferences(
            base_reward=base_combined_score,
            question_metrics=question_result,
            solution_metrics={
                # Only format_compliance still influences shaping — the
                # PRM/correctness signal lives inside ``solution_reward``
                # already and must not be double-counted here.
                "format_compliance": format_score,
            },
            iteration=self.curriculum_manager.current_iteration,
        )
        combined_score = float(expert_adjustment["adjusted_reward"])
        # Re-clip after additive shaping + respect the format cap one more
        # time so the shaping can't lift a badly-formatted solution back
        # above the cap.
        combined_score = max(0.0, min(format_cap, combined_score))

        # Curriculum mastery: consider self-play solution "successful" when
        # both the chain mean AND the final concluding step are above threshold.
        # Using prm_final as a required condition prevents a solution that gets
        # most steps right but fails the conclusion from being marked "mastered".
        solution_success = (
            (not prm_degraded)
            and (prm_mean >= 0.65)
            and (prm_final >= 0.50)
        )
        self.curriculum_manager.update_from_trajectory(
            topic=target_topic,
            question_reward=question_reward,
            solution_success=solution_success,
            combined_reward=combined_score,
            measured_difficulty=float(question_result["measured_difficulty"]),
        )

        modifier_val = float(expert_adjustment.get("reward_modifier", 0.0))
        floor_tag = " FLOOR" if format_floor_active else ""
        valid_tag = "" if sol_valid else " [SOL_INVALID]"
        logger.info(
            "PRM reward%s: combined=%.3f = clip(base=%.3f + mod=%+.3f, cap=%.2f)%s "
            "| Q=%.2f sol=%.3f novelty=%.2f | "
            "sol=0.45*prm_final(%.2f)+0.35*prm_mean(%.2f)+0.20*lccp(%.2f) "
            "| steps=%d",
            valid_tag,
            combined_score,
            base_combined_score,
            modifier_val,
            format_cap,
            floor_tag,
            effective_question_reward,
            solution_reward,
            _dataset_novelty,
            prm_final,
            prm_mean,
            _sp_lccp if sol_valid else 0.0,
            prm_num_steps,
        )

        # Shape a consensus-style verification_details dict so downstream
        # aggregation (which reads these keys) keeps working unchanged.
        verification_details = {
            "consensus": {
                "has_majority": prm_mean >= 0.5,
                "consensus_strength": prm_mean,
                "primary_matches_majority": prm_mean >= 0.5,
                "answer_diversity": 0,
                "majority_answer": None,
                "primary_answer": extract_final_answer_numeric_str(solution) or None,
                "prm_mean_score": prm_mean,
                "prm_min_score": prm_min,
                "prm_final_score": prm_final,
                "prm_step_scores": prm_result.get("step_scores", []),
                "prm_num_steps": prm_num_steps,
                "prm_degraded": prm_degraded,
            },
        }

        return {
            "combined_score": combined_score,
            "base_combined_score": base_combined_score,
            "effective_question_reward": effective_question_reward,  # gated (0 when sol invalid)
            "question_metrics": question_result,
            "solution_metrics": {
                "overall_score": solution_reward,
                "correctness": prm_mean,
                "format_compliance": format_score,
                "efficiency": prm_mean,          # legacy slot
                "consensus_score": prm_mean,     # legacy slot
                "prm_mean_score": prm_mean,
                "prm_min_score": prm_min,
                "prm_final_score": prm_final,
                "prm_step_scores": prm_result.get("step_scores", []),
                "prm_num_steps": prm_num_steps,
                "prm_degraded": prm_degraded,
                "verification_details": verification_details,
            },
            "curriculum_metrics": {
                "target_topic": target_topic,
                "target_difficulty": target_difficulty,
                "detected_topic": question_result["detected_topic"],
                "measured_difficulty": question_result["measured_difficulty"],
            },
            "expert_metrics": expert_adjustment,
            # Chain scoring metrics (Phase 2+; None when use_chain_scoring=False)
            "sp_chain_integrity_score": _sp_chain_integrity,
        }

    # ------------------------------------------------------------------
    # Grounded (GSM8K-anchored) rollouts
    # ------------------------------------------------------------------
    #
    # Why this exists: self-play rewards are dominated by consensus voting
    # between 3 same-model samples, which correlates poorly with GSM8K
    # accuracy (all three samples can be wrong in the same way).  For the
    # grounded path we solve a known GSM8K problem and score the solution
    # directly against the gold final answer, which is the only signal
    # guaranteed to move the benchmark we actually evaluate on.
    #
    # The reward:  R = 0.50·gt_match + 0.40·process(PRM) + 0.10·format
    #
    #   * gt_match = 1.0 iff the model's Final Answer is mathematically
    #     equivalent to the GSM8K gold final (via sympy.simplify on the
    #     extracted numeric string).
    #   * process = 0.60·prm_final + 0.40·prm_mean (PRM step-level quality)
    #   * format rewards Step N: lines and a Final Answer: line.
    #
    # No TripleVerifier call on this path — ground truth obviates consensus.

    @staticmethod
    def _norm_expr_for_match(s: str) -> str:
        s = (s or "").strip()
        s = s.replace("^", "**")
        s = re.sub(r"[,$€£\s]+", "", s)
        return s

    @classmethod
    def _answers_equivalent(cls, pred: str, gold: str) -> bool:
        """Return True iff ``pred`` and ``gold`` parse to the same number."""
        if not pred or not gold:
            return False
        p = cls._norm_expr_for_match(pred)
        g = cls._norm_expr_for_match(gold)
        if p == g:
            return True
        try:
            diff = simplify(
                parse_expr(normalize_for_parse_expr(p))
                - parse_expr(normalize_for_parse_expr(g))
            )
            return bool(diff == 0)
        except Exception:
            return False

    def compute_grounded_reward(
        self,
        question: str,
        solution: str,
        gold_final: str,
    ) -> Dict[str, object]:
        """
        Compute a ground-truth-anchored reward for a solution to a known
        GSM8K problem.  No TripleVerifier call — the gold final answer
        replaces consensus voting as the semantic check.
        """
        format_score = self._compute_format_score(solution)

        pred_final = extract_final_answer_numeric_str(solution) or ""
        gt_match_bool = self._answers_equivalent(pred_final, gold_final)
        if gt_match_bool:
            gt_match = 1.0
        else:
            # Soft numeric proximity: reward near-misses rather than cliffing at 0.
            # Gives partial credit proportional to how close the numeric answer is.
            # Capped at 0.85 so an exact match (1.0) is always strictly better.
            # Non-numeric wrong answers still get 0.0.
            try:
                _p = float(pred_final.replace(",", "").strip())
                _g = float(gold_final.replace(",", "").strip())
                _denom = max(abs(_g), 1.0)
                gt_match = min(0.85, 1.0 / (1.0 + 2.0 * abs(_p - _g) / _denom))
            except (ValueError, TypeError, AttributeError):
                gt_match = 0.0

        # Optional PRM step-level quality on grounded rollouts.
        # prm_final (last step score) is the strongest single predictor of
        # answer correctness. step_accuracy = fraction of steps the PRM
        # considers correct — the direct measure of reasoning process quality.
        prm_mean   = 0.0
        prm_final  = 0.0
        prm_step_scores: List[float] = []
        prm_num_steps = 0
        prm_degraded = True
        if self.prm_scorer is not None:
            prm_result = self.prm_scorer.score_solution(
                question=question, solution=solution
            )
            prm_degraded = bool(prm_result.get("degraded", False))
            if not prm_degraded:
                prm_mean        = float(prm_result.get("mean_score",   0.0))
                prm_final       = float(prm_result.get("final_score",  0.0))
                prm_step_scores = list(prm_result.get("step_scores",   []))
                prm_num_steps   = int(prm_result.get("num_steps",      0))

        # Step accuracy: fraction of individual steps rated correct by PRM.
        step_accuracy = (
            sum(1.0 for s in prm_step_scores if s > 0.5) / len(prm_step_scores)
            if prm_step_scores else 0.0
        )

        # Longest Correct Consecutive Prefix (LCCP): fraction of steps from
        # the start that are ALL rated correct before the first failure.
        # This captures chain integrity — a broken step 3 makes steps 4+ invalid
        # regardless of their individual PRM scores.
        # LCCP=1.0 means every step was correct (necessary condition for right answer).
        # LCCP=0.0 means step 1 itself was wrong (model never had a valid chain).
        if prm_step_scores:
            first_fail = next(
                (i for i, s in enumerate(prm_step_scores) if s <= 0.5), len(prm_step_scores)
            )
            lccp = first_fail / len(prm_step_scores)
        else:
            lccp = 0.0

        if self.prm_scorer is not None and not prm_degraded:
            # process_score: weight prm_final (conclusion step) more than mean
            # — the final step is the most critical and most predictive.
            process_score = 0.60 * prm_final + 0.40 * prm_mean
            combined = (
                0.50 * gt_match
                + 0.40 * process_score
                + 0.10 * format_score
            )
            _gt_tag = "exact" if gt_match_bool else f"prox={gt_match:.2f}"
            components_str = (
                f"0.50×{gt_match:.2f}({_gt_tag}) + 0.40×proc({process_score:.3f}"
                f"[fin={prm_final:.2f},mean={prm_mean:.2f}]) + "
                f"0.10×fmt({format_score:.3f})"
            )
        else:
            combined = 0.85 * gt_match + 0.15 * format_score
            components_str = (
                f"0.85×{gt_match:.2f} + 0.15×fmt({format_score:.3f})"
            )

        # Phase 2+ chain scoring: override process_score, step_accuracy, lccp,
        # and combined with formally-verified chain integrity metrics.
        # PRM is still called above so its scores remain logged for comparison.
        _chain_report = None
        if self.use_chain_scoring and self.unified_accuracy_calc is not None:
            try:
                _chain_report = self.unified_accuracy_calc.compute(
                    solution=solution,
                    gold_answer=gold_final,
                    topic="grounded",
                    phase="grounded",
                )
                process_score = _chain_report.chain_integrity_score
                step_accuracy = _chain_report.step_arithmetic_score
                lccp = _chain_report.lccp_score
                combined = max(0.0, min(1.0,
                    0.50 * gt_match + 0.30 * process_score + 0.20 * lccp
                ))
                components_str = (
                    f"0.50×{gt_match:.2f} + 0.30×chain({process_score:.3f}"
                    f"[arith={_chain_report.step_arithmetic_score:.2f},"
                    f"dep={_chain_report.step_dependency_score:.2f}]) + "
                    f"0.20×lccp({lccp:.3f})"
                )
            except Exception as _chain_exc:
                logger.debug("Unified accuracy calc failed, keeping PRM scores: %s", _chain_exc)
        else:
            combined = max(0.0, min(1.0, combined))

        # Hard negative mining: wrong-answer solutions still get a partial signal
        # proportional to how far they got before the first error (LCCP).
        # This prevents gradient starvation on hard problems where no solution in
        # the group is fully correct — the model still learns "longer correct prefix
        # is better" rather than receiving zero reward for all K samples.
        if gt_match < 0.5 and lccp > 0.0 and self.prm_scorer is not None:
            # Bonus = 0.15 × LCCP, capped so that a wrong answer (combined ≈ 0.40)
            # can never exceed 0.55 — always well below a correct answer (≈ 0.90+).
            _hnm_bonus = 0.15 * lccp
            combined = min(combined + _hnm_bonus, 0.55)

        _chain_depth = first_fail if prm_step_scores else 0
        logger.info(
            "Grounded reward: combined=%.3f = %s | pred=%r gold=%r | "
            "step_acc=%.0f%% lccp=%.0f%% (chain=%d/%d ok_count=%d) n_steps=%d",
            combined,
            components_str,
            pred_final,
            gold_final,
            100 * step_accuracy,
            100 * lccp,
            _chain_depth,
            len(prm_step_scores),
            sum(1 for s in prm_step_scores if s > 0.5),
            prm_num_steps,
        )

        return {
            "combined_score":    combined,
            "gt_match":          gt_match_bool,
            # process metrics
            "step_accuracy":     step_accuracy,
            "lccp":              lccp,        # longest correct consecutive prefix ratio
            "prm_mean_score":    prm_mean,
            "prm_final_score":   prm_final,
            "prm_step_scores":   prm_step_scores,
            "prm_num_steps":     prm_num_steps,
            "prm_degraded":      prm_degraded,
            # format / answer
            "format_score":      format_score,
            "pred_final":        pred_final,
            "gold_final":        gold_final,
            # chain scoring metrics (populated in Phase 2+, None otherwise)
            "chain_arith_score":     _chain_report.step_arithmetic_score if _chain_report else None,
            "chain_dep_score":       _chain_report.step_dependency_score if _chain_report else None,
            "chain_integrity_score": _chain_report.chain_integrity_score if _chain_report else None,
            "first_failure_step":    _chain_report.first_failure_step    if _chain_report else None,
            "final_consistent":      _chain_report.final_answer_consistent if _chain_report else None,
        }

    def rollout_grounded_trajectory(self, qa_pair: Dict[str, str]) -> Trajectory:
        """
        Run a rollout on a known GSM8K (question, gold_final) pair.

        The policy generates a solution to the real question; reward is
        dominated by whether the model's final number matches the gold
        final (ground-truth-anchored).
        """
        question = str(qa_pair["question"]).strip()
        gold_final = str(qa_pair["gold_final"]).strip()

        solution_prompt = self.format_solution_prompt(question)
        generated_solution, solution_transitions = self.generate_with_logging(
            initial_prompt=solution_prompt,
            max_tokens=self.max_solution_tokens,
            phase="grounded_solution",
        )

        reward_result = self.compute_grounded_reward(
            question=question,
            solution=generated_solution,
            gold_final=gold_final,
        )

        terminal_reward = float(reward_result["combined_score"])
        trajectory = Trajectory()
        for idx, transition in enumerate(solution_transitions):
            transition.reward = (
                terminal_reward if idx == len(solution_transitions) - 1 else 0.0
            )
            trajectory.add(transition)

        metadata = {
            "rollout_source": "grounded",
            "curriculum_iteration": self.curriculum_manager.current_iteration,
            "target_topic": "grounded_gsm8k",
            "target_difficulty": 0.5,
            "instruction": "",
            "generated_question": question,
            "generated_solution": generated_solution,
            "question_length": 0,
            "solution_length": len(solution_transitions),
            "detected_topic": "grounded_gsm8k",
            "detected_secondary_topics": [],
            "topic_match_score": 1.0,
            "estimated_difficulty": 0.5,
            "clarity_score": 1.0,
            "novelty_scores": {"combined": 0.0},
            "consensus_achieved": bool(reward_result["gt_match"]),
            "consensus_strength": 1.0 if reward_result["gt_match"] else 0.0,
            "answer_diversity": 0,
            "majority_answer": None,
            "primary_matches_majority": bool(reward_result["gt_match"]),
            "question_reward": 0.0,
            "solution_reward": terminal_reward,
            "pre_expert_reward": terminal_reward,
            "expert_reward_modifier": 0.0,
            "expert_phase": "grounded",
            "expert_feedback": "ground-truth anchored",
            "replay_candidate": False,
            "replay_novelty": 0.0,
            "replay_added": False,
            "combined_reward": terminal_reward,
            "reward_breakdown": {
                "grounded": True,
                "gt_match": bool(reward_result["gt_match"]),
                "format_score": float(reward_result["format_score"]),
                "pred_final": reward_result["pred_final"],
                "gold_final": reward_result["gold_final"],
                "prm_mean_score": float(reward_result.get("prm_mean_score", 0.0)),
                "prm_num_steps": int(reward_result.get("prm_num_steps", 0)),
                "prm_step_scores": list(reward_result.get("prm_step_scores", [])),
                "prm_degraded": bool(reward_result.get("prm_degraded", True)),
            },
            "topics_in_sweet_spot": self.curriculum_manager.get_sweet_spot_topics(),
            "current_focus_topics": self.curriculum_manager.get_current_focus(),
            "curriculum_state_snapshot": self.curriculum_manager.get_curriculum_stats(),
            "grounded_gt_match": bool(reward_result["gt_match"]),
            "grounded_pred_final": reward_result["pred_final"],
            "grounded_gold_final": reward_result["gold_final"],
        }
        trajectory.metadata = metadata
        return trajectory

    def rollout_trajectory(self) -> Trajectory:
        instruction, target_topic, target_difficulty = self.sample_instruction()
        question_prompt = self.format_question_generation_prompt(instruction)
        generated_question, question_transitions = self.generate_with_logging(
            initial_prompt=question_prompt,
            max_tokens=self.max_question_tokens,
            phase="question_generation",
        )
        return self._build_trajectory_from_question(
            instruction=instruction,
            target_topic=target_topic,
            target_difficulty=target_difficulty,
            generated_question=generated_question,
            question_transitions=question_transitions,
        )

    def _build_trajectory_from_question(
        self,
        instruction: str,
        target_topic: str,
        target_difficulty: float,
        generated_question: str,
        question_transitions: Optional[List] = None,
    ) -> Trajectory:
        trajectory = Trajectory()
        question_transitions = question_transitions or []

        solution_prompt = self.format_solution_prompt(generated_question)
        generated_solution, solution_transitions = self.generate_with_logging(
            initial_prompt=solution_prompt,
            max_tokens=self.max_solution_tokens,
            phase="solution",
        )

        reward_result = self.compute_reward(
            question=generated_question,
            solution=generated_solution,
            target_topic=target_topic,
            target_difficulty=target_difficulty,
        )

        terminal_reward = float(reward_result["combined_score"])
        all_transitions = question_transitions + solution_transitions
        # Terminal-only reward — gae_lambda=1.0 makes A_t = R - V(s_t) for all t.
        for idx, transition in enumerate(all_transitions):
            transition.reward = (
                terminal_reward if idx == len(all_transitions) - 1 else 0.0
            )
            trajectory.add(transition)

        verification = reward_result["solution_metrics"]["verification_details"]
        consensus = verification["consensus"]
        question_metrics = reward_result["question_metrics"]

        metadata = TrajectoryMetadata(
            curriculum_iteration=self.curriculum_manager.current_iteration,
            target_topic=target_topic,
            target_difficulty=target_difficulty,
            instruction=instruction,
            generated_question=generated_question,
            generated_solution=generated_solution,
            question_length=len(question_transitions),
            solution_length=len(solution_transitions),
            detected_topic=str(question_metrics["detected_topic"]["primary_topic"]),
            detected_secondary_topics=[
                str(x) for x in question_metrics["detected_topic"]["secondary_topics"]
            ],
            topic_match_score=float(question_metrics["topic_match"]),
            estimated_difficulty=float(question_metrics["measured_difficulty"]),
            clarity_score=float(question_metrics["clarity"]),
            novelty_scores=dict(question_metrics["novelty"]),
            consensus_achieved=bool(consensus["has_majority"]),
            consensus_strength=float(consensus["consensus_strength"]),
            answer_diversity=int(consensus["answer_diversity"]),
            majority_answer=consensus.get("majority_answer"),
            primary_matches_majority=bool(consensus["primary_matches_majority"]),
            sympy_verified=True,
            steps_total=int(consensus.get("prm_num_steps", 0)),
            steps_verified_ok=int(consensus.get("prm_num_steps", 0)),
            steps_failed=0,
            final_answer_ok=bool(consensus.get("primary_matches_majority", False)),
            question_reward=float(question_metrics["overall_score"]),
            solution_reward=float(reward_result["solution_metrics"]["overall_score"]),
            pre_expert_reward=float(reward_result["base_combined_score"]),
            expert_reward_modifier=float(
                reward_result["expert_metrics"]["reward_modifier"]
            ),
            expert_phase=str(reward_result["expert_metrics"]["phase"]),
            expert_feedback=str(reward_result["expert_metrics"]["feedback"]),
            replay_candidate=False,
            replay_novelty=0.0,
            replay_added=False,
            combined_reward=terminal_reward,
            reward_breakdown=reward_result,
            topics_in_sweet_spot=self.curriculum_manager.get_sweet_spot_topics(),
            current_focus_topics=self.curriculum_manager.get_current_focus(),
            curriculum_state_snapshot=self.curriculum_manager.get_curriculum_stats(),
        )
        metadata_dict = asdict(metadata)
        trajectory.metadata = metadata_dict

        # Replay admission: requires trajectory.metadata to already exist
        # because check_novelty reads metadata["generated_question"].
        is_candidate, reason = self.quality_filter.meets_replay_criteria(metadata_dict)
        metadata_dict["replay_candidate"] = is_candidate
        if is_candidate:
            novelty_score = self.quality_filter.check_novelty(
                trajectory, self.replay_buffer.buffer
            )
            metadata_dict["replay_novelty"] = float(novelty_score)
            if self.quality_filter.is_novel_enough(novelty_score):
                quality_score = self.quality_filter.compute_quality_score(metadata_dict)
                self.replay_buffer.add_trajectory(
                    trajectory=trajectory,
                    metadata=metadata_dict,
                    iteration=self.curriculum_manager.current_iteration,
                    quality_score=quality_score,
                )
                metadata_dict["replay_added"] = True
            else:
                metadata_dict["replay_added"] = False
        else:
            metadata_dict["replay_added"] = False
            metadata_dict["replay_reject_reason"] = reason

        trajectory.metadata = metadata_dict
        return trajectory

    def _get_adaptive_replay_ratio(self) -> float:
        iteration = self.curriculum_manager.current_iteration
        if iteration < 3:
            return 0.0
        if iteration < 5:
            return 0.15

        buffer_stats = self.replay_buffer.get_buffer_stats(current_iteration=iteration)
        buffer_health = float(buffer_stats.get("buffer_health", 0.0))
        if buffer_health >= 0.75:
            return 0.3
        if buffer_health >= 0.6:
            return 0.25
        return 0.2

    def collect_rollouts(
        self,
        num_trajectories: int,
        verbose: bool = True,
        grounded_ratio: float = 0.0,
    ) -> List[Trajectory]:
        """
        Generate ``num_trajectories`` episodes in-process on the current
        device.

        Mix:
          * ``grounded_ratio`` of rollouts are GSM8K-anchored (real question,
            reward scored against gold final answer).  These give the policy
            a clean gradient toward benchmark correctness and are also ~3x
            faster than self-play rollouts (no TripleVerifier call).
          * an adaptive fraction is drawn from the replay buffer when buffer
            health is good (self-play only).
          * the remainder are fresh self-play rollouts.
        """
        if num_trajectories <= 0:
            return []

        # Defensive .eval() on both policy and value before any generation.
        # PPOTrainer.train_step restores eval mode at its tail, but on the
        # very first iteration rollouts run right after model load (HF
        # default is .train()).  Qwen2.5 has zero dropout so this is
        # currently cosmetic, but cheap insurance against any future
        # model swap that has real stochastic layers.
        self.policy.eval()
        self.value.eval()

        # Grounded rollouts: only if we actually have QA pairs loaded.
        if grounded_ratio > 0.0 and self.grounded_qa_pairs:
            num_grounded = int(round(num_trajectories * grounded_ratio))
            num_grounded = min(num_grounded, num_trajectories)
        else:
            num_grounded = 0
        num_selfplay = num_trajectories - num_grounded

        # Within the self-play half, the existing replay-buffer mix applies.
        replay_ratio = self._get_adaptive_replay_ratio()
        num_replay = int(num_selfplay * replay_ratio)
        num_replay = min(num_replay, len(self.replay_buffer))
        num_fresh = max(0, num_selfplay - num_replay)

        # ---- Grounded rollouts (GSM8K-anchored) --------------------------
        grounded_trajectories: List[Trajectory] = []
        grounded_correct = 0
        grounded_reward_sum = 0.0
        if num_grounded > 0:
            qa_sample = random.sample(
                self.grounded_qa_pairs,
                k=min(num_grounded, len(self.grounded_qa_pairs)),
            )
            # If we asked for more grounded rollouts than we have distinct
            # pairs, pad by re-sampling with replacement.
            while len(qa_sample) < num_grounded:
                qa_sample.append(random.choice(self.grounded_qa_pairs))
            pbar = tqdm(
                qa_sample,
                desc="Grounded rollouts",
                unit="ep",
                dynamic_ncols=True,
                leave=False,
                disable=not verbose,
            )
            for qa in pbar:
                trajectory = self.rollout_grounded_trajectory(qa)
                grounded_trajectories.append(trajectory)
                r = float(trajectory.metadata.get("combined_reward", 0.0))
                grounded_reward_sum += r
                if bool(trajectory.metadata.get("grounded_gt_match", False)):
                    grounded_correct += 1
                done = len(grounded_trajectories)
                pbar.set_postfix(
                    acc=f"{grounded_correct / done:.1%}",
                    reward=f"{grounded_reward_sum / done:+.3f}",
                    refresh=False,
                )

        # ---- Fresh self-play rollouts ------------------------------------
        fresh_trajectories: List[Trajectory] = []
        pbar = tqdm(
            range(num_fresh),
            desc="Self-play rollouts",
            unit="ep",
            dynamic_ncols=True,
            leave=False,
            disable=not verbose,
        )
        running_reward = 0.0
        running_ok = 0
        for _ in pbar:
            trajectory = self.rollout_trajectory()
            trajectory.metadata["rollout_source"] = "fresh"
            fresh_trajectories.append(trajectory)

            running_reward += float(trajectory.metadata.get("combined_reward", 0.0))
            if trajectory.metadata.get("final_answer_ok", False):
                running_ok += 1
            done = len(fresh_trajectories)
            pbar.set_postfix(
                reward=f"{running_reward / done:+.3f}",
                ok=f"{running_ok}/{done}",
                refresh=False,
            )

        # ---- Replay buffer draws -----------------------------------------
        replay_trajectories = self.replay_buffer.sample_replay_batch(
            num_replay, diversity_sample=True
        )
        for trajectory in replay_trajectories:
            trajectory.metadata["rollout_source"] = "replay"

        trajectories = (
            grounded_trajectories + fresh_trajectories + replay_trajectories
        )
        random.shuffle(trajectories)

        self.last_replay_ratio = replay_ratio
        self.last_rollout_mix = {
            "fresh": len(fresh_trajectories),
            "replay": len(replay_trajectories),
            "grounded": len(grounded_trajectories),
        }
        grounded_count = len(grounded_trajectories)
        self.last_grounded_stats = {
            "count": grounded_count,
            "correct": grounded_correct,
            "accuracy": (
                grounded_correct / grounded_count if grounded_count > 0 else 0.0
            ),
            "mean_reward": (
                grounded_reward_sum / grounded_count if grounded_count > 0 else 0.0
            ),
        }

        if verbose:
            buffer_stats = self.replay_buffer.get_buffer_stats(
                current_iteration=self.curriculum_manager.current_iteration
            )
            logger.info(
                "Rollout mix: %d grounded + %d fresh + %d replay "
                "(grounded_ratio=%.2f, replay_ratio=%.2f, buffer_size=%d, health=%.3f)",
                len(grounded_trajectories),
                len(fresh_trajectories),
                len(replay_trajectories),
                grounded_ratio,
                replay_ratio,
                len(self.replay_buffer),
                float(buffer_stats.get("buffer_health", 0.0)),
            )
            if grounded_count > 0:
                logger.info(
                    "Grounded accuracy this iter: %d/%d = %.1f%%  (mean reward %.3f)",
                    grounded_correct,
                    grounded_count,
                    100.0 * grounded_correct / grounded_count,
                    grounded_reward_sum / grounded_count,
                )

        self.curriculum_manager.increment_iteration()
        self.curriculum_manager.save_state(
            iteration=self.curriculum_manager.current_iteration, rollout=None
        )
        return trajectories
