"""
Unified Accuracy Calculator for GRPO training.

Replaces opaque PRM-based step scoring (Phase 1) with formally-grounded
chain integrity scoring (Phase 2+) using a small LLM extractor plus
eval()/SymPy for arithmetic verification and dependency consistency checks.

Architecture:
    Solution text
         ↓
    StepChainExtractor  (small LLM, 4-bit; cache-first for grounded data)
         ↓
    ExtractionResult  (steps + success flag)
         ↓
    _pal_eval / _sympy_eval  (formal arithmetic verification)
    _value_used_in_expr      (dependency consistency check)
         ↓
    AccuracyReport  (arith + dep + lccp + final + q_score)
         ↓
    UnifiedAccuracyCalculator.compute()  →  AccuracyReport
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

CHAIN_EXTRACT_PROMPT = """\
Extract ALL arithmetic claims from these math solution steps.
Return ONLY a JSON array, no other text.

Each element:
{{
  "step": <int>,
  "expression": <lhs as Python expression; replace x and × with *, ÷ with /; null if no arithmetic>,
  "claimed": <rhs value as string; null if no arithmetic>,
  "produces": <short variable name this step computes, e.g. "uphill_miles">,
  "uses": [<list of variable names from prior steps that feed into this expression>]
}}

Rules:
- Replace x and × with *
- Replace ÷ with /
- Keep fractions as-is: (2/3) stays (2/3)
- If a step has no arithmetic claim, still include it with expression=null and claimed=null
- "uses" tracks which prior step's output feeds into this expression

Steps:
{steps}

JSON array:"""

# ---------------------------------------------------------------------------
# Module-level helper functions
# ---------------------------------------------------------------------------

_FINAL_ANSWER_RE = re.compile(r"final answer[:\s]*([^\n]+)", re.IGNORECASE)
_STEP_RE = re.compile(r"^\s*Step\s+\d+\s*:", re.IGNORECASE | re.MULTILINE)


def _cache_key(question: str, solution: str) -> str:
    """
    Cache key on (question, solution) to prevent collisions when two
    different problems share identical solution text (common in short
    MATH Level 1–2 examples).
    """
    return hashlib.md5(
        f"{question}\n{solution}".encode(), usedforsecurity=False
    ).hexdigest()


def _extract_final_answer(solution: str) -> Optional[str]:
    """Return the text after 'Final Answer:' in a solution."""
    m = _FINAL_ANSWER_RE.search(solution)
    return m.group(1).strip() if m else None


def _extract_step_bodies(solution: str) -> List[str]:
    """Split solution into individual step text strings."""
    parts = _STEP_RE.split(solution)
    bodies: List[str] = []
    for p in parts:
        stripped = p.strip()
        if stripped:
            bodies.append(stripped)
    return bodies


def _pal_eval(answer_str: str) -> Optional[float]:
    """
    Tier 1: arithmetic / basic algebra via safe eval.
    No builtins, no names — only numeric Python expressions.
    """
    try:
        val = eval(answer_str, {"__builtins__": {}}, {})  # noqa: S307
        f = float(val)
        return None if f != f else f  # NaN guard
    except Exception:
        return None


def _sympy_eval(answer_str: str) -> Optional[float]:
    """
    Tier 2: symbolic evaluation via SymPy for algebra,
    fractions, square roots, etc.
    """
    try:
        from sympy import N as _N, sympify  # type: ignore
        f = float(_N(sympify(answer_str), 15))
        return None if f != f else f  # NaN guard
    except Exception:
        return None


def _parse_value(raw: str) -> Optional[float]:
    """Try PAL eval first, fall back to SymPy."""
    return _pal_eval(raw) or _sympy_eval(raw)


def _value_used_in_expr(expression: str, expected_value: float, tol: float = 1e-4) -> bool:
    """
    Check whether a prior step's actual value appears in the expression
    that claims to use it — catches silent dependency breaks that PRM misses.

    Three-pass check:
      1. Raw numeric literals ("0.6667" matches 0.6667)
      2. Sub-expressions  ("(2/3)" evaluates to ≈0.6667 → matches)
      3. Full expression evaluation (whole expr IS the prior step's value)

    Example — Roberto Step 7 "60 * (2/3)" where dep value = 0.6667:
      Pass 1: literals [60, 2, 3] — none equal 0.6667 → no match yet
      Pass 2: "(2/3)" evaluates to 0.6667 → MATCH ✓

    Example — broken chain "60 * 0.5" where dep value = 0.6667:
      Pass 1: literals [60, 0.5] — neither equals 0.6667
      Pass 2: no sub-expressions
      Pass 3: 60 * 0.5 = 30.0 ≠ 0.6667
      → False ✓
    """
    # Pass 1: raw numeric literals
    nums = re.findall(r"\d+\.?\d*", expression)
    for n in nums:
        try:
            if abs(float(n) - expected_value) < tol:
                return True
        except ValueError:
            pass

    # Pass 2: evaluate sub-expressions like (2/3), (1+2), etc.
    sub_exprs = re.findall(r"\([\d\s\+\-\*\/\.]+\)", expression)
    for sub in sub_exprs:
        try:
            val = eval(sub, {"__builtins__": {}}, {})  # noqa: S307
            if abs(float(val) - expected_value) < tol:
                return True
        except Exception:
            pass

    # Pass 3: evaluate the full expression and check if it equals the dep
    try:
        full_val = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        if abs(float(full_val) - expected_value) < tol:
            return True
    except Exception:
        pass

    return False


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ExtractionResult:
    """
    Result of a step chain extraction attempt.

    Distinguishing ``success=False`` (LLM call failed / JSON unparseable)
    from ``steps=[]`` (no arithmetic claims found) is critical: the former
    should penalise the chain score to 0.5 (neutral), NOT reward it with 1.0
    (which would happen if the calculator sees an empty chain and defaults
    to 'all steps correct').
    """
    steps: List[dict]
    success: bool          # False = LLM or JSON parse failed
    n_steps_found: int     # steps with non-null expression (arithmetic claims)


@dataclass
class ChainStep:
    step: int
    expression: Optional[str]
    claimed: Optional[str]
    produces: str
    uses: List[str] = field(default_factory=list)
    arithmetic_correct: Optional[bool] = None   # None = no arithmetic claim
    dependency_consistent: Optional[bool] = None  # None = no deps to check
    actual_value: Optional[float] = None


@dataclass
class AccuracyReport:
    # Step chain integrity
    step_arithmetic_score: float    # fraction of steps with correct arithmetic
    step_dependency_score: float    # fraction of deps using correct prior values
    chain_integrity_score: float    # 0.6 * arith + 0.4 * dep
    first_failure_step: Optional[int]
    lccp_score: float               # fraction of clean steps before first failure

    # Final answer
    final_answer_correct: bool      # against gold (grounded) or own chain (self-play)
    final_answer_consistent: bool   # consistent with step chain

    # Question quality
    # Always float (0.0 default) so downstream averaging never hits TypeError.
    # Check question_scored to know whether it was actually evaluated.
    question_score: float = 0.0
    question_scored: bool = False

    # Extraction status
    extraction_succeeded: bool = True   # False when extractor returned failure

    # Composite (replaces PRM-based combined_score in Phase 2+)
    composite_accuracy: float = 0.0


# ---------------------------------------------------------------------------
# StepChainExtractor
# ---------------------------------------------------------------------------


class StepChainExtractor:
    """
    Extracts structured step chains from math solutions using a small LLM.

    For grounded data (fixed GSM8K + MATH training set) the cache avoids
    calling the LLM at training time — only novel self-play solutions
    incur a forward pass.

    Cache format: {"<md5(question+solution)>": {"steps": [...], "success": bool}}
    Stores success status so failure entries are not retried and are correctly
    penalised (not rewarded) by the calculator.
    """

    def __init__(
        self,
        model_name: str,
        device: str,
        cache_path: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.cache_path = cache_path
        # Each entry: {"steps": List[dict], "success": bool}
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._model: Any = None
        self._tokenizer: Any = None
        # Lifetime extraction counters for calibration reporting
        self.n_extractions: int = 0
        self.n_successful: int = 0

        if cache_path:
            self.load_cache()

    # ── Model loading ────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Load the small LLM. Call warmup() at startup for eager loading."""
        if self._model is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch

        logger.info("Loading step chain extractor: %s", self.model_name)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map={"": self.device},
            trust_remote_code=True,
        )
        self._model.eval()
        logger.info("Step chain extractor loaded")

    def warmup(self) -> None:
        """
        Eagerly load the extractor model at startup.

        Call this immediately after __init__ in the training script to avoid
        a 30–60 second stall on the first iteration that triggers live extraction.
        """
        self._ensure_loaded()

    # ── Core extraction ─────────────────────────────────────────────────────

    def extract(self, solution: str, question: str = "") -> ExtractionResult:
        """
        Return an ExtractionResult for ``solution``.

        Cache key is md5(question + solution) — keying on question prevents
        collisions when two MATH problems share identical solution text.

        Returns ExtractionResult with success=False on LLM/parse failure, so
        the calculator can apply a neutral penalty (0.5) instead of incorrectly
        rewarding the empty chain with score 1.0.
        """
        key = _cache_key(question, solution)
        if key in self._cache:
            entry = self._cache[key]
            steps = entry.get("steps") or []
            success = bool(entry.get("success", True))
            n_claims = sum(1 for s in steps if s.get("expression") is not None)
            return ExtractionResult(steps=steps, success=success, n_steps_found=n_claims)

        result = self._call_extractor(solution)
        self._cache[key] = {"steps": result.steps, "success": result.success}
        self.n_extractions += 1
        if result.success:
            self.n_successful += 1
        return result

    def _call_extractor(self, solution: str) -> ExtractionResult:
        """Run a forward pass of the small LLM to extract step chain JSON."""
        step_bodies = _extract_step_bodies(solution)
        if not step_bodies:
            # No Step N: lines — treat as no arithmetic claims (not a failure)
            return ExtractionResult(steps=[], success=True, n_steps_found=0)

        try:
            self._ensure_loaded()
            import torch

            steps_text = "\n".join(
                f"Step {i + 1}: {body}" for i, body in enumerate(step_bodies)
            )
            prompt = CHAIN_EXTRACT_PROMPT.format(steps=steps_text)

            if hasattr(self._tokenizer, "apply_chat_template"):
                messages = [{"role": "user", "content": prompt}]
                full_prompt = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                full_prompt = prompt

            inputs = self._tokenizer(
                full_prompt, return_tensors="pt", truncation=True, max_length=2048
            ).to(self.device)

            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
            raw_text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

            json_match = re.search(r"\[.*\]", raw_text, re.DOTALL)
            if not json_match:
                logger.debug("Extractor produced no JSON array; raw: %s", raw_text[:200])
                return ExtractionResult(steps=[], success=False, n_steps_found=0)

            chain = json.loads(json_match.group())
            if not isinstance(chain, list):
                return ExtractionResult(steps=[], success=False, n_steps_found=0)

            n_claims = sum(1 for s in chain if isinstance(s, dict) and s.get("expression") is not None)
            return ExtractionResult(steps=chain, success=True, n_steps_found=n_claims)

        except Exception as exc:
            logger.debug("StepChainExtractor._call_extractor failed: %s", exc)
            return ExtractionResult(steps=[], success=False, n_steps_found=0)

    # ── Cache management ────────────────────────────────────────────────────

    def build_cache(self, qa_pairs: List[Tuple[str, str]]) -> None:
        """
        Pre-extract step chains for (question, solution) pairs.

        Accepts a list of ``(question, solution)`` tuples.  Keying on both
        prevents cache collisions between MATH problems with identical
        solution text.

        Used by the offline preprocessing script to warm the cache before
        training.  Skips entries already in cache (resume support).
        """
        import tqdm as _tqdm
        for question, solution in _tqdm.tqdm(qa_pairs, desc="Extracting step chains"):
            key = _cache_key(question, solution)
            if key not in self._cache:
                result = self._call_extractor(solution)
                self._cache[key] = {"steps": result.steps, "success": result.success}

    def save_cache(self) -> None:
        if not self.cache_path:
            return
        import pathlib
        pathlib.Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f)
        logger.info(
            "Extraction cache saved: %d entries → %s", len(self._cache), self.cache_path
        )

    def load_cache(self) -> None:
        if not self.cache_path:
            return
        try:
            with open(self.cache_path, encoding="utf-8") as f:
                raw = json.load(f)
            # Migrate old format (plain list values) to new dict format
            migrated = 0
            for k, v in raw.items():
                if isinstance(v, list):
                    raw[k] = {"steps": v, "success": True}
                    migrated += 1
            self._cache = raw
            logger.info(
                "Extraction cache loaded: %d entries from %s%s",
                len(self._cache),
                self.cache_path,
                f" ({migrated} migrated from old format)" if migrated else "",
            )
        except FileNotFoundError:
            logger.info(
                "Extraction cache not found at %s — will build on first use",
                self.cache_path,
            )
        except Exception as exc:
            logger.warning("Failed to load extraction cache: %s", exc)


# ---------------------------------------------------------------------------
# UnifiedAccuracyCalculator
# ---------------------------------------------------------------------------


class UnifiedAccuracyCalculator:
    """
    Compute an AccuracyReport for a given solution.

    Phase-gated: activated when math_env.use_chain_scoring is True (Phase 2+).
    During Phase 2 SELFPLAY_RAMP the calculator also runs in shadow mode
    (computing scores without affecting rewards) to build calibration data
    for the data-driven chain-vs-PRM correlation check.
    """

    def __init__(
        self,
        extractor: StepChainExtractor,
        question_evaluator: Any = None,
    ) -> None:
        self.extractor = extractor
        self.question_evaluator = question_evaluator

    def compute(
        self,
        solution: str,
        gold_answer: Optional[str],
        question: Optional[str] = None,
        topic: str = "arithmetic",
        phase: str = "grounded",   # "grounded" or "selfplay"
    ) -> AccuracyReport:
        """
        Compute a unified AccuracyReport for one solution.

        Parameters
        ----------
        solution     : Full model-generated solution text.
        gold_answer  : Known correct answer (grounded) or None (self-play).
        question     : Question text — also used as cache key discriminator.
        topic        : Problem type tag (passed through for future routing).
        phase        : "grounded" uses gold for correctness; "selfplay" uses chain consistency.
        """
        # ── 1. Extract step chain ──────────────────────────────────────────
        extraction = self.extractor.extract(solution, question=question or "")

        # Handle extraction failure: apply neutral penalty (0.5) rather than
        # rewarding the empty chain with the default 1.0 score.
        if not extraction.success:
            return AccuracyReport(
                step_arithmetic_score=0.5,
                step_dependency_score=0.5,
                chain_integrity_score=0.5,
                first_failure_step=None,
                lccp_score=0.0,
                final_answer_correct=False,
                final_answer_consistent=False,
                question_score=0.0,
                question_scored=False,
                extraction_succeeded=False,
                composite_accuracy=0.25,  # penalised for unverifiable chain
            )

        # Handle genuine "no arithmetic claims" (no Step N: lines, or all
        # narrative steps): treat as neutral, not perfect or failed.
        if extraction.n_steps_found == 0:
            arith_score = 0.5
            dep_score   = 0.5
            lccp        = 1.0   # no steps → no failures in prefix
        else:
            arith_score, dep_score, lccp, _ = self._verify_chain(
                extraction.steps
            )

        chain_steps_parsed = self._parse_chain(extraction.steps)
        first_failure = self._find_first_failure(chain_steps_parsed)
        chain_score = 0.6 * arith_score + 0.4 * dep_score

        # ── 2. Final answer ────────────────────────────────────────────────
        final_raw = _extract_final_answer(solution)
        final_val = _parse_value(final_raw) if final_raw else None

        # Reconstruct value_registry for consistency check
        value_registry: Dict[str, float] = {}
        for sr in chain_steps_parsed:
            if sr.actual_value is not None:
                value_registry[sr.produces] = sr.actual_value

        chain_final: Optional[float] = (
            list(value_registry.values())[-1] if value_registry else None
        )
        final_consistent = (
            abs(final_val - chain_final) < 1e-4
            if final_val is not None and chain_final is not None
            else False
        )

        if phase == "grounded" and gold_answer is not None and final_val is not None:
            gold_val = _parse_value(gold_answer)
            final_correct = (
                abs(final_val - gold_val) < 1e-4
                if gold_val is not None else False
            )
        else:
            final_correct = final_consistent

        # ── 3. Question quality (self-play only) ──────────────────────────
        q_score: float = 0.0
        q_scored: bool = False
        if phase == "selfplay" and question and self.question_evaluator is not None:
            try:
                q_result = self.question_evaluator.evaluate(
                    question=question,
                    solution=solution,
                    consensus_result={
                        "has_majority": final_correct,
                        "consensus_strength": float(chain_score),
                        "primary_matches_majority": final_correct,
                        "answer_diversity": 0,
                        "majority_answer": None,
                        "primary_answer": None,
                    },
                    target_topic=topic,
                    target_difficulty=2.0,
                )
                q_score = float(q_result.get("overall_score", 0.0))
                q_scored = True
            except Exception as exc:
                logger.debug("question_evaluator failed in unified calc: %s", exc)

        # ── 4. Composite accuracy ─────────────────────────────────────────
        if phase == "grounded":
            composite = (
                0.50 * float(final_correct)
                + 0.30 * chain_score
                + 0.20 * lccp
            )
        else:  # selfplay
            composite = (
                0.35 * float(final_correct)
                + 0.30 * chain_score
                + 0.15 * lccp
                + 0.20 * q_score
            )
        composite = max(0.0, min(1.0, composite))

        return AccuracyReport(
            step_arithmetic_score=arith_score,
            step_dependency_score=dep_score,
            chain_integrity_score=chain_score,
            first_failure_step=first_failure,
            lccp_score=lccp,
            final_answer_correct=final_correct,
            final_answer_consistent=final_consistent,
            question_score=q_score,
            question_scored=q_scored,
            extraction_succeeded=True,
            composite_accuracy=composite,
        )

    # ── Internal helpers ────────────────────────────────────────────────────

    def _verify_chain(
        self, raw_chain: List[dict]
    ) -> Tuple[float, float, float, List[ChainStep]]:
        """
        Verify arithmetic and dependencies for a parsed chain.

        Returns (arith_score, dep_score, lccp, chain_steps).
        """
        chain_steps = self._parse_chain(raw_chain)
        value_registry: Dict[str, float] = {}
        first_failure: Optional[int] = None

        for sr in chain_steps:
            if sr.expression is None:
                sr.arithmetic_correct = None
                continue

            actual  = _parse_value(sr.expression)
            claimed = _parse_value(sr.claimed) if sr.claimed else None

            if actual is not None and claimed is not None:
                sr.arithmetic_correct = abs(actual - claimed) < 1e-4
                sr.actual_value = actual
            else:
                sr.arithmetic_correct = None

            if sr.uses and actual is not None:
                dep_ok = True
                for dep_name in sr.uses:
                    if dep_name in value_registry:
                        dep_ok = dep_ok and _value_used_in_expr(
                            sr.expression, value_registry[dep_name]
                        )
                sr.dependency_consistent = dep_ok

            if actual is not None:
                value_registry[sr.produces] = actual

            if sr.arithmetic_correct is False and first_failure is None:
                first_failure = sr.step

        checked     = [s for s in chain_steps if s.arithmetic_correct is not None]
        dep_checked = [s for s in chain_steps if s.dependency_consistent is not None]

        arith_score = (
            sum(1.0 for s in checked if s.arithmetic_correct) / len(checked)
            if checked else 0.5
        )
        dep_score = (
            sum(1.0 for s in dep_checked if s.dependency_consistent) / len(dep_checked)
            if dep_checked else 0.5
        )

        lccp = (
            (first_failure - 1) / len(chain_steps)
            if first_failure is not None and chain_steps
            else 1.0
        )
        lccp = max(0.0, min(1.0, lccp))

        return arith_score, dep_score, lccp, chain_steps

    @staticmethod
    def _find_first_failure(chain_steps: List[ChainStep]) -> Optional[int]:
        for sr in chain_steps:
            if sr.arithmetic_correct is False:
                return sr.step
        return None

    @staticmethod
    def _parse_chain(raw_chain: List[dict]) -> List[ChainStep]:
        """Convert raw JSON dicts from the extractor into ChainStep objects."""
        steps: List[ChainStep] = []
        for item in raw_chain:
            if not isinstance(item, dict):
                continue
            try:
                steps.append(ChainStep(
                    step=int(item.get("step", len(steps) + 1)),
                    expression=item.get("expression"),
                    claimed=item.get("claimed"),
                    produces=str(
                        item.get("produces") or f"step_{len(steps) + 1}_result"
                    ),
                    uses=list(item.get("uses") or []),
                ))
            except Exception:
                continue
        return steps
