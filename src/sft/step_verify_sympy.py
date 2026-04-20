"""
Verify ``Step N:`` lines in solver output using SymPy where possible.

Steps may mix prose and equations. We look for single ``=`` chains (not ``==``,
``<=``, ``>=``) and check adjacent segments parse to equal expressions.

Limitations:

- Narrative-only lines (no ``=``) are ``skipped``.
- Lines that SymPy cannot parse (e.g. ``f'(x)`` notation) are ``skipped``, not
  counted as arithmetic errors.
- Internally consistent but **wrong reasoning** (bad problem setup) still shows
  ``ok`` for each equality chain — only adjacent segments along ``=`` are checked.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, List, Optional

from sympy import Expr, simplify, sympify
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from src.sft.solution_format import _step_bodies, extract_final_answer_numeric_str
from src.sft.sympy_normalize import normalize_for_parse_expr, prefer_arithmetic_tail

_TRANSFORMATIONS_STRICT = standard_transformations
_TRANSFORMATIONS_LOOSE = standard_transformations + (implicit_multiplication_application,)

# Single '=', not part of ==, <=, >=, !=, =>
_SINGLE_EQ_SPLIT = re.compile(r"(?<![=<>!])=(?!=)")


@dataclass
class LineCheck:
    line_index: int
    raw: str
    status: str  # "ok" | "fail" | "skipped"
    detail: str
    parts: List[str]
    parsed_ok: bool


@dataclass
class StepCheck:
    step_index: int
    body: str
    status: str  # "ok" | "fail" | "skipped" | "mixed"
    lines: List[LineCheck]
    detail: str


@dataclass
class FinalCheck:
    status: str
    detail: str
    raw: str


@dataclass
class VerificationReport:
    steps: List[StepCheck]
    final_answer: FinalCheck
    summary: dict[str, Any]


def _try_parse(s: str):
    """Return a SymPy expr or None. Strict parse first, then loose (implicit mult) for ``3x``-style."""
    s0 = normalize_for_parse_expr(s)
    if not s0:
        return None
    candidates: List[str] = [s0, prefer_arithmetic_tail(s0)]
    cur = s0
    for _ in range(24):
        m = re.search(r"\s+[^\d=+\-*/().\s][^\n]*$", cur)
        if not m:
            break
        cur = cur[: m.start()].strip()
        if cur and cur not in candidates:
            candidates.append(cur)
            candidates.append(prefer_arithmetic_tail(cur))
        if not cur:
            break
    seen: set[str] = set()
    uniq = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            uniq.append(c)
    for c in uniq:
        for trans in (_TRANSFORMATIONS_STRICT, _TRANSFORMATIONS_LOOSE):
            try:
                return parse_expr(c, transformations=trans)
            except Exception:
                continue
    return None


def _coerce_expr_for_subtraction(obj: Any, segment: str) -> Optional[Expr]:
    """
    ``parse_expr`` may return a Python ``tuple`` from comma-separated subexpressions
    (Tuple semantics). Subtraction is only defined for single SymPy ``Expr`` values.
    """
    if obj is None:
        return None
    if isinstance(obj, Expr):
        return obj
    if isinstance(obj, (tuple, list)):
        return None
    try:
        s = sympify(obj)
        if isinstance(s, Expr):
            return s
    except (TypeError, ValueError, AttributeError):
        pass
    return None


def _strip_leading_nonmath(s: str) -> str:
    """Prefer substring from first digit to avoid English words as symbols; else full string."""
    return prefer_arithmetic_tail(s)


def _verify_chain(parts: List[str]) -> tuple[str, str]:
    """Return (status, detail) where status is ok | fail | skipped."""
    if len(parts) < 2:
        return "skipped", "need at least two sides for '='"
    exprs: List[Expr] = []
    for i, p in enumerate(parts):
        p2 = _strip_leading_nonmath(p)
        e = _try_parse(p2)
        if e is None:
            return "skipped", f"could not parse segment {i}: {p!r}"
        coerced = _coerce_expr_for_subtraction(e, p2)
        if coerced is None:
            return (
                "skipped",
                f"segment {i} is not a single SymPy expression (got {type(e).__name__}): {p!r}",
            )
        exprs.append(coerced)
    for i in range(len(exprs) - 1):
        diff = simplify(exprs[i] - exprs[i + 1])
        if diff != 0:
            return "fail", f"not equal: {exprs[i]} vs {exprs[i + 1]}"
    return "ok", "chain verified"


def _check_line(line: str, line_index: int) -> LineCheck:
    raw = line.strip()
    if not raw:
        return LineCheck(
            line_index=line_index,
            raw=raw,
            status="skipped",
            detail="empty line",
            parts=[],
            parsed_ok=False,
        )
    if "=" not in raw:
        return LineCheck(
            line_index=line_index,
            raw=raw,
            status="skipped",
            detail="no '=' to verify",
            parts=[],
            parsed_ok=False,
        )
    if "==" in raw:
        return LineCheck(
            line_index=line_index,
            raw=raw,
            status="skipped",
            detail="contains '==' (skipped)",
            parts=[],
            parsed_ok=False,
        )
    parts = [p.strip() for p in _SINGLE_EQ_SPLIT.split(raw)]
    if len(parts) < 2:
        return LineCheck(
            line_index=line_index,
            raw=raw,
            status="skipped",
            detail="no splittable equality",
            parts=parts,
            parsed_ok=False,
        )
    st, detail = _verify_chain(parts)
    return LineCheck(
        line_index=line_index,
        raw=raw,
        status=st,
        detail=detail,
        parts=parts,
        parsed_ok=st == "ok",
    )


def verify_step_body(step_index: int, body: str) -> StepCheck:
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if not lines:
        return StepCheck(
            step_index=step_index,
            body=body,
            status="skipped",
            lines=[],
            detail="empty step body",
        )
    checks: List[LineCheck] = []
    for i, ln in enumerate(lines):
        checks.append(_check_line(ln, i))
    statuses = {c.status for c in checks}
    if statuses <= {"skipped"}:
        st = "skipped"
        det = "no line contained a verifiable equality chain"
    elif "fail" in statuses:
        st = "fail"
        det = "at least one equality check failed"
    elif statuses <= {"ok", "skipped"} and "ok" in statuses:
        st = "ok"
        det = "all equality chains that were present verify"
    else:
        st = "mixed"
        det = "unexpected line status mix"
    return StepCheck(
        step_index=step_index,
        body=body,
        status=st,
        lines=checks,
        detail=det,
    )


def verify_final_answer_line(text: str) -> FinalCheck:
    raw = extract_final_answer_numeric_str(text) or ""
    if not raw:
        return FinalCheck(status="skipped", detail="no Final Answer line", raw="")
    e = _try_parse(raw)
    if e is None:
        return FinalCheck(status="fail", detail="could not parse final answer", raw=raw)
    return FinalCheck(status="ok", detail=f"parses as {e}", raw=raw)


def verify_solution_text(text: str) -> VerificationReport:
    bodies = _step_bodies(text)
    steps: List[StepCheck] = []
    for i, body in enumerate(bodies, start=1):
        steps.append(verify_step_body(i, body))
    final = verify_final_answer_line(text)
    n_ok = sum(1 for s in steps if s.status == "ok")
    n_fail = sum(1 for s in steps if s.status == "fail")
    n_skip = sum(1 for s in steps if s.status == "skipped")
    summary = {
        "steps_total": len(steps),
        "steps_verified_ok": n_ok,
        "steps_failed": n_fail,
        "steps_skipped_no_equality": n_skip,
        "final_answer": final.status,
    }
    return VerificationReport(steps=steps, final_answer=final, summary=summary)


def report_to_dict(r: VerificationReport) -> dict[str, Any]:
    return {
        "summary": r.summary,
        "steps": [asdict(s) for s in r.steps],
        "final_answer": asdict(r.final_answer),
    }


def print_report(r: VerificationReport, *, verbose: bool = True) -> None:
    print(json.dumps(r.summary, indent=2))
    for s in r.steps:
        print(f"\n--- Step {s.step_index} [{s.status}] ---")
        print(s.detail)
        if verbose:
            print(s.body)
        for lc in s.lines:
            sym = {"ok": "[ok]", "fail": "[fail]", "skipped": "[skip]"}.get(lc.status, "[?]")
            print(f"  {sym} line {lc.line_index}: {lc.detail}")
            if verbose and lc.raw:
                print(f"      {lc.raw}")
    print("\n--- Final Answer ---")
    fa = r.final_answer
    print(f"  status: {fa.status} — {fa.detail}")
    if fa.raw:
        print(f"  raw: {fa.raw}")
