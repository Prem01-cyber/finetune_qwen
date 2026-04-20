"""
Utilities for SymPy-oriented solver output: validation and GSM8K trace cleanup.

Aligned with ``src.agent.math_agent.SOLVER_SYSTEM_PROMPT`` (Step N: / Final Answer:).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from sympy.parsing.sympy_parser import parse_expr

from src.sft.sympy_normalize import normalize_for_parse_expr


STEP_RE = re.compile(r"^Step\s+(\d+)\s*:", re.IGNORECASE | re.MULTILINE)
FINAL_RE = re.compile(r"(?im)^Final\s*Answer\s*:\s*([^\n]+?)\s*$")


@dataclass
class FormatCheckResult:
    ok: bool
    step_count: int
    has_final_line: bool
    final_answer_raw: str
    sympy_parseable_steps: int
    sympy_parseable_final: bool
    errors: List[str]


def strip_gsm8k_scratchpads(text: str) -> str:
    """Remove GSM8K ``<<...>>`` calculator traces; collapse extra spaces."""
    s = re.sub(r"<<[^>]*>>", "", text)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _step_bodies(text: str) -> List[str]:
    """Text after each 'Step N:' up to next Step or Final Answer (best-effort)."""
    lines = text.splitlines()
    bodies: List[str] = []
    cur: List[str] = []
    in_step = False
    for line in lines:
        if re.match(r"^\s*Step\s+\d+\s*:", line, re.I):
            if cur:
                bodies.append("\n".join(cur).strip())
            cur = [re.sub(r"^\s*Step\s+\d+\s*:\s*", "", line, flags=re.I)]
            in_step = True
        elif re.match(r"^\s*Final\s*Answer\s*:", line, re.I):
            if cur:
                bodies.append("\n".join(cur).strip())
            cur = []
            in_step = False
            break
        elif in_step:
            cur.append(line)
    if cur:
        bodies.append("\n".join(cur).strip())
    return [b for b in bodies if b]


def _sympy_can_parse_fragment(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    # Normalize using shared normalizer (handles ^, currency, etc.)
    s = normalize_for_parse_expr(s)
    # Take first line or expression-ish segment after last '='
    chunk = s
    if "=" in s and "==" not in s:
        chunk = s.split("=")[-1].strip()
    chunk = chunk.split()[0] if chunk.split() else chunk
    try:
        parse_expr(chunk)
        return True
    except Exception:
        try:
            parse_expr(s[:200])
            return True
        except Exception:
            return False


def validate_sympy_solution_format(
    text: str,
    *,
    require_step_prefix: bool = True,
    require_final_answer: bool = True,
    min_steps: int = 1,
) -> FormatCheckResult:
    """
    Check solution text for structural compliance and loose SymPy parseability.

    Steps: at least ``min_steps`` lines starting with ``Step N:``.
    Final: a line ``Final Answer: ...`` where the RHS should parse with SymPy
    (integers and simple rationals usually succeed).
    """
    errors: List[str] = []
    steps = STEP_RE.findall(text)
    step_count = len(steps)

    if require_step_prefix and step_count < min_steps:
        errors.append(f"expected at least {min_steps} Step N: line(s), found {step_count}")

    m_final = None
    for m in FINAL_RE.finditer(text):
        m_final = m
    has_final = m_final is not None
    final_raw = m_final.group(1).strip() if m_final else ""

    if require_final_answer and not has_final:
        errors.append("missing 'Final Answer:' line")

    sympy_final = False
    if final_raw:
        try:
            parse_expr(normalize_for_parse_expr(final_raw))
            sympy_final = True
        except Exception:
            errors.append(f"final answer does not parse as SymPy expr: {final_raw!r}")

    bodies = _step_bodies(text)
    sympy_parseable_steps = len([b for b in bodies if _sympy_can_parse_fragment(b)])

    ok = len(errors) == 0
    return FormatCheckResult(
        ok=ok,
        step_count=step_count,
        has_final_line=has_final,
        final_answer_raw=final_raw,
        sympy_parseable_steps=sympy_parseable_steps,
        sympy_parseable_final=sympy_final,
        errors=errors,
    )


def extract_final_answer_numeric_str(text: str) -> Optional[str]:
    """Return substring after 'Final Answer:' if present."""
    m = list(FINAL_RE.finditer(text))
    if not m:
        return None
    return m[-1].group(1).strip()
