"""Tests for SymPy-oriented solver formatting (structure + parse checks)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.sft.solution_format import (
    strip_gsm8k_scratchpads,
    validate_sympy_solution_format,
)

FIXTURE_JSONL = Path(__file__).resolve().parent / "fixtures" / "gsm8k_sft_mini.jsonl"

VALID_SIMPLE = """Step 1: 48/2
Step 2: 48+24
Final Answer: 72"""

VALID_WITH_SCRATCH = """Step 1: Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Final Answer: 24"""


def test_validate_good_simple():
    r = validate_sympy_solution_format(VALID_SIMPLE)
    assert r.ok
    assert r.step_count == 2
    assert r.has_final_line
    assert r.sympy_parseable_final
    assert r.final_answer_raw == "72"


def test_validate_rejects_missing_final():
    r = validate_sympy_solution_format("Step 1: 1+1")
    assert not r.ok
    assert any("Final Answer" in e for e in r.errors)


def test_validate_rejects_bad_final_sympy():
    r = validate_sympy_solution_format("Step 1: 1\nFinal Answer: @@@")
    assert not r.ok
    assert any("parse" in e.lower() for e in r.errors)


def test_strip_scratchpads():
    out = strip_gsm8k_scratchpads(VALID_WITH_SCRATCH)
    assert "<<" not in out
    assert "24 clips" in out


def test_fixture_jsonl_messages_validate():
    line = FIXTURE_JSONL.read_text(encoding="utf-8").strip().splitlines()[0]
    rec = json.loads(line)
    assistant = next(m["content"] for m in rec["messages"] if m["role"] == "assistant")
    r = validate_sympy_solution_format(assistant)
    assert r.ok, r.errors


@pytest.mark.parametrize(
    "text,min_steps,expect_ok",
    [
        ("Step 1: x\nFinal Answer: 1", 1, True),
        ("Final Answer: 42", 1, False),  # no Step line when min_steps=1
    ],
)
def test_min_steps(text, min_steps, expect_ok):
    r = validate_sympy_solution_format(text, min_steps=min_steps)
    assert r.ok is expect_ok
