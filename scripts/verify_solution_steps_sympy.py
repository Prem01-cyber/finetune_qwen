#!/usr/bin/env python3
"""
Verify each ``Step N:`` line in model output using SymPy (equality chains).

Reads full solution text (stdin, file, or ``--text``) and prints a JSON report.

Examples
--------
  python scripts/verify_solution_steps_sympy.py --text "$(cat sample.txt)"

  python scripts/verify_solution_steps_sympy.py -i outputs/run1.txt --verbose

Environment: project root must be on ``PYTHONPATH`` (this script adds it).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if "HF_HUB_DISABLE_XET" not in os.environ:
    os.environ["HF_HUB_DISABLE_XET"] = "1"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sft.step_verify_sympy import print_report, report_to_dict, verify_solution_text


def main() -> None:
    p = argparse.ArgumentParser(description="SymPy step-by-step verification for solver output.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", type=str, help="Full solution text (include Step lines and Final Answer).")
    g.add_argument("-i", "--input", type=Path, help="Path to UTF-8 text file.")
    p.add_argument("--json", action="store_true", help="Print machine-readable JSON only.")
    p.add_argument("--verbose", action="store_true", help="Include step bodies and line text.")
    args = p.parse_args()

    if args.text is not None:
        text = args.text
    else:
        text = args.input.read_text(encoding="utf-8")

    report = verify_solution_text(text)
    if args.json:
        print(json.dumps(report_to_dict(report), indent=2, ensure_ascii=False))
    else:
        print_report(report, verbose=args.verbose)


if __name__ == "__main__":
    main()
