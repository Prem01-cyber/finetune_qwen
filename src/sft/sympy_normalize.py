"""
Normalization layer for LLM outputs before SymPy parsing.

This module provides a single, well-tested function to convert common LLM output
patterns (Unicode operators, currency symbols, implicit styles) into SymPy-friendly
ASCII Python-like expressions suitable for `sympy.parsing.sympy_parser.parse_expr`.

## Why normalize instead of controlling LLM output?

LLMs generate diverse textual math notation (^, ×, π, commas in numbers, etc.) that
cannot be reliably controlled at the token level. A deterministic preprocessing layer
is more robust than trying to force specific character-level outputs during training.

## SymPy parsing context

SymPy's `parse_expr` (docs: https://docs.sympy.org/latest/modules/parsing.html):
- Uses Python-like expression syntax as the base grammar.
- Applies **transformations** (token rewrites) before evaluation.
- Notable transformations:
  - `standard_transformations`: auto symbol/number conversion, factorial notation.
  - `convert_xor`: treats `^` as power (not bitwise XOR).
  - `implicit_multiplication_application`: relaxes syntax (implicit mult, split symbols).
  - LaTeX is a **separate path** via `sympy.parsing.latex.parse_latex` (experimental).

**Security note:** `parse_expr` uses `eval` internally. Treat LLM outputs as untrusted;
this module helps but does not sandbox.

## Normalization mapping (categories)

| Category           | LLM output           | Normalized        | Notes                                  |
|--------------------|----------------------|-------------------|----------------------------------------|
| Power              | `^`                  | `**`              | Python power operator                  |
| Multiplication     | `×`, `·`, `•`        | `*`               | Unicode operators → ASCII              |
| Division           | `÷`                  | `/`               | Unicode division sign → ASCII          |
| Minus sign         | `−` (U+2212)         | `-`               | Typography minus → ASCII hyphen-minus  |
| Comparisons        | `≤`, `≥`, `≠`        | `<=`, `>=`, `!=`  | Relational operators (if parsing them) |
| Constants          | `π`                  | `pi`              | Greek letter → SymPy symbol name       |
| Thousands sep      | `80,000`             | `80000`           | Remove commas in numeric literals      |
| Currency           | `$`, `€`, `£`        | (removed)         | Strip before parsing numeric tails     |
| Extra whitespace   | multiple spaces/tabs | single space      | Collapse for cleaner parsing           |

Not handled (by design):
- **LaTeX** (`\\frac`, `\\sqrt`, etc.): route to `parse_latex` separately if needed.
- **Natural language prefix** ("Janet sells 16-3-4=9 eggs"): caller extracts math tail first.
- **Grouping `[` `]`**: context-dependent; avoid substituting without semantic analysis.

Version lock: sympy==1.14.0 (line 84 in requirements.txt at time of writing).
"""

from __future__ import annotations

import re


def normalize_for_parse_expr(text: str) -> str:
    """
    Normalize LLM-generated math text for SymPy's `parse_expr`.

    Converts common Unicode operators, currency symbols, and formatting quirks
    into ASCII Python-like syntax. This is the single source of truth for
    string preprocessing before SymPy parsing in this project.

    Parameters
    ----------
    text : str
        Raw string (potentially mixed prose and math from LLM).

    Returns
    -------
    str
        Normalized ASCII expression.

    Examples
    --------
    >>> normalize_for_parse_expr("2^3")
    '2**3'
    >>> normalize_for_parse_expr("16 × 3 − 4")
    '16 * 3 - 4'
    >>> normalize_for_parse_expr("$2,500")
    '2500'
    >>> normalize_for_parse_expr("π/2")
    'pi/2'
    """
    s = text.strip()

    # Power: ^ → **
    s = s.replace("^", "**")

    # Multiplication: Unicode operators → *
    s = s.replace("×", "*")
    s = s.replace("·", "*")
    s = s.replace("•", "*")
    s = s.replace("\u00d7", "*")  # U+00D7 MULTIPLICATION SIGN (×)
    s = s.replace("\u22c5", "*")  # U+22C5 DOT OPERATOR (⋅)
    s = s.replace("\u2022", "*")  # U+2022 BULLET (•)

    # Division: Unicode ÷ → /
    s = s.replace("÷", "/")
    s = s.replace("\u00f7", "/")  # U+00F7 DIVISION SIGN

    # Minus: typography minus (U+2212) → ASCII hyphen-minus
    s = s.replace("\u2212", "-")  # U+2212 MINUS SIGN (−)

    # Comparison operators (if ever parsing relations)
    s = s.replace("≤", "<=")
    s = s.replace("≥", ">=")
    s = s.replace("≠", "!=")
    s = s.replace("\u2264", "<=")  # U+2264 LESS-THAN OR EQUAL TO
    s = s.replace("\u2265", ">=")  # U+2265 GREATER-THAN OR EQUAL TO
    s = s.replace("\u2260", "!=")  # U+2260 NOT EQUAL TO

    # Greek constants: π → pi (SymPy symbol name)
    s = s.replace("π", "pi")
    s = s.replace("\u03c0", "pi")  # U+03C0 GREEK SMALL LETTER PI

    # Currency symbols: remove (caller typically strips or segments numeric tails)
    s = re.sub(r"[$€£¥₹]", "", s)

    # Thousands separators in numbers: 80,000 → 80000
    # Match comma only between digits in a numeric context
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)

    # Spoken "times" with ASCII letter x (grade-school / LLM): "4 x 90" must not
    # become 4*x*90 in SymPy (x parsed as a symbol → false failures on chains).
    # Only between digit and digit or digit and '('.
    s = re.sub(r"(?<=\d)\s+[xX]\s+(?=\d|\()", "*", s)

    # Collapse multiple spaces/tabs to single space
    s = re.sub(r"[ \t]+", " ", s)

    # Collapse excessive newlines (keep at most double)
    s = re.sub(r"\n{3,}", "\n\n", s)

    return s.strip()


def prefer_arithmetic_tail(text: str) -> str:
    """
    Return substring starting from the first digit (if present), else full text.

    Useful when LLM outputs mix natural language with equations, e.g.:
      "Janet sells 16 - 3 - 4 = 9 eggs every day"
    This heuristic extracts "16 - 3 - 4 = 9 eggs every day" (digit onward),
    reducing risk of English words being parsed as symbols when implicit
    multiplication transformations are enabled.

    Parameters
    ----------
    text : str
        Potentially mixed prose + math.

    Returns
    -------
    str
        Substring from first digit onward, or original if no digit.

    Examples
    --------
    >>> prefer_arithmetic_tail("Janet sells 16-3-4=9")
    '16-3-4=9'
    >>> prefer_arithmetic_tail("no digits here")
    'no digits here'
    """
    s = normalize_for_parse_expr(text)
    m = re.search(r"\d", s)
    if m:
        return s[m.start() :].strip()
    return s.strip()
