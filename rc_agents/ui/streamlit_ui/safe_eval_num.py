"""
safe_eval_num.py

Safe numeric expression evaluator for Streamlit text inputs.

Design intent:
- Allow users to type numbers like:
    "1500", "20*20*3", "1e-2", "(5+5)*10"
- Avoid python eval() entirely (security + stability).
- Only allow numeric literals and basic arithmetic operators.

This is intentionally restrictive.
If a user types something outside the whitelist, we fail safely.

Copyright (c) 2026 Michael Garcia, M&E Design
https://mandedesign.studio
CSC370 Spring 2026
"""
from __future__ import annotations

import ast
import operator as op

_ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}


def _safe_eval_number(expr: str) -> float:
    """
    Evaluate a numeric expression safely and return float.

    Accepted:
    - numeric literals (int/float)
    - + - * / // % **
    - unary +/-
    - parentheses

    Rejected:
    - names, function calls, attributes, indexing, etc.

    Raises:
        ValueError if expression is empty or unsupported.
    """
    expr = expr.strip()
    if expr == "":
        raise ValueError("empty")

    node = ast.parse(expr, mode="eval")

    # Evaluate the expression, and return the result
    def _eval(n):
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _ALLOWED_OPS:
            return _ALLOWED_OPS[type(n.op)](_eval(n.operand))
        if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED_OPS:
            return _ALLOWED_OPS[type(n.op)](_eval(n.left), _eval(n.right))
        raise ValueError("unsupported expression")

    return float(_eval(node))


# ---------------------------------------------------------------------------
# Public aliases (kept for readability in higher-level modules)
# NOTE:
# - We keep the underscore-prefixed functions as the canonical implementation.
# - These aliases let higher-level modules import without "private" naming.
# ---------------------------------------------------------------------------

safe_eval_number = _safe_eval_number