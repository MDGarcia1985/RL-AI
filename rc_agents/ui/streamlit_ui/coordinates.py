"""
coordinates.py

Streamlit helper: text input -> (row, col) coordinate.

Design intent:
- Engineers will type coordinates quickly:
    "(45, 56)" or "45,56" or "45 56"
- Clamp coords to grid bounds so env never receives invalid indices.
- Keep prior valid coordinate if user types something broken mid-edit.

Copyright (c) 2026 Michael Garcia, M&E Design
https://mandedesign.studio
CSC370 Spring 2026
"""
from __future__ import annotations

import streamlit as st


# ---------------------------------------------------------------------------
# Safe coordinate inputs (row, col)
#
# We want to be able to type:
#   "(45, 56)"  or "45,56"  or "45 56"
#
# This will matter more later when we add:
# - mazes
# - multiple start corners
# - agent-vs-agent tournaments
#
# Design intent:
# - Keep UI flexible for humans
# - Keep parsing safe and simple (no arbitrary eval)
# ---------------------------------------------------------------------------

def _clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _parse_coord(raw: str) -> tuple[int, int]:
    """
    Parse a coordinate string into (row, col) ints.

    Accepted formats:
    - "(45, 56)"
    - "45,56"
    - "45 56"

    NOTE:
    - This is intentionally strict: only two integers.
    - If you want expressions later (e.g., "60-1"), you can add it using _safe_eval_number().
    """
    s = raw.strip()
    if not s:
        raise ValueError("empty coordinate")

    # Strip optional parentheses
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()

    # Split by comma or whitespace
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
    else:
        parts = [p.strip() for p in s.split()]

    if len(parts) != 2:
        raise ValueError("coordinate must have exactly two values")

    r = int(parts[0]) # Row
    c = int(parts[1]) # Collum
    return (r, c)

# Coordinates entered by text string
def _text_coord(
    label: str,
    default: tuple[int, int],
    *,
    rows: int,
    cols: int,
    key: str,
) -> tuple[int, int]:
    """
    Sidebar text input -> (row, col) tuple.

    - Keeps prior value if parsing fails.
    - Clamps to [0..rows-1] and [0..cols-1] so the env never receives illegal coordinates.
    """
    if key not in st.session_state:
        st.session_state[key] = f"({default[0]}, {default[1]})"
    if f"{key}_input" not in st.session_state:
        st.session_state[f"{key}_input"] = st.session_state[key]

    # The actual input field
    raw = st.text_input(
        label,
        value=st.session_state[f"{key}_input"],
        key=f"{key}_input",
        help="Examples: (45, 56) | 45,56 | 45 56",
    )

    try:
        r, c = _parse_coord(raw) # Parse row, col from string

        # Clamp to valid grid bounds
        r = _clamp_int(r, 0, max(0, rows - 1))
        c = _clamp_int(c, 0, max(0, cols - 1))

        st.session_state[key] = f"({r}, {c})" # Update session state
        st.session_state[f"{key}_input"] = st.session_state[key]
        return (r, c)

    # If we get here, the value was invalid
    except Exception:
        # Silent fallback to last known-good coordinate.
        try:
            return _parse_coord(st.session_state[key])
        except Exception:
            return default
        

# ---------------------------------------------------------------------------
# Public aliases (kept for readability in higher-level modules)
# NOTE:
# - We keep the underscore-prefixed functions as the canonical implementation.
# - These aliases let higher-level modules import without "private" naming.
# ---------------------------------------------------------------------------

text_coord = _text_coord
parse_coord = _parse_coord
clamp_int = _clamp_int