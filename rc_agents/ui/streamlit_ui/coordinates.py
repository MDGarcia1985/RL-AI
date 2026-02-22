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


def clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def parse_coord(raw: str) -> tuple[int, int]:
    """
    Parse "(r,c)" or "r,c" or "r c" into (r, c).

    NOTE:
    - Strictly two integers.
    - This is not an expression parser (on purpose).
    """
    s = raw.strip()
    if not s:
        raise ValueError("empty coordinate")

    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()

    if "," in s:
        parts = [p.strip() for p in s.split(",")]
    else:
        parts = [p.strip() for p in s.split()]

    if len(parts) != 2:
        raise ValueError("coordinate must have exactly two values")

    return (int(parts[0]), int(parts[1]))


def text_coord(
    label: str,
    default: tuple[int, int],
    *,
    rows: int,
    cols: int,
    key: str,
) -> tuple[int, int]:
    """
    Sidebar text input -> (row, col) tuple.

    Behavior:
    - Keeps prior value if parsing fails
    - Clamps to valid grid bounds
    """
    if key not in st.session_state:
        st.session_state[key] = f"({default[0]}, {default[1]})"
    if f"{key}_input" not in st.session_state:
        st.session_state[f"{key}_input"] = st.session_state[key]

    raw = st.text_input(
        label,
        value=st.session_state[f"{key}_input"],
        key=f"{key}_input",
        help="Examples: (45, 56) | 45,56 | 45 56",
    )

    try:
        r, c = parse_coord(raw)

        # Clamp so env never receives illegal coords.
        r = clamp_int(r, 0, max(0, rows - 1))
        c = clamp_int(c, 0, max(0, cols - 1))

        st.session_state[key] = f"({r}, {c})"
        st.session_state[f"{key}_input"] = st.session_state[key]
        return (r, c)

    except Exception:
        # Silent fallback to last known-good coordinate.
        try:
            return parse_coord(st.session_state[key])
        except Exception:
            return default