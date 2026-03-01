"""
text_num.py

Streamlit helper: text input -> numeric value.

Design intent:
- Streamlit number_input is fine, but typed expressions are better for engineers.
- Maintain "last known-good" value so UI doesn't thrash on partial typing.
- Clamp inputs so downstream env/config never gets illegal values.

NOTE on Streamlit session_state:
- st.session_state[key] stores normalized, last-good value (string)
- st.session_state[f"{key}_input"] stores raw textbox content (string)
Keeping both prevents UI from fighting the user while they type.

Copyright (c) 2026 Michael Garcia, M&E Design
https://mandedesign.studio
CSC370 Spring 2026
"""
from __future__ import annotations

import streamlit as st

from .safe_eval_num import safe_eval_number


def _text_num(
    label: str,
    default: float,
    *,
    min_v: float | None = None,
    max_v: float | None = None,
    step_hint: str | None = None,
    cast=int,
    key: str,
):
    """
    Sidebar text input -> numeric value.

    - Keeps the prior value if parsing fails.
    - Optional clamp.
    - Supports small expressions via _safe_eval_number().

    NOTE on Streamlit state:
    - st.session_state[key] stores "last known-good" normalized value as string
    - st.session_state[f"{key}_input"] stores the live textbox content
    - Keeping both prevents the UI from "fighting" the user while typing.
    """
    if key not in st.session_state:
        st.session_state[key] = str(default)
    if f"{key}_input" not in st.session_state:
        st.session_state[f"{key}_input"] = st.session_state[key]

    help_txt = "Examples: 1500, 20*20*3, 1e-2"
    if step_hint:
        help_txt += f" | Hint: {step_hint}"

    # The actual input field
    raw = st.text_input(
        label,
        value=st.session_state[f"{key}_input"],
        key=f"{key}_input",
        help=help_txt,
    )

    # Try to parse the input, with clamping and casting
    try:
        val = safe_eval_number(raw)

        # Clamp if requested
        if min_v is not None:
            val = max(min_v, val)
        if max_v is not None:
            val = min(max_v, val)

        # Store normalized display back into session (clean it up)
        if cast is int:
            val = int(round(val))
            st.session_state[key] = str(val)
            st.session_state[f"{key}_input"] = st.session_state[key]
            return val

        val = float(val)
        st.session_state[key] = str(val)
        st.session_state[f"{key}_input"] = st.session_state[key]
        return val

    # If parsing fails, warn the user and keep the previous value
    except Exception:
        # Keep last known-good value
        try:
            return cast(float(st.session_state[key]))
        except Exception:
            return cast(default)
        

# ---------------------------------------------------------------------------
# Public aliases (kept for readability in higher-level modules)
# NOTE:
# - We keep the underscore-prefixed functions as the canonical implementation.
# - These aliases let higher-level modules import without "private" naming.
# ---------------------------------------------------------------------------

text_num = _text_num