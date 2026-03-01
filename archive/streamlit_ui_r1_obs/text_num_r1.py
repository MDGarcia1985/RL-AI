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


def text_num(
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

    Behavior:
    - Parses numbers and small expressions via safe_eval_number()
    - Clamps if min_v/max_v are provided
    - If parsing fails, returns prior value (or default if state is broken)
    """
    if key not in st.session_state:
        st.session_state[key] = str(default)
    if f"{key}_input" not in st.session_state:
        st.session_state[f"{key}_input"] = st.session_state[key]

    help_txt = "Examples: 1500, 20*20*3, 1e-2"
    if step_hint:
        help_txt += f" | Hint: {step_hint}"

    raw = st.text_input(
        label,
        value=st.session_state[f"{key}_input"],
        key=f"{key}_input",
        help=help_txt,
    )

    try:
        val = safe_eval_number(raw)

        # Clamp if requested (prevents illegal env/config values).
        if min_v is not None:
            val = max(min_v, val)
        if max_v is not None:
            val = min(max_v, val)

        # Normalize display back into session (clean output).
        if cast is int:
            val = int(round(val))
            st.session_state[key] = str(val)
            st.session_state[f"{key}_input"] = st.session_state[key]
            return val

        val = float(val)
        st.session_state[key] = str(val)
        st.session_state[f"{key}_input"] = st.session_state[key]
        return val

    except Exception:
        # Silent fallback to last-known-good value.
        try:
            return cast(float(st.session_state[key]))
        except Exception:
            return cast(default)