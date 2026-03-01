"""
progressive_learning.py

Shared helpers for:
- persistent agents across Streamlit reruns
- multi-agent training runs (RL + RLF)
- training context keys (prevents Q-table mixing)

Design intent:
- Streamlit reruns everything on any UI event.
- st.session_state is our RAM.
- We preserve progressive learning WITHOUT silently cross-contaminating agents.

Copyright (c) 2026 Michael Garcia, M&E Design
https://mandedesign.studio
CSC370 Spring 2026
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from rc_agents.config import TrainingUIConfig


# ---------------------------------------------------------------------------
# Progressive learning: persist agent across Streamlit reruns
#
# Streamlit reruns the script top-to-bottom on every UI interaction.
# st.session_state is our "memory" so training can be progressive across runs.
# ---------------------------------------------------------------------------

def ensure_state() -> None:
    """
    Ensure Streamlit session storage exists.

    Why:
    - Prevent KeyError and keep app deterministic.
    - Gives you clean named buckets for debugging.
    """
    if "agent" not in st.session_state:
        st.session_state.agent = None
    # Optional: track the “context” so we can reset when the grid or hyperparams change.
    if "agent_key" not in st.session_state:
        st.session_state.agent_key = None
    # Track the grid size associated with the current Q-table (used for transfer logic).
    if "agent_grid" not in st.session_state:
        st.session_state.agent_grid = None

    # Per-agent cache for progressive learning (key/grid so we know when to reuse or transfer)
    if "agent_store" not in st.session_state:
        st.session_state.agent_store = {}
    if "agent_key_store" not in st.session_state:
        st.session_state.agent_key_store = {}
    if "agent_grid_store" not in st.session_state:
        st.session_state.agent_grid_store = {}
    if "history_store" not in st.session_state:
        st.session_state.history_store = {}
    if "last_scoreboard" not in st.session_state:
        st.session_state.last_scoreboard = []


def _agent_key(cfg: TrainingUIConfig) -> tuple:
    """
    Defines the agent identity context.

    If these change, we treat it as a new training context and create a fresh agent.
    This prevents silently mixing Q-tables across incompatible hyperparameters.
    """
    return (
        float(cfg.alpha),
        float(cfg.gamma),
        float(cfg.epsilon),
        int(cfg.seed or 0),
    )


def _transfer_q_table(old_agent: object, new_agent: object, rows: int, cols: int) -> None:
    """
    Copy overlapping learned Q-values from old_agent into new_agent.

    Keeps any state (r,c) where 0<=r<rows and 0<=c<cols.
    New states remain uninitialized (lazy zeros on first visit).
    Works with any agent that has .q_table (QAgent, RLAgent, RLFAgent).

    Design intent:
    - Allows progressive learning when scaling the environment size.
    - Avoids throwing away knowledge on small-to-large transitions.
    """
    if not hasattr(old_agent, "q_table") or not hasattr(new_agent, "q_table"):
        return
    for k, v in old_agent.q_table.items():
        if isinstance(k, tuple) and len(k) == 2:
            r, c = int(k[0]), int(k[1])
            if 0 <= r < rows and 0 <= c < cols:
                new_agent.q_table[(r, c)] = np.asarray(v, dtype=float).copy()


# ---------------------------------------------------------------------------
# Public aliases (kept for readability in higher-level modules)
# NOTE:
# - We keep the underscore-prefixed functions as the canonical implementation.
# - These aliases let higher-level modules import without "private" naming.
# ---------------------------------------------------------------------------

agent_key = _agent_key
transfer_q_table = _transfer_q_table