"""
sidebar_ui.py

Sidebar controls for Streamlit trainer.

Design intent:
- Sidebar defines the "experiment settings" (env + agents + hyperparams).
- Main panel runs training + displays results.
- Keep state stable across reruns.

Copyright (c) 2026 Michael Garcia, M&E Design
https://mandedesign.studio
CSC370 Spring 2026
"""
from __future__ import annotations

import streamlit as st

from rc_agents.config import TrainingUIConfig

from .coordinates import text_coord
from .text_num import text_num


def sidebar_config() -> tuple[TrainingUIConfig, str, list[str]]:
    """
    Build UI config + selection state.

    Returns:
        cfg: TrainingUIConfig
        game_type: str ("Open World", "Maze", ...)
        selected_agent_ids: list[str] e.g. ["rl", "rlf"]
    """
    cfg = TrainingUIConfig()

    with st.sidebar:
        st.header("Game Settings")

        game_type = st.selectbox(
            "Game type",
            options=["Open World", "Maze", "Labyrinth (future)"],
            index=0,
        )

        st.subheader("Agent selection")
        run_rl = st.checkbox("RL Base Agent", value=True)
        run_rlf = st.checkbox("RL with Fractal Exploration (RLF)", value=False)

        selected_agent_ids: list[str] = []
        if run_rl:
            selected_agent_ids.append("rl")
        if run_rlf:
            selected_agent_ids.append("rlf")

        if not selected_agent_ids:
            st.warning("Select at least one agent.")

        st.divider()
        st.header("Training Settings")

        # Episodes and max_steps define how much search is even possible.
        cfg.episodes = text_num("Episodes", cfg.episodes, min_v=1, max_v=5000, cast=int, key="episodes")
        cfg.max_steps = text_num("Max steps", cfg.max_steps, min_v=1, max_v=20000, cast=int, key="max_steps")

        # Hyperparams
        cfg.epsilon = text_num("Epsilon", cfg.epsilon, min_v=0.0, max_v=1.0, cast=float, key="epsilon", step_hint="0.0–1.0")
        cfg.alpha = text_num("Alpha", cfg.alpha, min_v=0.0, max_v=1.0, cast=float, key="alpha", step_hint="0.0–1.0")
        cfg.gamma = text_num("Gamma", cfg.gamma, min_v=0.0, max_v=1.0, cast=float, key="gamma", step_hint="0.0–1.0")

        # Grid geometry
        cfg.rows = text_num("Grid rows", cfg.rows, min_v=2, max_v=200, cast=int, key="rows")
        cfg.cols = text_num("Grid cols", cfg.cols, min_v=2, max_v=200, cast=int, key="cols")

        rows_i, cols_i = int(cfg.rows), int(cfg.cols)

        # Start and goal (coordinate inputs)
        #
        # Why:
        # - Required for large grids (goal not always bottom-right)
        # - Required for mazes (goal can be deep in structure)
        # - Required for future corner-start tournaments
        cfg.start = text_coord("Start (row, col)", default=(0, 0), rows=rows_i, cols=cols_i, key="start_coord")
        cfg.goal = text_coord("Goal (row, col)", default=(rows_i - 1, cols_i - 1), rows=rows_i, cols=cols_i, key="goal_coord")

    return cfg, game_type, selected_agent_ids