"""
sidebar_ui.py

Sidebar controls for Streamlit trainer.

Design intent:
- Sidebar defines the "experiment settings" (env + agents + hyperparams).
- Environment is selected via dropdown; agents are selected via checkboxes
  (one per entry in the factory agent catalog).
- Main panel runs training + displays results.
- Keep state stable across reruns.

Copyright (c) 2026 Michael Garcia, M&E Design
https://mandedesign.studio
CSC370 Spring 2026
"""
from __future__ import annotations

from typing import List

import streamlit as st

from rc_agents.config import TrainingUIConfig

from .coordinates import text_coord
from .text_num import text_num
from .factory import get_env_options
from .agent_catalog import build_agent_catalog


# ---------------------------------------------------------------------------
# Sidebar UI
# ---------------------------------------------------------------------------


def sidebar_config() -> tuple[TrainingUIConfig, str, List[str]]:
    """
    Build sidebar UI and return (cfg, game_type, selected_agent_ids).

    - cfg: training and grid/maze parameters.
    - game_type: selected environment (e.g. "Open World", "Maze").
    - selected_agent_ids: list of agent_id for agents that are checked.
    """
    cfg = TrainingUIConfig()

    with st.sidebar:
        st.header("Training Settings")

        # ------------------------------------------------------------------
        # Environment selection (dropdown)
        # Refactor: environments are selected via dropdown; value is passed to
        # make_env(cfg, game_type=...) in main_panel. Options come from factory.
        # ------------------------------------------------------------------
        env_options = get_env_options()  # (value, label) for selectbox
        env_labels = [label for _value, label in env_options]
        env_values = [value for value, _label in env_options]
        selected_label = st.selectbox(
            "Environment",
            options=env_labels,
            key="sidebar_env_select",
        )
        selected_idx = env_labels.index(selected_label)
        game_type = env_values[selected_idx]  # e.g. "Open World", "Maze"

        # ------------------------------------------------------------------
        # Agent checkboxes (one per catalog entry)
        # Refactor: as agents are added to the factory (agent_catalog), each
        # gets a checkbox here. selected_agent_ids is passed to main_panel so
        # it can run training and render a panel per selected agent.
        # ------------------------------------------------------------------
        st.subheader("Agents")
        catalog = build_agent_catalog()
        selected_agent_ids: List[str] = []
        for agent_id, spec in catalog.items():
            checked = st.checkbox(spec.label, key=f"agent_cb_{agent_id}")
            if checked:
                selected_agent_ids.append(agent_id)
        if not selected_agent_ids:
            st.caption("Select at least one agent to run training.")

        st.divider()
        st.subheader("Hyperparameters")

        # Use text input for sidebar values to allow for easy editing.
        # This is a bit of a hack, but it's the only way to get the
        # text input to update the value in the config object.
        cfg.episodes = text_num("Episodes", cfg.episodes, min_v=1, max_v=5000, cast=int, key="episodes")
        cfg.max_steps = text_num("Max steps", cfg.max_steps, min_v=1, max_v=5000, cast=int, key="max_steps")

        cfg.epsilon = text_num(
            "Epsilon",
            cfg.epsilon,
            min_v=0.0,
            max_v=1.0,
            cast=float,
            key="epsilon",
            step_hint="0.0–1.0",
        )
        cfg.alpha = text_num(
            "Alpha",
            cfg.alpha,
            min_v=0.0,
            max_v=1.0,
            cast=float,
            key="alpha",
            step_hint="0.0–1.0",
        )
        cfg.gamma = text_num(
            "Gamma",
            cfg.gamma,
            min_v=0.0,
            max_v=1.0,
            cast=float,
            key="gamma",
            step_hint="0.0–1.0",
        )

        cfg.rows = text_num("Grid rows", cfg.rows, min_v=2, max_v=200, cast=int, key="rows")
        cfg.cols = text_num("Grid cols", cfg.cols, min_v=2, max_v=200, cast=int, key="cols")

        # Start and goal coordinates
        #
        # Design intent:
        # - Make the destination configurable for large grids (e.g., 60x60 -> (45, 56))
        # - Prepare for mazes and multi-corner starts later
        # - Keep coords clamped so the env never gets invalid indices
        #
        # NOTE:
        # - We set sensible defaults the first time.
        # - The text input allows "(r,c)" or "r,c" or "r c".
        rows_i, cols_i = int(cfg.rows), int(cfg.cols)

        # Default start is top-left; you can move it later for corner tests.
        cfg.start = text_coord(
            "Start (row, col)",
            default=(0, 0),
            rows=rows_i,
            cols=cols_i,
            key="start_coord",
        )

        # Default goal is bottom-right unless user overrides.
        cfg.goal = text_coord(
            "Goal (row, col)",
            default=(rows_i - 1, cols_i - 1),
            rows=rows_i,
            cols=cols_i,
            key="goal_coord",
        )

        # Reset button for agents (clear all cached agents).
        # Refactor: with multi-agent UI we clear any agent-related session keys
        # (agent, agent_key, agent_grid) so the next run builds fresh agents.
        # NOTE: Keys like agent_cb_* are Streamlit widget keys; we only clear
        # our own agent state to avoid breaking the sidebar.
        if st.button("Reset Agents (clear Q-tables)"):
            for key in list(st.session_state.keys()):
                if key.startswith("agent_") and key != "agent_key":
                    del st.session_state[key]
            if "agent" in st.session_state:
                st.session_state.agent = None
            if "agent_key" in st.session_state:
                st.session_state.agent_key = None
            if "agent_grid" in st.session_state:
                st.session_state.agent_grid = None
            st.success("Agents reset.")

        st.divider()
        st.subheader("Save / Load")

        # Download learned Q-table (only if agent exists and supports to_bytes)
        agent = st.session_state.get("agent")
        if agent is not None and hasattr(agent, "to_bytes"):
            st.download_button(
                "Download Q-table (.npz)",
                data=agent.to_bytes(),
                file_name="q_table.npz",
                mime="application/octet-stream",
            )
        else:
            st.caption("Train first to enable download.")

        # Upload learned Q-table
        uploaded = st.file_uploader("Load Q-table (.npz)", type=["npz"])
        if uploaded is not None:
            # Lazy import keeps sidebar stable if agents are in flux.
            from rc_agents.edge_ai.rcg_edge.agents import QAgent
            from .progressive_learning import agent_key

            st.session_state.agent = QAgent.from_bytes(uploaded.read(), seed=cfg.seed)
            st.session_state.agent_grid = (int(cfg.rows), int(cfg.cols))
            st.session_state.agent_key = agent_key(cfg)
            st.success("Loaded Q-table into agent.")

    # Return value is consumed by app_streamlit: render_main_panel(cfg, game_type, selected_agent_ids)
    return (cfg, game_type, selected_agent_ids)