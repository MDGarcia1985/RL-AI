"""
main_panel.py

Main panel renderer for Streamlit trainer.

Design intent:
- Keep UI readable: scoreboard summary first, expandable details second, visualization last.
- Separate panels per agent type: summary (wins/steps) + heat maps (value grid, policy).
- Trail graph for the best maze run (single best trajectory across all runs).
- Environment and agents come from sidebar (dropdown + checkboxes); build via factory.
- Provide an "exhaustive download" bundle for engineering logs (future).
- At the bottom, limit visual clutter based on grid size (ticks/annotations only when readable).

Copyright (c) 2026 Michael Garcia, M&E Design
https://mandedesign.studio
CSC370 Spring 2026
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from rc_agents.edge_ai.rcg_edge.agents import Action
from rc_agents.edge_ai.rcg_edge.runners import run_training
from rc_agents.ui.viz import greedy_policy_grid, plot_trail, q_table_to_matrix, state_value_grid

from .agent_catalog import build_agent_catalog
from .factory import make_agent, make_env
from .progressive_learning import ensure_state, agent_key, transfer_q_table

# Map Action ordering to arrows.
# list(Action) == [FORWARD, BACKWARD, RIGHT, LEFT]
#
# NOTE on future refactor (diagonal / half-step actions):
# - If diagonal actions are added (e.g., NW, NE, SW, SE), this mapping
#   must be extended with additional direction symbols.
# - Avoid relying on hard-coded indices; prefer resolving:
#       index -> Action enum -> visual symbol
# - Recommended refactor path:
#     * Extend the Action enum with diagonal directions.
#     * Replace index-based mapping with a dict keyed by Action name
#       (e.g., "NORTHWEST": "↖", "NORTHEAST": "↗", etc.).
#     * If GPS or heading-based guidance is added, compass directions
#       should become explicit rather than inferred.
# - This prevents silent misalignment if Action ordering changes.
_ARROW_BY_INDEX = {
    0: "↑",  # FORWARD
    1: "↓",  # BACKWARD
    2: "→",  # RIGHT
    3: "←",  # LEFT
}


def _render_agent_summary(results: List[Any], agent_label: str) -> None:
    """
    Write summary metrics for one agent (wins, best steps, last episodes).
    reached_goal is defined in train_runner > run_training.
    """
    wins = sum(1 for r in results if r.reached_goal)
    st.metric("Reached goal", f"{wins}/{len(results)}")
    successful = [r for r in results if r.reached_goal]
    if successful:
        min_steps = min(r.steps for r in successful)
        st.caption(f"Best run: {min_steps} steps")
    st.write("Last 10 episodes:")
    st.json([r.__dict__ for r in results[-10:]])


def _render_heatmaps(agent: Any, rows_i: int, cols_i: int) -> None:
    """
    Render state-value heatmap and policy grid if agent has q_table.
    We use getattr so we support any agent that exposes .q_table (e.g. QAgent, RLAgent, RLFAgent)
    without requiring a common base; agents without q_table get a short caption instead.
    """
    if not getattr(agent, "q_table", None):
        st.caption("No Q-table (agent does not expose heat maps).")
        return

    # Q-table display (dense view)
    Q = q_table_to_matrix(agent.q_table, rows_i, cols_i)
    actions = [a.name for a in Action]
    state_ids = [f"({r},{c})" for r in range(rows_i) for c in range(cols_i)]
    df_q = pd.DataFrame(Q, columns=actions)
    df_q.insert(0, "state", state_ids)
    st.subheader("Learned Q-table (dense view)")
    st.dataframe(df_q, use_container_width=True)

    # State-value heatmap (V(s) = max_a Q(s,a))
    st.subheader("State-value grid (V(s) = max_a Q(s,a))")
    vgrid = state_value_grid(agent.q_table, rows_i, cols_i)
    policy = greedy_policy_grid(agent.q_table, rows_i, cols_i)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(vgrid, interpolation="nearest")
    fig.colorbar(im, ax=ax, label="V(s)")
    ax.set_title("Learned Values per Grid Cell")
    # Align plot coordinates with (row, col) grid indexing (row 0 at top).
    ax.invert_yaxis()

    # Ticks/grid only when readable
    if rows_i <= 30 and cols_i <= 30:
        ax.set_xticks(np.arange(cols_i))
        ax.set_yticks(np.arange(rows_i))
        ax.grid(True)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    # Decide when to annotate numeric values
    show_cell_text = rows_i <= 12 and cols_i <= 12
    # Annotate numeric V(s) values for small grids
    if show_cell_text:
        for r in range(rows_i):
            for c in range(cols_i):
                ax.text(c, r, f"{vgrid[r, c]:.2f}", ha="center", va="center")

    # Draw policy arrows only when grid is readable (less than 30x30 grid cells)
    # AND not already showing numeric values
    show_policy = rows_i <= 30 and cols_i <= 30 and not show_cell_text
    if show_policy:
        for r in range(rows_i):
            for c in range(cols_i):
                ax.text(c, r, _ARROW_BY_INDEX.get(int(policy[r, c]), "?"), ha="center", va="center")

    st.pyplot(fig)


def render_main_panel(*, cfg, game_type: str, selected_agent_ids: List[str]) -> None:
    """
    Main panel entry point.

    app_streamlit.py is the router:
        cfg, game_type, selected_agent_ids = sidebar_config()
        render_main_panel(cfg=cfg, game_type=game_type, selected_agent_ids=selected_agent_ids)

    When "Run Training" is pressed:
    1. Create environment (from game_type) and agents (from selected_agent_ids)
    2. Run training loop per agent
    3. Display results: separate panel per agent (summary, q-table, value grid, policy grid)
    4. At the bottom: trail graph for the best maze run (fewest steps to goal)
    """
    ensure_state()

    run = st.button("Run Training", type="primary")
    if not run:
        return

    if not selected_agent_ids:
        st.warning("Select at least one agent in the sidebar.")
        return

    # Build env once from sidebar selection (dropdown)
    env = make_env(cfg, game_type=game_type)
    rows_i = int(cfg.rows)
    cols_i = int(cfg.cols)
    catalog = build_agent_catalog()

    # Optional: get walls for maze trail viz. MazeEnv has .walls; GridEnv does not.
    # plot_trail uses walls to draw the maze background when present.
    walls: Optional[np.ndarray] = getattr(env, "walls", None)

    # Across all selected agents we keep the single best trajectory (fewest steps to goal)
    # so the "Best run (trail)" section shows one graph for the best run from any agent.
    best_trajectory_overall: Optional[List[Tuple[int, int]]] = None
    best_steps_overall: Optional[int] = None

    # Progressive learning: reuse cached agent when key and grid match; transfer Q-table when grid changes.
    key = agent_key(cfg)
    agent_store = st.session_state.agent_store
    agent_key_store = st.session_state.agent_key_store
    agent_grid_store = st.session_state.agent_grid_store

    last_trained_agent_for_save: Any = None
    for agent_id in selected_agent_ids:
        spec = catalog.get(agent_id)
        label = spec.label if spec else agent_id

        cached = agent_store.get(agent_id)
        stored_key = agent_key_store.get(agent_id)
        stored_grid = agent_grid_store.get(agent_id)
        current_grid = (rows_i, cols_i)

        if cached is not None and getattr(cached, "q_table", None) is not None and stored_key == key and stored_grid == current_grid:
            agent = cached
        elif cached is not None and getattr(cached, "q_table", None) is not None and stored_key == key and stored_grid != current_grid:
            # Transfer only when hyperparameters (key) are unchanged; grid size changed.
            # If key changed too, we must not transfer (Q-values were learned under different alpha/gamma/epsilon/seed).
            agent = make_agent(agent_id, cfg)
            transfer_q_table(cached, agent, rows_i, cols_i)
            st.info(f"[{label}] Transferred Q-table from {stored_grid} to {current_grid}.")
            agent_store[agent_id] = agent
            agent_key_store[agent_id] = key
            agent_grid_store[agent_id] = current_grid
        else:
            agent = make_agent(agent_id, cfg)
            agent_store[agent_id] = agent
            agent_key_store[agent_id] = key
            agent_grid_store[agent_id] = current_grid

        with st.expander(f"**{label}**", expanded=True):
            results, best_trajectory = run_training(env=env, agent=agent, cfg=cfg)

            _render_agent_summary(results, label)
            st.divider()
            _render_heatmaps(agent, rows_i, cols_i)

            if getattr(agent, "to_bytes", None) is not None:
                last_trained_agent_for_save = agent

            # Track best trajectory across all agents (fewest steps that reached goal)
            if best_trajectory and results:
                successful = [r for r in results if r.reached_goal]
                if successful:
                    min_steps = min(r.steps for r in successful)
                    if best_steps_overall is None or min_steps < best_steps_overall:
                        best_steps_overall = min_steps
                        best_trajectory_overall = best_trajectory

    # After training: store last trained agent in session so sidebar "Download Q-table" appears
    if last_trained_agent_for_save is not None:
        st.session_state.agent = last_trained_agent_for_save
        st.session_state.agent_grid = (rows_i, cols_i)

    # Best run trail: one graph for the best trajectory from any agent (fewest steps to goal)
    st.divider()
    st.subheader("Best run (trail)")
    if best_trajectory_overall:
        fig = plot_trail(best_trajectory_overall, rows_i, cols_i, walls=walls)
        st.pyplot(fig)
    else:
        st.caption("No successful run to display. Run training with at least one agent that reaches the goal.")
