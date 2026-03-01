"""
main_panel.py

Main panel renderer for Streamlit trainer.

Design intent:
- Keep UI readable:
  - scoreboard summary first
  - expandable details second
  - visualization last
- Provide an "exhaustive download" bundle for engineering logs.

Copyright (c) 2026 Michael Garcia, M&E Design
https://mandedesign.studio
CSC370 Spring 2026
"""
from __future__ import annotations

import json
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from rc_agents.edge_ai.rcg_edge.agents import Action
from rc_agents.ui.viz import greedy_policy_grid, q_table_to_matrix, state_value_grid

from .progressive_learning import ensure_state, run_selected_agents


def render_main_panel(*, cfg, game_type: str, selected_agent_ids: List[str]) -> None:
    """
    Render main panel: run button, summary, details, downloads, viz.

    Why:
    - app_streamlit.py should be a thin router.
    - all UI logic lives here.
    """
    ensure_state()

    run = st.button("Run Training", type="primary")

    if not run:
        # Show last summary if it exists (prevents "blank UI" syndrome).
        if st.session_state.get("last_scoreboard"):
            st.subheader("Last agent summary performance")
            st.dataframe(pd.DataFrame(st.session_state.last_scoreboard), use_container_width=True)
        else:
            st.caption("Configure settings and click Run Training.")
        return

    if not selected_agent_ids:
        st.warning("No agents selected.")
        return

    scoreboard, details = run_selected_agents(
        cfg=cfg,
        game_type=game_type,
        selected_agent_ids=selected_agent_ids,
    )

    # ------------------------------------------------------------
    # Summary performance table
    # ------------------------------------------------------------
    st.subheader("Agent summary performance")
    st.dataframe(pd.DataFrame(scoreboard), use_container_width=True)

    # ------------------------------------------------------------
    # Exhaustive download bundle (engineering log export)
    # ------------------------------------------------------------
    bundle_json = json.dumps(details, indent=2, sort_keys=False).encode("utf-8")
    st.download_button(
        "Download results bundle (JSON)",
        data=bundle_json,
        file_name="rc_agents_results_bundle.json",
        mime="application/json",
    )

    # ------------------------------------------------------------
    # Per-agent details (expanders keep UI calm)
    # ------------------------------------------------------------
    st.subheader("Agent details")
    for agent_id in selected_agent_ids:
        with st.expander(f"Details: {agent_id}", expanded=False):
            st.write(details.get(agent_id, {}))

    # ------------------------------------------------------------
    # Visualization: show the last trained agent’s Q-table/heatmap
    #
    # Why:
    # - You still want the classic heatmap view while comparing agents.
    # - We keep it simple: visualize last selected agent from this run.
    #
    # Future:
    # - Add a dropdown: "which agent to visualize?"
    # - Add animation of best run once trajectory logging exists.
    # ------------------------------------------------------------
    last_agent_id = selected_agent_ids[-1]
    agent_payload = details.get(last_agent_id, {})
    st.subheader(f"Visualization — {agent_payload.get('agent', last_agent_id)}")

    # We need the actual agent object for q_table access.
    # The runner stores agents in session under progressive_learning.
    # For now, easiest approach is: reuse the current session agent_store.
    # (If you later want strict isolation, you can also include q_table in the bundle.)
    #
    # NOTE:
    # We reconstruct the q_table by reading it from the active agent instance.
    ctx_key = tuple(agent_payload.get("context_key", []))
    store_key = (last_agent_id, ctx_key)
    agent_store = st.session_state.get("agent_store", {})

    agent = agent_store.get(store_key)
    if agent is None:
        st.warning("Agent not found in session_state for visualization (try rerun).")
        return

    rows_i, cols_i = int(cfg.rows), int(cfg.cols)

    # Q-table display
    Q = q_table_to_matrix(agent.q_table, rows_i, cols_i)
    actions = [a.name for a in Action]

    st.subheader("Learned Q-table (dense view)")
    state_ids = [f"({r},{c})" for r in range(rows_i) for c in range(cols_i)]
    df_q = pd.DataFrame(Q, columns=actions)
    df_q.insert(0, "state", state_ids)
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

    # ticks/grid only when readable
    if rows_i <= 30 and cols_i <= 30:
        ax.set_xticks(np.arange(cols_i))
        ax.set_yticks(np.arange(rows_i))
        ax.grid(True)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    # Decide when to annotate numeric values
    show_cell_text = (rows_i <= 12 and cols_i <= 12)

    # Annotate numeric V(s) values for small grids
    if show_cell_text:
        for r in range(rows_i):
            for c in range(cols_i):
                ax.text(
                    c,
                    r,
                    f"{vgrid[r, c]:.2f}",
                    ha="center",
                    va="center",
                )

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
    arrow_by_index = {
        0: "↑",  # FORWARD
        1: "↓",  # BACKWARD
        2: "→",  # RIGHT
        3: "←",  # LEFT
    }

    # Draw policy arrows only when grid is readable (less than 30x30 grid cells)
    # AND not already showing numeric values
    show_policy = (rows_i <= 30 and cols_i <= 30 and not show_cell_text)

    if show_policy:
        for r in range(rows_i):
            for c in range(cols_i):
                ax.text(
                    c,
                    r,
                    arrow_by_index.get(int(policy[r, c]), "?"),
                    ha="center",
                    va="center",
                )

    st.pyplot(fig)