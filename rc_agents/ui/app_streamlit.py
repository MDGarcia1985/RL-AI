"""
app_streamlit.py

Streamlit app for training and testing RC Agents.

Copyright (c) 2026 Michael Garcia, M&E Design
https://mandedesign.studio
michael@mandedesign.studio

CSC370 Spring 2026
"""
from __future__ import annotations

import streamlit as st # don't miss this. Everything with the "st." prefix is doing work for streamlit
import pandas as pd # pd is pandas, used for dataframes
import matplotlib.pyplot as plt
import numpy as np

from rc_agents.ui.viz.q_table_viz import q_table_to_matrix, state_value_grid, greedy_policy_grid
from rc_agents.edge_ai.rcg_edge.agents.base import Action
from rc_agents.envs.grid_env import GridEnv
from rc_agents.edge_ai.rcg_edge.agents.q_agent import QAgent
from rc_agents.edge_ai.rcg_edge.runners.train_runner import run_training
from rc_agents.config.ui_config import TrainingUIConfig

st.set_page_config(page_title="RC Agents Trainer", layout="wide")

st.title("CSC370 Q-Learning Trainer (Streamlit)")


def sidebar_config() -> TrainingUIConfig:
    cfg = TrainingUIConfig()

    with st.sidebar:
        st.header("Training Settings")

        cfg.episodes = st.number_input("Episodes", 1, 5000, cfg.episodes, 1)
        cfg.max_steps = st.number_input("Max steps", 1, 5000, cfg.max_steps, 1)

        cfg.epsilon = st.number_input("Epsilon", 0.0, 1.0, cfg.epsilon, 0.01, format="%.2f")
        cfg.alpha = st.number_input("Alpha", 0.0, 1.0, cfg.alpha, 0.01, format="%.2f")
        cfg.gamma = st.number_input("Gamma", 0.0, 1.0, cfg.gamma, 0.01, format="%.2f")

        cfg.rows = st.number_input("Grid rows", 2, 200, cfg.rows, 1)
        cfg.cols = st.number_input("Grid cols", 2, 200, cfg.cols, 1)

        # Default goal is bottom-right corner
        cfg.goal = (int(cfg.rows) - 1, int(cfg.cols) - 1)

    return cfg


cfg = sidebar_config()

run = st.button("Run Training", type="primary")

if run:
    env = GridEnv(cfg.to_grid_config())
    agent = QAgent(cfg.to_q_config(), seed=cfg.seed)

    results = run_training(env=env, agent=agent, cfg=cfg)

    # Minimal results display
    wins = sum(1 for r in results if r.reached_goal)
    st.success(f"Reached goal: {wins}/{len(results)}")

    st.write("Last 10 episodes:")
    st.write([r.__dict__ for r in results[-10:]])

    rows_i, cols_i = int(cfg.rows), int(cfg.cols)

    # Q-table display
    Q = q_table_to_matrix(agent.q_table, rows_i, cols_i)
    actions = [a.name for a in Action]

    st.subheader("Learned Q-table (dense view)")
    state_ids = [f"({r},{c})" for r in range(int(cfg.rows)) for c in range(cols_i)]
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
                    va="center"
                )

    # Map Action ordering to arrows.
    # list(Action) == [FORWARD, BACKWARD, RIGHT, LEFT]
    arrow_by_index = {
        0: "↑",  # FORWARD
        1: "↓",  # BACKWARD
        2: "→",  # RIGHT
        3: "←",  # LEFT
    }

    # Draw policy arrows only when grid is readable
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
                    va="center"
                )

    st.pyplot(fig)