"""
app_streamlit.py

Streamlit app for training and testing RC Agents.

Copyright (c) 2026 Michael Garcia, M&E Design
https://mandedesign.studio
michael@mandedesign.studio

CSC370 Spring 2026
"""
from __future__ import annotations

import streamlit as st # Streamlit API; most "st.*" calls render UI or manage app state.
import pandas as pd # Pandas; used for DataFrame tables in the UI.
import matplotlib.pyplot as plt # Used for ploting q-table
import numpy as np

import ast
import operator as op

# Safe, tiny expression evaluator (supports numbers + + - * / // % ** and parentheses)
_ALLOWED_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}

# Parse and evaluate simple numeric expressions (e.g. "20*20*3", "1e-2", "(5+5)*10")
def _safe_eval_number(expr: str) -> float: # FLoat the result of the expr (expression)
    """
    Evaluate a numeric expression safely.
    Examples: "1500", "20*20*3", "1e-2", "(5+5)*10"
    """
    expr = expr.strip()
    if expr == "":
        raise ValueError("empty")

    node = ast.parse(expr, mode="eval")

    # Evaluate the expression, and return the resul
    def _eval(n):
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _ALLOWED_OPS:
            return _ALLOWED_OPS[type(n.op)](_eval(n.operand))
        if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED_OPS:
            return _ALLOWED_OPS[type(n.op)](_eval(n.left), _eval(n.right))
        raise ValueError("unsupported expression")

    return float(_eval(node))

# Text input field that parses into a number, with optional clamping and step hint.
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
    """
    if key not in st.session_state:
        st.session_state[key] = str(default)

    help_txt = "Examples: 1500, 20*20*3, 1e-2"
    if step_hint:
        help_txt += f" | Hint: {step_hint}"

    # The actual input field
    raw = st.text_input(label, value=st.session_state[key], key=f"{key}_input", help=help_txt)

    # Try to parse the input, with clamping and casting
    try:
        val = _safe_eval_number(raw)
        if min_v is not None:
            val = max(min_v, val)
        if max_v is not None:
            val = min(max_v, val)

        # Store normalized display back into session (clean it up)
        if cast is int:
            val = int(round(val))
            st.session_state[key] = str(val)
        else:
            val = float(val)
            st.session_state[key] = str(val)

        return val
    # If parsing fails, warn the user and keep the previous value
    except Exception:
        st.warning(f"{label}: couldn't parse '{raw}'. Keeping {st.session_state[key]}.")
        # Return last known-good value
        try:
            return cast(float(st.session_state[key]))
        except Exception:
            return cast(default)

# Uses the package imports from __init__ packages.
from rc_agents.ui.viz import q_table_to_matrix, state_value_grid, greedy_policy_grid
from rc_agents.edge_ai.rcg_edge.agents import Action, QAgent
from rc_agents.edge_ai.rcg_edge.runners import run_training
from rc_agents.envs import GridEnv
from rc_agents.config import TrainingUIConfig

st.set_page_config(page_title="RC Agents Trainer", layout="wide")

st.title("CSC370 Q-Learning Trainer (Streamlit)")

# TODO: Add a button to save the trained agent's Q-table to disk.
# TODO: Add a button to load a pre-trained agent from disk.

# Side-bar configuration.
# I chose to use configurable interfaces for clarity and easy reading.
def sidebar_config() -> TrainingUIConfig:
    cfg = TrainingUIConfig()

    with st.sidebar:
        st.header("Training Settings")

        # Use text input for sidebar values to allow for easy editing.
        # This is a bit of a hack, but it's the only way to get the
        # text input to update the value in the config object
        cfg.episodes = _text_num("Episodes", cfg.episodes, min_v=1, max_v=5000, cast=int, key="episodes")
        cfg.max_steps = _text_num("Max steps", cfg.max_steps, min_v=1, max_v=5000, cast=int, key="max_steps")

        cfg.epsilon = _text_num("Epsilon", cfg.epsilon, min_v=0.0, max_v=1.0, cast=float, key="epsilon", step_hint="0.0–1.0")
        cfg.alpha   = _text_num("Alpha",   cfg.alpha,   min_v=0.0, max_v=1.0, cast=float, key="alpha",   step_hint="0.0–1.0")
        cfg.gamma   = _text_num("Gamma",   cfg.gamma,   min_v=0.0, max_v=1.0, cast=float, key="gamma",   step_hint="0.0–1.0")

        cfg.rows = _text_num("Grid rows", cfg.rows, min_v=2, max_v=200, cast=int, key="rows")
        cfg.cols = _text_num("Grid cols", cfg.cols, min_v=2, max_v=200, cast=int, key="cols")

        # Default goal is bottom-right corner
        cfg.goal = (int(cfg.rows) - 1, int(cfg.cols) - 1)

        # Reset button for the agent.
        if st.sidebar.button("Reset Agent (clear Q-table)"):
            st.session_state.agent = None
            st.session_state.agent_key = None
            st.sidebar.success("Agent reset.")

        st.subheader("Save / Load")

        # Download learned Q-table
        if st.session_state.get("agent") is not None:
            st.download_button(
                "Download Q-table (.npz)",
                data=st.session_state.agent.to_bytes(),
                file_name="q_table.npz",
                mime="application/octet-stream",
            )
        else:
            st.caption("Train first to enable download.")

        # Upload learned Q-table
        uploaded = st.file_uploader("Load Q-table (.npz)", type=["npz"])
        if uploaded is not None:
            st.session_state.agent = QAgent.from_bytes(uploaded.read(), seed=cfg.seed)
            # Set grid to current UI grid (transfer will handle changes on next run)
            st.session_state.agent_grid = (int(cfg.rows), int(cfg.cols))
            st.success("Loaded Q-table into agent.")

    return cfg


cfg = sidebar_config()

# Persist a single agent across Streamlit reruns (progressive learning).
if "agent" not in st.session_state:
    st.session_state.agent = None

# Optional: track the “context” so we can reset when the grid or hyperparams change.
if "agent_key" not in st.session_state:
    st.session_state.agent_key = None

if "agent_grid" not in st.session_state:
    st.session_state.agent_grid = None

# If the agent has changed, reset i
def _agent_key(cfg: TrainingUIConfig) -> tuple:
    return (
        float(cfg.alpha),
        float(cfg.gamma),
        float(cfg.epsilon),
        int(cfg.seed or 0),
    )

def _transfer_q_table(old_agent: QAgent, new_agent: QAgent, rows: int, cols: int) -> None:
    """
    Copy overlapping learned Q-values from old_agent into new_agent.

    Keeps any state (r,c) where 0<=r<rows and 0<=c<cols.
    New states remain uninitialized (lazy zeros on first visit).
    """
    for k, v in old_agent.q_table.items():
        if isinstance(k, tuple) and len(k) == 2:
            r, c = int(k[0]), int(k[1])
            if 0 <= r < rows and 0 <= c < cols:
                new_agent.q_table[(r, c)] = np.asarray(v, dtype=float).copy()

# Main panel logic
run = st.button("Run Training", type="primary")

# when "Run Training" is pressed it
# runs the training loop and displays results.
# In order:
# 1. Create environment and agent
# 2. Run training loop
# 3. Display results in a few different ways (summary, q-table, value grid, policy grid)
# At the bottom is a UX improvement that limits visual clutter based on grid size.

if run:
    env = GridEnv(cfg.to_grid_config())

    key = _agent_key(cfg)

    # Create first agent if missing
    if st.session_state.agent is None or st.session_state.agent_key != key:
        st.session_state.agent = QAgent(cfg.to_q_config(), seed=cfg.seed)
        st.session_state.agent_key = key
        st.session_state.agent_grid = (int(cfg.rows), int(cfg.cols))

    # Transfer Q-table when grid size changes
    old_grid = st.session_state.get("agent_grid")
    new_rows, new_cols = int(cfg.rows), int(cfg.cols)

    if not (isinstance(old_grid, tuple) and len(old_grid) == 2): # If the grid doesn't change
        old_grid = (new_rows, new_cols)

    old_rows, old_cols = old_grid

    # Transfer whenever grid changes (or restrict to grow-only if you prefer)
    if (new_rows, new_cols) != (old_rows, old_cols):
        old_agent = st.session_state.agent
        new_agent = QAgent(cfg.to_q_config(), seed=cfg.seed)

        _transfer_q_table(old_agent, new_agent, rows=new_rows, cols=new_cols)

        st.session_state.agent = new_agent
        st.session_state.agent_grid = (new_rows, new_cols)

        st.info(f"Transferred Q-table from {old_rows}x{old_cols} to {new_rows}x{new_cols}.")

    agent = st.session_state.agent

    results = run_training(env=env, agent=agent, cfg=cfg)

    # Minimal results display
    wins = sum(1 for r in results if r.reached_goal) # reached_goal is defined in train_runner>run_training
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
                    va="center"
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
                    va="center"
                )

    st.pyplot(fig)