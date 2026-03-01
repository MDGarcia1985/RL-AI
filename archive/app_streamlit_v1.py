"""
app_streamlit.py

Streamlit app for training and testing RC Agents.

Copyright (c) 2026 Michael Garcia, M&E Design
https://mandedesign.studio
michael@mandedesign.studio

CSC370 Spring 2026
"""
from __future__ import annotations

import logging

# Configure logging for the application.
# This sets up the root logger once so all module loggers inherit it.
# Guarded to avoid duplicate handlers in framework environments (Streamlit may preconfigure logging).
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,  # or DEBUG
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

import ast
import operator as op

import matplotlib.pyplot as plt  # Used for plotting q-table
import numpy as np
import pandas as pd  # Pandas; used for DataFrame tables in the UI.
import streamlit as st  # Streamlit API; most "st.*" calls render UI or manage app state.

# Uses the package imports from __init__ packages.
from rc_agents.config import TrainingUIConfig
from rc_agents.edge_ai.rcg_edge.agents import Action, QAgent
from rc_agents.edge_ai.rcg_edge.runners import run_training
from rc_agents.envs import GridEnv
from rc_agents.ui.viz import q_table_to_matrix, state_value_grid, greedy_policy_grid

st.set_page_config(page_title="RC Agents Trainer", layout="wide")
st.title("CSC370 Q-Learning Trainer (Streamlit)")

# ---------------------------------------------------------------------------
# Safe numeric text inputs (string -> number)
#
# Streamlit number_input is great, but sometimes you want to type:
#   "20*20*3" or "1e-2" or "(5+5)*10"
#
# We evaluate a tiny subset of math expressions safely using AST.
# This is intentionally restrictive: only numeric literals + basic arithmetic.
# ---------------------------------------------------------------------------

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
def _safe_eval_number(expr: str) -> float:  # Float the result of the expr (expression)
    """
    Evaluate a numeric expression safely.
    Examples: "1500", "20*20*3", "1e-2", "(5+5)*10"
    """
    expr = expr.strip()
    if expr == "":
        raise ValueError("empty")

    node = ast.parse(expr, mode="eval")

    # Evaluate the expression, and return the result
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
        val = _safe_eval_number(raw)

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
# Safe coordinate inputs (row, col)
#
# We want to be able to type:
#   "(45, 56)"  or "45,56"  or "45 56"
#
# This will matter more later when we add:
# - mazes
# - multiple start corners
# - agent-vs-agent tournaments
#
# Design intent:
# - Keep UI flexible for humans
# - Keep parsing safe and simple (no arbitrary eval)
# ---------------------------------------------------------------------------

def _clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _parse_coord(raw: str) -> tuple[int, int]:
    """
    Parse a coordinate string into (row, col) ints.

    Accepted formats:
    - "(45, 56)"
    - "45,56"
    - "45 56"

    NOTE:
    - This is intentionally strict: only two integers.
    - If you want expressions later (e.g., "60-1"), you can add it using _safe_eval_number().
    """
    s = raw.strip()
    if not s:
        raise ValueError("empty coordinate")

    # Strip optional parentheses
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()

    # Split by comma or whitespace
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
    else:
        parts = [p.strip() for p in s.split()]

    if len(parts) != 2:
        raise ValueError("coordinate must have exactly two values")

    r = int(parts[0]) # Row
    c = int(parts[1]) # Collum
    return (r, c)

# Coordinates entered by text string
def _text_coord(
    label: str,
    default: tuple[int, int],
    *,
    rows: int,
    cols: int,
    key: str,
) -> tuple[int, int]:
    """
    Sidebar text input -> (row, col) tuple.

    - Keeps prior value if parsing fails.
    - Clamps to [0..rows-1] and [0..cols-1] so the env never receives illegal coordinates.
    """
    if key not in st.session_state:
        st.session_state[key] = f"({default[0]}, {default[1]})"
    if f"{key}_input" not in st.session_state:
        st.session_state[f"{key}_input"] = st.session_state[key]

    # The actual input field
    raw = st.text_input(
        label,
        value=st.session_state[f"{key}_input"],
        key=f"{key}_input",
        help="Examples: (45, 56) | 45,56 | 45 56",
    )

    try:
        r, c = _parse_coord(raw) # Parse row, col from string

        # Clamp to valid grid bounds
        r = _clamp_int(r, 0, max(0, rows - 1))
        c = _clamp_int(c, 0, max(0, cols - 1))

        st.session_state[key] = f"({r}, {c})" # Update session state
        st.session_state[f"{key}_input"] = st.session_state[key]
        return (r, c)

    # If we get here, the value was invalid
    except Exception:
        # Silent fallback to last known-good coordinate.
        try:
            return _parse_coord(st.session_state[key])
        except Exception:
            return default

# ---------------------------------------------------------------------------
# Progressive learning: persist agent across Streamlit reruns
#
# Streamlit reruns the script top-to-bottom on every UI interaction.
# st.session_state is our "memory" so training can be progressive across runs.
# ---------------------------------------------------------------------------

if "agent" not in st.session_state:
    st.session_state.agent = None

# Optional: track the “context” so we can reset when the grid or hyperparams change.
if "agent_key" not in st.session_state:
    st.session_state.agent_key = None

# Track the grid size associated with the current Q-table (used for transfer logic).
if "agent_grid" not in st.session_state:
    st.session_state.agent_grid = None


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


def _transfer_q_table(old_agent: QAgent, new_agent: QAgent, rows: int, cols: int) -> None:
    """
    Copy overlapping learned Q-values from old_agent into new_agent.

    Keeps any state (r,c) where 0<=r<rows and 0<=c<cols.
    New states remain uninitialized (lazy zeros on first visit).

    Design intent:
    - Allows progressive learning when scaling the environment size.
    - Avoids throwing away knowledge on small-to-large transitions.
    """
    for k, v in old_agent.q_table.items():
        if isinstance(k, tuple) and len(k) == 2:
            r, c = int(k[0]), int(k[1])
            if 0 <= r < rows and 0 <= c < cols:
                new_agent.q_table[(r, c)] = np.asarray(v, dtype=float).copy()


# ---------------------------------------------------------------------------
# Sidebar UI
# ---------------------------------------------------------------------------


def sidebar_config() -> TrainingUIConfig:
    cfg = TrainingUIConfig()

    with st.sidebar:
        st.header("Training Settings")

        # Use text input for sidebar values to allow for easy editing.
        # This is a bit of a hack, but it's the only way to get the
        # text input to update the value in the config object.
        cfg.episodes = _text_num("Episodes", cfg.episodes, min_v=1, max_v=5000, cast=int, key="episodes")
        cfg.max_steps = _text_num("Max steps", cfg.max_steps, min_v=1, max_v=5000, cast=int, key="max_steps")

        cfg.epsilon = _text_num(
            "Epsilon",
            cfg.epsilon,
            min_v=0.0,
            max_v=1.0,
            cast=float,
            key="epsilon",
            step_hint="0.0–1.0",
        )
        cfg.alpha = _text_num(
            "Alpha",
            cfg.alpha,
            min_v=0.0,
            max_v=1.0,
            cast=float,
            key="alpha",
            step_hint="0.0–1.0",
        )
        cfg.gamma = _text_num(
            "Gamma",
            cfg.gamma,
            min_v=0.0,
            max_v=1.0,
            cast=float,
            key="gamma",
            step_hint="0.0–1.0",
        )

        cfg.rows = _text_num("Grid rows", cfg.rows, min_v=2, max_v=200, cast=int, key="rows")
        cfg.cols = _text_num("Grid cols", cfg.cols, min_v=2, max_v=200, cast=int, key="cols")

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
        cfg.start = _text_coord(
            "Start (row, col)",
            default=(0, 0),
            rows=rows_i,
            cols=cols_i,
            key="start_coord",
        )

        # Default goal is bottom-right unless user overrides.
        cfg.goal = _text_coord(
            "Goal (row, col)",
            default=(rows_i - 1, cols_i - 1),
            rows=rows_i,
            cols=cols_i,
            key="goal_coord",
        )

        # Reset button for the agent.
        # NOTE: This resets progressive learning and forces a fresh Q-table.
        if st.button("Reset Agent (clear Q-table)"):
            st.session_state.agent = None
            st.session_state.agent_key = None
            st.session_state.agent_grid = None
            st.success("Agent reset.")

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
            st.session_state.agent = QAgent.from_bytes(uploaded.read(), seed=cfg.seed)
            st.session_state.agent_grid = (int(cfg.rows), int(cfg.cols))
            st.session_state.agent_key = _agent_key(cfg)
            st.success("Loaded Q-table into agent.")

    return cfg


cfg = sidebar_config()


# ---------------------------------------------------------------------------
# Main_Panel UI
# ---------------------------------------------------------------------------

run = st.button("Run Training", type="primary")

# when "Run Training" is pressed it:
# 1. Create environment and agent
# 2. Run training loop
# 3. Display results in a few different ways (summary, q-table, value grid, policy grid)
# At the bottom is a UX improvement that limits visual clutter based on grid size.

if run:
    env = GridEnv(cfg.to_grid_config())

    key = _agent_key(cfg)

    # Create first agent if missing OR if hyperparameter/seed context changed.
    # This is the core of progressive learning: keep the same agent unless context changes.
    if st.session_state.agent is None or st.session_state.agent_key != key:
        st.session_state.agent = QAgent(cfg.to_q_config(), seed=cfg.seed)
        st.session_state.agent_key = key
        st.session_state.agent_grid = (int(cfg.rows), int(cfg.cols))

    # Transfer Q-table when grid size changes
    old_grid = st.session_state.get("agent_grid")
    new_rows, new_cols = int(cfg.rows), int(cfg.cols)

    # If old_grid is missing/corrupt, treat current grid as baseline (no transfer needed).
    if not (isinstance(old_grid, tuple) and len(old_grid) == 2):
        old_grid = (new_rows, new_cols)

    old_rows, old_cols = old_grid

    # Transfer whenever grid changes (grow or shrink).
    # NOTE:
    # - For grow-only behavior, change condition to: (new_rows >= old_rows and new_cols >= old_cols)
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
    wins = sum(1 for r in results if r.reached_goal)  # reached_goal is defined in train_runner>run_training
    st.success(f"Reached goal: {wins}/{len(results)}")

    st.write("Last 10 episodes:")
    st.write([r.__dict__ for r in results[-10:]])

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