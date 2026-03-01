"""
factory.py

Factory helpers for Streamlit UI.

Design intent:
- Centralize "how to build things" (agents + envs) so UI stays clean.
- Avoid import-time failures:
  - Heavy or WIP imports happen INSIDE factory functions.
- Make it easy to add new agent architectures without rewriting UI logic.

Current Scope:
- make_agent(): builds an agent from agent_catalog by agent_id
- make_env(): builds the selected environment from game_type

Future:
- env_catalog similar to agent_catalog
- per-agent parameter tabs (agent-specific configs)
- maze authoring (ASCII editor + presets)
- environment variants (hazards, checkpoints, multi-goal)

This module should remain:
- Import-safe (no heavy imports at module level).
- Deterministic (seed-driven where applicable).
- Easy to add new agents without rewriting UI logic.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np

from rc_agents.config import TrainingUIConfig
from .agent_catalog import build_agent_catalog


# ---------------------------------------------------------------------------
# Environment options (for sidebar dropdown)
#
# Refactor: sidebar_ui uses this to populate the environment selectbox.
# (value, label) so the UI shows a friendly label while we pass the same
# value to make_env(cfg, game_type=value). Adding a new env type: add
# a tuple here and handle it in make_env() below.
# ---------------------------------------------------------------------------

def get_env_options() -> List[Tuple[str, str]]:
    """
    Return list of (value, label) for environment dropdown.
    Value is passed to make_env(cfg, game_type=value).
    """
    return [
        ("Open World", "Open World"),
        ("Maze", "Maze"),
    ]


# ---------------------------------------------------------------------------
# Agent Factory
# ---------------------------------------------------------------------------

def make_agent(agent_id: str, cfg: TrainingUIConfig) -> Any:
    """
    Instantiate an agent using the catalog definition.

    The catalog defines:
    - Human-readable label
    - Constructor binding
    - Feature flags (e.g., Q-table IO support)

    Agents are created lazily to prevent import-time failures
    when experimenting with new architectures.
    """
    catalog = build_agent_catalog()
    spec = catalog.get(agent_id)

    if spec is None:
        raise ValueError(f"Unknown agent_id: {agent_id}")

    # Lazy construction: spec.make(cfg) does the actual import + instantiation
    return spec.make(cfg)


def agent_supports_qtable_io(agent_id: str) -> bool:
    """
    Return whether the specified agent supports
    Q-table save/load functionality.
    """
    catalog = build_agent_catalog()
    spec = catalog.get(agent_id)
    return bool(spec.supports_qtable_io) if spec else False


# ---------------------------------------------------------------------------
# Environment Factory
# ---------------------------------------------------------------------------

def make_env(cfg: TrainingUIConfig, *, game_type: str):
    """
    Build environment instance from UI selection.

    Today:
    - Open World -> GridEnv
    - Maze       -> MazeEnv (DFS-generated walls)

    Future:
    - Labyrinth -> different rules / hazards / checkpoints
    """
    if game_type == "Open World":
        from rc_agents.envs import GridEnv
        return GridEnv(cfg.to_grid_config())

    if game_type == "Maze":
        return _make_maze_env_from_dfs(cfg)

    # Keep the UI option visible but enforce "not yet".
    # Future: Labyrinth -> different rules / hazards / checkpoints
    raise NotImplementedError("Labyrinth (future) is not implemented yet.")


# ---------------------------------------------------------------------------
# Maze Builder
# ---------------------------------------------------------------------------

def _make_maze_env_from_dfs(cfg: TrainingUIConfig):
    """
    Build a MazeEnv using deterministic DFS wall generation.

    Notes:
    - DFS mazes typically require odd dimensions.
    - Border cells are usually walls.
    - Start/goal are adjusted if user selects corner positions.
    - Seed controls deterministic maze layout.
    """
    from rc_agents.envs import MazeEnv, MazeConfig

    # Use the generator from maze_runner.py.
    # Preferred: public API generate_dfs_maze_walls(...); fallback for import quirks.
    try:
        from rc_agents.edge_ai.rcg_edge.runners.maze_runner import generate_dfs_maze_walls
    except Exception:
        from rc_agents.edge_ai.rcg_edge.runners.maze_runner import _dfs_maze_walls as generate_dfs_maze_walls

    rng = np.random.default_rng(cfg.seed)
    walls = generate_dfs_maze_walls(int(cfg.rows), int(cfg.cols), rng=rng)

    rows, cols = int(walls.shape[0]), int(walls.shape[1])

    # Practical defaults:
    # - In DFS mazes, (0,0) is almost always a wall border.
    # - Use (1,1) and (rows-2, cols-2) as "interior corners".
    start = tuple(cfg.start)
    goal = tuple(cfg.goal)

    if start == (0, 0):
        start = (1, 1)
    if goal == (int(cfg.rows) - 1, int(cfg.cols) - 1):
        goal = (rows - 2, cols - 2)

    # Hard clamp to grid bounds (MazeEnv will validate; this prevents mistakes)
    sr, sc = int(start[0]), int(start[1])
    gr, gc = int(goal[0]), int(goal[1])
    sr = max(0, min(rows - 1, sr))
    sc = max(0, min(cols - 1, sc))
    gr = max(0, min(rows - 1, gr))
    gc = max(0, min(cols - 1, gc))

    start = (sr, sc)
    goal = (gr, gc)

    # If the chosen start/goal land on a wall, push them to canonical interior cells.
    if bool(walls[start]):
        start = (1, 1)
    if bool(walls[goal]):
        goal = (rows - 2, cols - 2)

    return MazeEnv(
        MazeConfig(
            rows=rows,
            cols=cols,
            start=start,
            goal=goal,
            walls=walls,
            # Reward shaping aligned to MazeConfig defaults
            step_cost=-1.0,
            goal_reward=0.0,
            wall_penalty=-1.0,
            allow_stay_on_wall_hit=True,
            terminate_on_goal=True,
        )
    )