"""
maze_runner.py

Maze Runner Helper
Builds MazeEnv (walls/obstacles) and runs the shared training loop.

Design intent:
- Do NOT fork training logic.
- Keep one training loop (train_runner.run_training).
- Provide a convenient entry point for MazeEnv experiments.

Copyright (c) 2026 Michael Garcia, M&E Design
https://mandedesign.studio
michael@mandedesign.studio

CSC370 Spring 2026
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from rc_agents.config.ui_config import TrainingUIConfig
from rc_agents.envs import MazeConfig, MazeEnv, ascii_maze_to_walls
from .train_runner import run_training

def _make_odd(n: int) -> int:
    n = max(3, int(n))
    return n if (n % 2 == 1) else (n + 1)

def run_maze_training_from_ascii(
    *,
    ascii_lines: Sequence[str],
    agent,
    cfg: TrainingUIConfig,
    start: tuple[int, int] | None = None,
    goal: tuple[int, int] | None = None,
    wall_chars: str = "#",
    step_cost: float = -1.0,
    goal_reward: float = 0.0,
    wall_penalty: float = -1.0,
):
    """
    Convenience helper:
    - Convert ASCII -> walls grid
    - Create MazeEnv
    - Call shared run_training()

    This keeps env swapping simple:
        GridEnv  -> run_training(...)
        MazeEnv  -> run_maze_training_from_ascii(...)
    """
    # ------------------------------------------------------------------
    # Resolve start/goal:
    # - If explicitly provided, use them.
    # - Otherwise fall back to UI config values.
    # This keeps TrainingUIConfig as primary source of truth.
    # ------------------------------------------------------------------
    start = start if start is not None else cfg.start
    goal  = goal  if goal  is not None else cfg.goal

    walls = ascii_maze_to_walls(ascii_lines, wall_chars=wall_chars)
    rows, cols = int(walls.shape[0]), int(walls.shape[1])

    env = MazeEnv(
        MazeConfig(
            rows=rows,
            cols=cols,
            start=start,
            goal=goal,
            step_cost=step_cost,
            goal_reward=goal_reward,
            wall_penalty=wall_penalty,
            walls=walls,
        )
    )

    # Refactor: run_training now returns (results, best_trajectory). We pass through
    # so callers (e.g. tests) can unpack or use both; main_panel uses factory + run_training
    # directly for Streamlit, but maze_runner remains the canonical maze entry point.
    results, best_trajectory = run_training(env=env, agent=agent, cfg=cfg)
    return (results, best_trajectory)

def _dfs_maze_walls(rows: int, cols: int, *, rng: np.random.Generator) -> np.ndarray:
    """
    True = wall, False = open.
    DFS backtracker carved maze. Borders remain walls.
    """
    rows = _make_odd(rows)
    cols = _make_odd(cols)

    walls = np.ones((rows, cols), dtype=bool)
    start = (1, 1)
    walls[start] = False

    stack = [start]
    dirs = [(0, 2), (2, 0), (0, -2), (-2, 0)]  # (dr, dc)

    while stack:
        r, c = stack[-1]
        carved = False

        for i in rng.permutation(len(dirs)):
            dr, dc = dirs[i]
            nr, nc = r + dr, c + dc

            if 1 <= nr < rows - 1 and 1 <= nc < cols - 1 and walls[nr, nc]:
                walls[nr, nc] = False
                walls[r + dr // 2, c + dc // 2] = False
                stack.append((nr, nc))
                carved = True
                break

        if not carved:
            stack.pop()

    return walls


def generate_dfs_maze_walls(rows: int, cols: int, *, rng: np.random.Generator) -> np.ndarray:
    """
    Public API for tests and callers.

    Returns a boolean wall grid:
        True  => wall
        False => open
    """
    return _dfs_maze_walls(rows, cols, rng=rng)


def run_maze_training_generated_dfs(
    *,
    agent,
    cfg: TrainingUIConfig,
    start: tuple[int, int] | None = None,
    goal: tuple[int, int] | None = None,
    step_cost: float = -1.0,
    goal_reward: float = 0.0,
    wall_penalty: float = -1.0,
):
    start = start if start is not None else cfg.start
    goal  = goal  if goal  is not None else cfg.goal

    rng = np.random.default_rng(cfg.seed)

    walls = _dfs_maze_walls(cfg.rows, cfg.cols, rng)

    # Practical default: if user left goal as (4,4) but maze got forced odd size,
    # clamp to bottom-right open cell convention.
    rows, cols = walls.shape
    if goal == cfg.goal and goal == (cfg.rows - 1, cfg.cols - 1):
        goal = (rows - 2, cols - 2)

    env = MazeEnv(
        MazeConfig(
            rows=rows,
            cols=cols,
            start=start,
            goal=goal,
            step_cost=step_cost,
            goal_reward=goal_reward,
            wall_penalty=wall_penalty,
            walls=walls,
        )
    )

    # Optional, but consistent with your design
    if cfg.seed is not None:
        env.seed(cfg.seed)

    # Refactor: run_training returns (results, best_trajectory); return both to caller.
    results, best_trajectory = run_training(env=env, agent=agent, cfg=cfg)
    return (results, best_trajectory)