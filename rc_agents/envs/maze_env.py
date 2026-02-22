"""
maze_env.py

Maze Environment (Grid World with Obstacles)
A swappable environment for training agents in structured mazes.

Design intent:
- Preserve the same env contract used by train_runner:
    reset() -> obs
    step(action_int) -> (next_obs, reward, done, info)
- Keep movement deterministic, like GridEnv.
- Add walls/obstacles as an environment complexity upgrade.
- Support future features:
  - multiple start corners
  - multiple goals / checkpoints
  - terminal failure states (timeouts, traps, collisions)

Observation model:
- Phase 1: just (row, col) tuples (same as your GridEnv state key model)
- Phase 2 (optional): local sensor view (raycasts, neighborhood mask, etc.)

Reward model (default):
- step_cost < 0 each step to penalize wandering
- goal_reward >= 0 when goal is reached
- wall_penalty (optional) if agent attempts to walk into a wall (no movement)

Copyright (c) 2026 Michael Garcia, M&E Design
https://mandedesign.studio
michael@mandedesign.studio

CSC370 Spring 2026
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

from rc_agents.edge_ai.rcg_edge.agents.base import Action


@dataclass(frozen=True)
class MazeConfig:
    """
    Maze configuration container.

    NOTE:
    - walls is a 2D boolean grid:
        True  => wall/blocked
        False => open space
    """
    rows: int
    cols: int
    start: tuple[int, int] = (0, 0)
    goal: tuple[int, int] = (0, 0)

    # Reward shaping
    # In order to make banging into walls costly, total penalty is step + wall = penalty of -2
    step_cost: float = -1.0
    goal_reward: float = 0.0
    wall_penalty: float = -1.0

    # Behavior flags
    allow_stay_on_wall_hit: bool = True  # When hitting wall/bounds, stay in place.
    terminate_on_goal: bool = True

    # Wall grid
    walls: Optional[np.ndarray] = None  # shape (rows, cols), dtype bool


def _validate_cfg(cfg: MazeConfig) -> None:
    """
    Validate config values that do not depend on runtime state.

    NOTE:
    - Wall checks (start/goal inside wall) are handled after walls are constructed.
      That can’t be done here because cfg.walls may be None or not yet normalized.
    """
    r, c = int(cfg.rows), int(cfg.cols)
    if r <= 1 or c <= 1:
        raise ValueError("MazeConfig rows/cols must be > 1")

    sr, sc = cfg.start
    gr, gc = cfg.goal

    if not (0 <= sr < r and 0 <= sc < c):
        raise ValueError(f"Start out of bounds: {cfg.start}")

    if not (0 <= gr < r and 0 <= gc < c):
        raise ValueError(f"Goal out of bounds: {cfg.goal}")


def ascii_maze_to_walls(lines: Sequence[str], *, wall_chars: str = "#") -> np.ndarray:
    """
    Convert ASCII maze lines into a boolean wall grid.

    Example:
        lines = [
            "########",
            "#......#",
            "#.####.#",
            "#......#",
            "########",
        ]

    wall_chars:
        Any character in this set is treated as "wall".
        Everything else is treated as "open".

    Returns:
        np.ndarray[bool] with shape (rows, cols)
    """
    # Normalize lines (file reads often include trailing newlines).
    lines = [ln.rstrip("\n") for ln in lines]

    if len(lines) == 0:
        raise ValueError("ASCII maze cannot be empty")

    width = len(lines[0])
    if any(len(row) != width for row in lines):
        raise ValueError("ASCII maze must be rectangular (equal line lengths)")

    rows = len(lines)
    cols = width

    walls = np.zeros((rows, cols), dtype=bool)
    wall_set = set(wall_chars)

    for r in range(rows):
        for c in range(cols):
            walls[r, c] = (lines[r][c] in wall_set)

    return walls


class MazeEnv:
    """
    MazeEnv is a minimal, deterministic grid world with walls.

    Contract:
    - reset() returns an observation (row, col)
    - step(action_int) returns (next_obs, reward, done, info)

    WHY MazeEnv exists:
    - GridEnv is good for first learning.
    - MazeEnv gives structure: corridors, dead ends, and planning pressure.
    - This is a stepping stone toward real-world constraints (obstacles, no-go zones).
    """

    name = "maze_env"

    # Movement deltas in (dr, dc) row/col grid coordinates.
    # - FORWARD decreases row (moves "up")
    # - BACKWARD increases row (moves "down")
    # - LEFT decreases col
    # - RIGHT increases col
    _ACTION_DELTAS: Dict[Action, tuple[int, int]] = {
        Action.FORWARD:  (-1,  0),
        Action.BACKWARD: ( 1,  0),
        Action.LEFT:     ( 0, -1),
        Action.RIGHT:    ( 0,  1),
    }

    def __init__(self, cfg: MazeConfig):
        _validate_cfg(cfg)
        self.cfg = cfg

        # If walls grid isn't supplied, default to "open world" (no obstacles).
        if cfg.walls is None:
            self.walls = np.zeros((cfg.rows, cfg.cols), dtype=bool)
        else:
            w = np.asarray(cfg.walls, dtype=bool)
            if w.shape != (cfg.rows, cfg.cols):
                raise ValueError(f"walls shape must be ({cfg.rows},{cfg.cols}), got {w.shape}")
            self.walls = w

        # Validate start/goal are not walls.
        # This must happen AFTER walls are constructed and normalized.
        sr, sc = int(self.cfg.start[0]), int(self.cfg.start[1])
        gr, gc = int(self.cfg.goal[0]), int(self.cfg.goal[1])

        if self.walls[sr, sc]:
            raise ValueError(f"Start position is inside a wall: {self.cfg.start}")
        if self.walls[gr, gc]:
            raise ValueError(f"Goal position is inside a wall: {self.cfg.goal}")

        # State
        self._pos = (sr, sc)

        # Optional deterministic seeding.
        # _seed is stored for introspection/debugging only; the rng is already seeded from it.
        self._seed: Optional[int] = None
        self._rng = np.random.default_rng(None)

    def seed(self, seed: int) -> None:
        """
        Seed hook for the runner (keeps reproducibility consistent with GridEnv).

        NOTE:
        - MazeEnv doesn't need randomness for movement right now,
          but it may for future maze generation.
        """
        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)

    def reset(self) -> Any:
        """
        Start a new episode.

        NOTE:
        - If start is a wall, this is a config error (we already validated in __init__).
        """
        sr, sc = int(self.cfg.start[0]), int(self.cfg.start[1])
        self._pos = (sr, sc)
        return self._pos

    def _is_open(self, r: int, c: int) -> bool:
        """
        Returns True if (r, c) is within bounds and not a wall.
        Combines the bounds check and wall check into a single guard.
        """
        return (
            0 <= r < self.cfg.rows
            and 0 <= c < self.cfg.cols
            and not bool(self.walls[r, c])
        )

    def step(self, action: int) -> tuple[Any, float, bool, Dict[str, object]]:
        """
        Apply action (int) and return step tuple.

        NOTE:
        - action is an IntEnum value (Action), so int(action) is stable.
        - We convert action exactly once, with a clear error if invalid.
        """
        try:
            a = Action(int(action))
        except ValueError as e:
            raise ValueError(f"Invalid action int: {action}") from e

        r, c = self._pos
        pos_before = (r, c)

        dr, dc = self._ACTION_DELTAS.get(a, (0, 0))
        nr, nc = r + dr, c + dc

        hit_wall_or_bounds = not self._is_open(nr, nc)

        if hit_wall_or_bounds:
            if self.cfg.allow_stay_on_wall_hit:
                # Blocked move -> agent stays in place.
                nr, nc = r, c
            else:
                # TODO: implement terminal-on-collision behavior here.
                raise NotImplementedError("allow_stay_on_wall_hit=False is not yet implemented")

        self._pos = (nr, nc)
        pos_after = self._pos

        reached_goal = (self._pos == self.cfg.goal)
        done = bool(reached_goal and self.cfg.terminate_on_goal)

        # Reward shaping:
        # - step_cost always applies (discourages wandering)
        # - wall_penalty stacks when you attempt a blocked move
        # - goal_reward applied once on reaching goal
        reward = float(self.cfg.step_cost)
        if hit_wall_or_bounds:
            reward += float(self.cfg.wall_penalty)
        if reached_goal:
            reward += float(self.cfg.goal_reward)

        info: Dict[str, object] = {
            "reached_goal": reached_goal,
            "hit_wall_or_bounds": hit_wall_or_bounds,
            "pos_before": pos_before,
            "pos_after": pos_after,
        }

        return self._pos, reward, done, info