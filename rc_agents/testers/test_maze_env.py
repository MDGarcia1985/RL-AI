"""
test_maze_env.py

MazeEnv Tests
Verifies MazeEnv behavior matches the training runner contract and design intent.

Design intent:
- Tests are behavioral, not cosmetic.
- We verify environment guarantees:
  - reset() returns start (row, col)
  - step() respects walls and bounds (blocked moves don't change position)
  - wall_penalty stacks on blocked moves
  - reached_goal is reported through info["reached_goal"]
  - done is True when goal is reached (terminate_on_goal=True)
  - config validation rejects invalid start/goal (bounds + inside-wall)

NOTE:
- These tests do not train an agent.
- They validate MazeEnv mechanics so agents can assume stable physics.

Copyright (c) 2026 Michael Garcia, M&E Design
https://mandedesign.studio
michael@mandedesign.studio

CSC370 Spring 2026
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pytest

from rc_agents.edge_ai.rcg_edge.agents.base import Action
from rc_agents.envs.maze_env import MazeConfig, MazeEnv, ascii_maze_to_walls


def _tiny_open_cfg(*, rows: int = 3, cols: int = 3) -> MazeConfig:
    """
    Helper: produce a simple open maze config.

    WHY:
    - Keeps tests short.
    - Centralizes defaults so changes to MazeConfig fields update in one place.
    """
    return MazeConfig(
        rows=rows,
        cols=cols,
        start=(0, 0),
        goal=(rows - 1, cols - 1),
        step_cost=-1.0,
        goal_reward=0.0,
        wall_penalty=-1.0,
        allow_stay_on_wall_hit=True,
        terminate_on_goal=True,
        walls=None,  # no obstacles
    )


def test_reset_returns_start_position() -> None:
    cfg = _tiny_open_cfg(rows=4, cols=4)
    env = MazeEnv(cfg)

    obs = env.reset()
    assert obs == cfg.start


def test_step_moves_in_open_space() -> None:
    """
    In an open maze, actions should translate the agent deterministically.
    """
    cfg = _tiny_open_cfg(rows=4, cols=4)
    env = MazeEnv(cfg)
    env.reset()

    # Start at (0,0); RIGHT => (0,1)
    obs, reward, done, info = env.step(int(Action.RIGHT))
    assert obs == (0, 1)
    assert done is False
    assert info["hit_wall_or_bounds"] is False
    assert info["reached_goal"] is False
    assert reward == pytest.approx(cfg.step_cost)


def test_bounds_block_movement_and_stack_wall_penalty() -> None:
    """
    Bounds are treated like walls:
    - Position does not change
    - hit_wall_or_bounds=True
    - reward includes step_cost + wall_penalty
    """
    cfg = _tiny_open_cfg(rows=3, cols=3)
    env = MazeEnv(cfg)
    env.reset()

    # At (0,0); FORWARD would go out of bounds -> blocked.
    obs, reward, done, info = env.step(int(Action.FORWARD))

    assert obs == (0, 0)  # stayed in place
    assert done is False
    assert info["hit_wall_or_bounds"] is True
    assert reward == pytest.approx(cfg.step_cost + cfg.wall_penalty)


def test_wall_blocks_movement_and_stack_wall_penalty() -> None:
    """
    Walls are blocking cells:
    - Attempting to step into a wall leaves you in place (default behavior)
    - reward includes wall_penalty
    """
    # 3x3 maze with a wall at (0,1)
    walls = np.zeros((3, 3), dtype=bool)
    walls[0, 1] = True

    cfg = MazeConfig(
        rows=3,
        cols=3,
        start=(0, 0),
        goal=(2, 2),
        step_cost=-1.0,
        goal_reward=0.0,
        wall_penalty=-2.0,
        allow_stay_on_wall_hit=True,
        terminate_on_goal=True,
        walls=walls,
    )
    env = MazeEnv(cfg)
    env.reset()

    # RIGHT tries to move into wall (0,1) -> blocked.
    obs, reward, done, info = env.step(int(Action.RIGHT))

    assert obs == (0, 0)
    assert done is False
    assert info["hit_wall_or_bounds"] is True
    assert reward == pytest.approx(cfg.step_cost + cfg.wall_penalty)


def test_reaching_goal_sets_done_and_info() -> None:
    """
    When the agent reaches the goal:
    - reached_goal=True in info
    - done=True if terminate_on_goal=True
    - reward includes step_cost (+ optional goal_reward)
    """
    cfg = MazeConfig(
        rows=2,
        cols=2,
        start=(0, 0),
        goal=(0, 1),  # one step to the right
        step_cost=-1.0,
        goal_reward=5.0,
        wall_penalty=-1.0,
        allow_stay_on_wall_hit=True,
        terminate_on_goal=True,
        walls=None,
    )
    env = MazeEnv(cfg)
    env.reset()

    obs, reward, done, info = env.step(int(Action.RIGHT))
    assert obs == (0, 1)
    assert info["reached_goal"] is True
    assert done is True
    assert reward == pytest.approx(cfg.step_cost + cfg.goal_reward)


def test_config_rejects_start_or_goal_out_of_bounds() -> None:
    with pytest.raises(ValueError):
        _ = MazeEnv(MazeConfig(rows=3, cols=3, start=(-1, 0), goal=(2, 2)))

    with pytest.raises(ValueError):
        _ = MazeEnv(MazeConfig(rows=3, cols=3, start=(0, 0), goal=(99, 2)))


def test_config_rejects_start_or_goal_inside_wall() -> None:
    walls = np.zeros((3, 3), dtype=bool)
    walls[0, 0] = True
    walls[2, 2] = True

    with pytest.raises(ValueError):
        _ = MazeEnv(MazeConfig(rows=3, cols=3, start=(0, 0), goal=(1, 1), walls=walls))

    with pytest.raises(ValueError):
        _ = MazeEnv(MazeConfig(rows=3, cols=3, start=(1, 1), goal=(2, 2), walls=walls))


def test_ascii_maze_to_walls_rectangular_and_wall_chars() -> None:
    """
    ascii_maze_to_walls is a convenience loader for quick maze authoring.

    We verify:
    - It enforces rectangle shape
    - It maps wall characters to True
    """
    lines = [
        "###",
        "#.#",
        "###",
    ]
    w = ascii_maze_to_walls(lines, wall_chars="#")
    assert w.shape == (3, 3)
    assert bool(w[0, 0]) is True
    assert bool(w[1, 1]) is False

    with pytest.raises(ValueError):
        _ = ascii_maze_to_walls(["###", "##"], wall_chars="#")