"""
test_maze_runner.py

MazeEnv + Training Runner Integration Tests
Verifies MazeEnv works correctly with the shared run_training loop.

Design intent:
- MazeEnv must be swappable without changes to train_runner.
- This test confirms:
  - runner can execute episodes against MazeEnv
  - reached_goal uses info["reached_goal"] consistently
  - termination on goal works in a maze corridor (not just open grids)

NOTE:
- These are *integration tests*, not performance tests.
- We keep episodes small and environments tiny to keep CI fast.

Copyright (c) 2026 Michael Garcia, M&E Design
https://mandedesign.studio
michael@mandedesign.studio

CSC370 Spring 2026
"""
from __future__ import annotations

from typing import Any

import numpy as np

from rc_agents.config.ui_config import TrainingUIConfig
from rc_agents.edge_ai.rcg_edge.agents.base import Action, StepResult
from rc_agents.edge_ai.rcg_edge.runners.train_runner import run_training
from rc_agents.envs.maze_env import MazeConfig, MazeEnv


class RightThenDownAgent:
    """
    Tiny deterministic policy agent for testing runner + MazeEnv wiring.

    Behavior:
    - Always tries RIGHT until it can't (or chooses a simple pattern)
    - Then tries BACKWARD (down)

    WHY:
    - We don't want stochastic exploration in an integration test.
    - We want repeatable behavior so failures mean "wiring broke."
    """

    def __init__(self) -> None:
        self.reset_calls = 0
        self.act_calls = 0
        self.learn_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def act(self, obs: Any) -> StepResult:
        self.act_calls += 1
        r, c = obs

        # Simple corridor strategy for our specific test maze:
        # If we're on top row, go RIGHT. Otherwise go DOWN.
        if r == 0 and c < 3:
            return StepResult(action=Action.RIGHT, info={"policy": "test"})
        return StepResult(action=Action.BACKWARD, info={"policy": "test"})

    def learn(self, obs: Any, action: Action, reward: float, next_obs: Any, done: bool) -> None:
        # Runner must call this; we don't need learning behavior here.
        self.learn_calls += 1


def _cfg(*, episodes: int = 5, max_steps: int = 50, seed: int | None = None) -> TrainingUIConfig:
    """
    Build a minimal TrainingUIConfig for runner tests.
    """
    cfg = TrainingUIConfig()
    cfg.episodes = int(episodes)
    cfg.max_steps = int(max_steps)
    cfg.seed = seed
    return cfg


def test_maze_env_runs_under_training_runner_and_can_reach_goal() -> None:
    """
    Integration check:
    - run_training should complete without error
    - at least one episode should reach the goal under a simple deterministic policy

    Maze layout:
        S . . .
        # # # .
        . . . G
        . . . .

    Corridor is top row, then down at last column if needed.
    """
    rows, cols = 4, 4
    walls = np.zeros((rows, cols), dtype=bool)

    # Block row 1 except last column (a "drop" at the far right).
    walls[1, 0] = True
    walls[1, 1] = True
    walls[1, 2] = True

    cfg_env = MazeConfig(
        rows=rows,
        cols=cols,
        start=(0, 0),
        goal=(2, 3),  # As marked in the maze layout
        step_cost=-1.0,
        goal_reward=0.0,
        wall_penalty=-1.0,
        allow_stay_on_wall_hit=True,
        terminate_on_goal=True,
        walls=walls,
    )
    env = MazeEnv(cfg_env)

    # Run training with a simple deterministic agent.
    agent = RightThenDownAgent()
    cfg = _cfg(episodes=3, max_steps=20, seed=123)

    results, _ = run_training(env=env, agent=agent, cfg=cfg)

    assert len(results) == 3
    assert agent.reset_calls == 3
    assert agent.act_calls >= 1
    assert agent.learn_calls == agent.act_calls

    # Our agent should reliably reach the goal in this maze.
    wins = sum(1 for r in results if r.reached_goal)
    assert wins == 3


def test_maze_env_reached_goal_comes_from_info_flag() -> None:
    """
    Confirms the runner uses:
        reached_goal = bool(info.get("reached_goal", done))

    WHY:
    - MazeEnv (and future envs) may have multiple terminal conditions.
    - done=True does not always mean "goal reached."
    """
    env = MazeEnv(MazeConfig(rows=2, cols=2, start=(0, 0), goal=(0, 1)))

    class OneStepRightAgent:
        def reset(self) -> None:
            return None

        def act(self, obs: Any) -> StepResult:
            return StepResult(action=Action.RIGHT, info={"policy": "test"})

        def learn(self, obs: Any, action: Action, reward: float, next_obs: Any, done: bool) -> None:
            return None

    agent = OneStepRightAgent()
    cfg = _cfg(episodes=1, max_steps=5)

    results, _ = run_training(env=env, agent=agent, cfg=cfg)
    assert results[0].reached_goal is True


def test_runner_does_not_assume_done_means_goal() -> None:
    """
    Confirm runner does not assume:
        reached_goal = done

    This is a regression test for a bug where the runner
    incorrectly used `done` as a synonym for `reached_goal`.
    """
    class DoneButNotGoalEnv:
        name = "done_but_not_goal_env"

        def reset(self):
            return (0, 0)

        def step(self, action: int):
            # done=True but explicitly not goal
            return (0, 0), -1.0, True, {"reached_goal": False}

    class NoopAgent:
        def reset(self) -> None:
            return None

        def act(self, obs: Any) -> StepResult:
            return StepResult(action=Action.RIGHT, info={"policy": "test"})

        def learn(self, obs: Any, action: Action, reward: float, next_obs: Any, done: bool) -> None:
            return None

    env = DoneButNotGoalEnv()
    agent = NoopAgent()
    cfg = _cfg(episodes=1, max_steps=5)

    results, _ = run_training(env=env, agent=agent, cfg=cfg)
    assert results[0].reached_goal is False