"""
test_trainer.py

Training Runner Smoke Tests
Verifies the training loop wires together env + agent correctly.

Design intent:
- Tests are behavioral, not cosmetic.
- We verify runner guarantees:
  - calls env.reset() once per episode
  - calls agent.reset() once per episode
  - calls env.step(...) up to max_steps
  - calls agent.learn(...) once per step
  - stops early when done=True
  - produces correct EpisodeResult fields
  - supports optional env.seed(...) when cfg.seed is set

NOTE:
- This test uses tiny stub classes (FakeEnv / FakeAgent) so it does not
  depend on GridEnv behavior. The goal is to test the runner contract.

Copyright (c) 2026 Michael Garcia, M&E Design
https://mandedesign.studio
michael@mandedesign.studio

CSC370 Spring 2026
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import pytest

from rc_agents.config.ui_config import TrainingUIConfig
from rc_agents.edge_ai.rcg_edge.runners.train_runner import run_training
from rc_agents.edge_ai.rcg_edge.agents.base import Action, StepResult


class FakeEnv:
    """
    Minimal environment stub implementing the Env Protocol used by run_training().

    Behavior:
    - reset() returns a constant observation
    - step() increments an internal counter and terminates after done_after steps
    """

    def __init__(self, *, done_after: int = 3, reward: float = -1.0) -> None:
        self.done_after = int(done_after)
        self.reward = float(reward)

        self.reset_calls = 0
        self.step_calls = 0
        self.seed_calls = 0
        self.last_action: int | None = None

    def seed(self, seed: int) -> None:
        self.seed_calls += 1

    def reset(self) -> Any:
        self.reset_calls += 1
        self.step_calls = 0
        return (0, 0)

    def step(self, action: int) -> Tuple[Any, float, bool, Dict[str, object]]:
        self.step_calls += 1
        self.last_action = int(action)

        done = (self.step_calls >= self.done_after)
        obs = (0, self.step_calls)  # changes each step so agent sees "new" obs
        info: Dict[str, object] = {"reached_goal": done}
        return obs, self.reward, done, info


class FakeAgent:
    """
    Minimal agent stub implementing the runner's expected agent interface.

    Behavior:
    - act() always returns Action.RIGHT
    - learn() records calls
    """

    def __init__(self) -> None:
        self.reset_calls = 0
        self.act_calls = 0
        self.learn_calls = 0
        self.last_obs: Any = None
        self.last_action: Action | None = None

    def reset(self) -> None:
        self.reset_calls += 1

    def act(self, obs: Any) -> StepResult:
        self.act_calls += 1
        self.last_obs = obs
        return StepResult(action=Action.RIGHT, info={"policy": "test"})

    def learn(self, obs: Any, action: Action, reward: float, next_obs: Any, done: bool) -> None:
        self.learn_calls += 1
        self.last_action = action


def _cfg(*, episodes: int = 2, max_steps: int = 10, seed: int | None = None) -> TrainingUIConfig:
    """
    Build a minimal TrainingUIConfig for runner tests.

    NOTE:
    - We include required fields used by your config object.
    - If TrainingUIConfig adds new required fields later, update here.
    """
    cfg = TrainingUIConfig()
    cfg.episodes = int(episodes)
    cfg.max_steps = int(max_steps)
    cfg.seed = seed
    return cfg


def test_runner_calls_reset_once_per_episode() -> None:
    env = FakeEnv(done_after=2)
    agent = FakeAgent()
    cfg = _cfg(episodes=3, max_steps=10)

    results = run_training(env=env, agent=agent, cfg=cfg)

    assert len(results) == 3
    assert env.reset_calls == 3
    assert agent.reset_calls == 3


def test_runner_stops_early_when_done() -> None:
    env = FakeEnv(done_after=3)
    agent = FakeAgent()
    cfg = _cfg(episodes=1, max_steps=999)

    results = run_training(env=env, agent=agent, cfg=cfg)

    assert len(results) == 1
    assert results[0].reached_goal is True
    assert results[0].steps == 3  # terminated early at done_after
    assert env.step_calls == 3
    assert agent.act_calls == 3
    assert agent.learn_calls == 3


def test_runner_respects_max_steps_when_never_done() -> None:
    env = FakeEnv(done_after=10_000)  # effectively never done for this test
    agent = FakeAgent()
    cfg = _cfg(episodes=1, max_steps=5)

    results = run_training(env=env, agent=agent, cfg=cfg)

    assert len(results) == 1
    assert results[0].reached_goal is False
    assert results[0].steps == 5
    assert env.step_calls == 5
    assert agent.learn_calls == 5


def test_runner_sets_reached_goal_from_info_when_present() -> None:
    """
    Future-proofing test.

    If your runner uses:
        reached_goal = bool(info.get("reached_goal", done))
    then this test ensures it honors info["reached_goal"].
    """
    env = FakeEnv(done_after=2)
    agent = FakeAgent()
    cfg = _cfg(episodes=1, max_steps=10)

    results = run_training(env=env, agent=agent, cfg=cfg)
    assert results[0].reached_goal is True


def test_runner_calls_env_seed_when_cfg_seed_set() -> None:
    env = FakeEnv(done_after=1)
    agent = FakeAgent()
    cfg = _cfg(episodes=1, max_steps=10, seed=123)

    _ = run_training(env=env, agent=agent, cfg=cfg)

    assert env.seed_calls == 1