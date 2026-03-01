"""
test_runtime.py

Runtime / Integration Smoke Tests
Ensures the package can be executed as a "whole system" in a minimal way.

Design intent:
- These tests are not about optimal learning.
- They are about "does the system run end-to-end without crashing?"
- We keep episode counts small so CI/local runs stay fast.

What this test protects:
- Import paths (relative imports stay correct)
- Runner can execute with real Env + real Agent
- Q-table populates (learning path is exercised)
- Convergence summary attaches (instrumentation doesn't break runtime)

NOTE:
- This test intentionally avoids Streamlit UI execution.
  UI testing is a separate concern (and Streamlit isn't pytest-friendly by default).

Copyright (c) 2026 Michael Garcia, M&E Design
https://mandedesign.studio
michael@mandedesign.studio

CSC370 Spring 2026
"""
from __future__ import annotations

from rc_agents.config.ui_config import TrainingUIConfig
from rc_agents.envs import GridEnv
from rc_agents.edge_ai.rcg_edge.agents import QAgent, RLAgent, RLFAgent
from rc_agents.edge_ai.rcg_edge.runners.train_runner import run_training


def _tiny_cfg(*, episodes: int = 10, max_steps: int = 200, rows: int = 6, cols: int = 6, seed: int = 123) -> TrainingUIConfig:
    """
    Build a small config for runtime smoke tests.

    WHY:
    - We want something that runs fast.
    - We want enough steps that the agent can plausibly reach the goal sometimes.
    - We keep grids small so "random exploration" can still succeed occasionally.
    """
    cfg = TrainingUIConfig()
    cfg.episodes = int(episodes)
    cfg.max_steps = int(max_steps)
    cfg.rows = int(rows)
    cfg.cols = int(cols)

    # Start/goal are explicit to avoid "defaults drift" later.
    cfg.start = (0, 0)
    cfg.goal = (rows - 1, cols - 1)

    # Hyperparams: not "best", just sane.
    cfg.alpha = 0.5
    cfg.gamma = 0.9
    cfg.epsilon = 0.2

    cfg.seed = int(seed)
    return cfg


def test_runtime_q_agent_runs_end_to_end() -> None:
    """
    End-to-end run with the baseline QAgent.

    Pass condition:
    - No exceptions
    - Results list length equals episodes
    - Agent's q_table is non-empty (learning path executed)
    - Convergence summary exists (runner instrumentation attached)
    """
    cfg = _tiny_cfg(episodes=12, max_steps=150, rows=6, cols=6, seed=1)

    env = GridEnv(cfg.to_grid_config())
    agent = QAgent(cfg.to_q_config(), seed=cfg.seed)

    results, _ = run_training(env=env, agent=agent, cfg=cfg)

    assert len(results) == cfg.episodes
    assert isinstance(agent.q_table, dict)
    assert len(agent.q_table) > 0  # visited at least one state, updated table

    # Runner attaches convergence_summary as a convenience attribute.
    assert hasattr(agent, "convergence_summary")
    assert agent.convergence_summary is not None


def test_runtime_rl_agent_runs_end_to_end() -> None:
    """
    End-to-end run with RLAgent baseline.

    WHY:
    - RLAgent is the benchmark for exploration variants.
    - This test ensures new agents remain plug-compatible with the runner.
    """
    cfg = _tiny_cfg(episodes=10, max_steps=150, rows=6, cols=6, seed=2)

    env = GridEnv(cfg.to_grid_config())
    agent = RLAgent(seed=cfg.seed)  # RLAgent has its own config defaults

    results, _ = run_training(env=env, agent=agent, cfg=cfg)

    assert len(results) == cfg.episodes
    assert len(agent.q_table) > 0
    assert hasattr(agent, "convergence_summary")


def test_runtime_rlf_agent_runs_end_to_end() -> None:
    """
    End-to-end run with RLFAgent (fractal-driven exploration).

    WHY:
    - Confirms JuliaScout exploration code runs during training
    - Confirms runner doesn't care about how exploration is generated
    """
    cfg = _tiny_cfg(episodes=10, max_steps=150, rows=6, cols=6, seed=3)

    env = GridEnv(cfg.to_grid_config())
    agent = RLFAgent(seed=cfg.seed)

    results, _ = run_training(env=env, agent=agent, cfg=cfg)

    assert len(results) == cfg.episodes
    assert len(agent.q_table) > 0
    assert hasattr(agent, "convergence_summary")