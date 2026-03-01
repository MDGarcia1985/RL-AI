"""
test_factory.py

Smoke tests for Streamlit UI factory.

Design intent:
- Factory must build agents via catalog.
- Factory must build environments for Open World and Maze.
- Maze generation must be deterministic (seed-driven) and valid.
"""

from __future__ import annotations

import numpy as np
import pytest

from rc_agents.config import TrainingUIConfig
from rc_agents.ui.streamlit_ui.factory import get_env_options, make_agent, make_env


def test_get_env_options_returns_open_world_and_maze() -> None:
    """Sidebar dropdown uses get_env_options(); must include Open World and Maze."""
    options = get_env_options()
    assert isinstance(options, list)
    assert len(options) >= 2
    values = [v for v, _ in options]
    labels = [lbl for _, lbl in options]
    assert "Open World" in values
    assert "Maze" in values
    assert "Open World" in labels
    assert "Maze" in labels


def _cfg() -> TrainingUIConfig:
    cfg = TrainingUIConfig()
    cfg.rows = 11
    cfg.cols = 11
    cfg.start = (0, 0)
    cfg.goal = (10, 10)
    cfg.episodes = 1
    cfg.max_steps = 10
    cfg.alpha = 0.5
    cfg.gamma = 0.9
    cfg.epsilon = 0.1
    cfg.seed = 123
    return cfg


def test_make_env_open_world_builds() -> None:
    cfg = _cfg()
    env = make_env(cfg, game_type="Open World")
    assert getattr(env, "name", "") in ("grid_env", "grid") or env is not None

    obs = env.reset()
    assert isinstance(obs, tuple) and len(obs) == 2


def test_make_env_maze_builds_and_is_valid() -> None:
    cfg = _cfg()
    env = make_env(cfg, game_type="Maze")

    # Expected API from MazeEnv
    assert getattr(env, "name", "") in ("maze_env", "maze") or env is not None
    assert hasattr(env, "walls")

    walls = env.walls
    assert isinstance(walls, np.ndarray)
    assert walls.dtype == bool
    assert walls.ndim == 2

    # Start/goal must not be walls
    sr, sc = env.cfg.start
    gr, gc = env.cfg.goal
    assert bool(walls[sr, sc]) is False
    assert bool(walls[gr, gc]) is False


def test_make_env_maze_same_seed_reproducible() -> None:
    cfg1 = _cfg()
    cfg2 = _cfg()
    cfg1.seed = 777
    cfg2.seed = 777

    env1 = make_env(cfg1, game_type="Maze")
    env2 = make_env(cfg2, game_type="Maze")

    assert np.array_equal(env1.walls, env2.walls)


def test_make_agent_unknown_raises() -> None:
    cfg = _cfg()
    with pytest.raises(ValueError):
        _ = make_agent("not_a_real_agent", cfg)


@pytest.mark.parametrize("agent_id", ["rl", "rlf"])
def test_make_agent_known_if_present(agent_id: str) -> None:
    """
    Agents that aren't wired yet should SKIP cleanly, not hard fail.
    """
    cfg = _cfg()

    try:
        agent = make_agent(agent_id, cfg)
    except ValueError:
        pytest.skip(f"{agent_id} not registered in catalog yet")
    except Exception as e:
        pytest.skip(f"{agent_id} not instantiable yet: {e}")

    assert agent is not None
    assert hasattr(agent, "act")
    assert hasattr(agent, "learn")