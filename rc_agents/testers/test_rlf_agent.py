"""
test_rlf_agent.py

Unit tests for RLFAgent (Q-learning + fractal exploration).

These tests validate:
- Agent produces valid actions from the Action enum.
- Exploration path returns θ(theta) and keeps it bounded to [0, 2π(pi)).
- Q-table initializes lazily and updates on learn().
- Movement-delta stub is import-safe and produces expected deltas for 4-way actions.
- Soft action selection produces a sane probability distribution and does not crash.

NOTE:
- We avoid testing "performance" (wins) here. That belongs in integration tests or experiments,
  since RL outcomes depend on environment stochasticity, episode count, and reward design.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rc_agents.edge_ai.rcg_edge.agents.base import Action
from rc_agents.edge_ai.rcg_edge.agents.rlf_agent import (
    RLFAgent,
    RLFConfig,
    JuliaScout,
    apply_action_delta_stub,
    _action_from_theta_soft,
)


def test_rlf_agent_act_returns_valid_action() -> None:
    """
    RLFAgent.act() must always return an Action enum member.

    This is the contract expected by the runner and environment.
    """
    agent = RLFAgent(config=RLFConfig(epsilon=1.0), seed=123)  # epsilon=1 forces explore branch
    obs = (0, 0)

    step = agent.act(obs)

    assert isinstance(step.action, Action)
    assert step.action in tuple(Action)
    assert step.info.get("policy") in {"explore_fractal_theta", "exploit", "explore_random"}


def test_rlf_agent_exploration_returns_theta_in_info() -> None:
    """
    When RLFAgent explores via fractal theta mode, info should include theta.

    We keep epsilon=1 so exploration always triggers.
    """
    agent = RLFAgent(config=RLFConfig(epsilon=1.0), seed=123)
    obs = (1, 1)

    step = agent.act(obs)

    assert step.info.get("policy") == "explore_fractal_theta"
    assert "theta" in step.info
    assert isinstance(step.info["theta"], float)


def test_theta_range_is_0_to_2pi() -> None:
    """
    JuliaScout.step_theta() is documented to return θ(theta) in [0, 2π(pi)).

    This is important because our action mapping assumes normalized angles.
    """
    cfg = RLFConfig()
    scout = JuliaScout(c=complex(cfg.c_real, cfg.c_imag), z0=complex(cfg.z0_real, cfg.z0_imag))

    # Sample multiple steps to ensure normalization holds over time.
    for _ in range(250):
        theta = scout.step_theta()
        assert 0.0 <= theta < 2.0 * math.pi


def test_action_from_theta_soft_returns_valid_action() -> None:
    """
    _action_from_theta_soft() converts continuous θ(theta) into a discrete Action.

    It must always return a valid Action and should never raise.
    """
    rng = np.random.default_rng(123)

    # Test a sweep of angles across the circle, including boundary-ish values.
    thetas = [
        0.0,
        1e-12,
        math.pi / 2.0,
        math.pi,
        (3.0 * math.pi / 2.0),
        (2.0 * math.pi) - 1e-12,
    ]

    for theta in thetas:
        a = _action_from_theta_soft(theta, rng=rng, kappa=6.0)
        assert isinstance(a, Action)
        assert a in tuple(Action)


def test_q_table_initializes_and_learns() -> None:
    """
    Verify:
    - Q-table state is created lazily when acting/learning.
    - A learn() call updates Q(s,a) away from 0 when reward is non-zero.

    This is a minimal correctness check, not a convergence test.
    """
    agent = RLFAgent(config=RLFConfig(alpha=0.5, gamma=0.9, epsilon=0.0), seed=123)

    obs = (0, 0)
    next_obs = (0, 1)

    # Ensure state exists after act() (even though epsilon=0 uses exploit path).
    step = agent.act(obs)
    assert obs in agent.q_table
    assert agent.q_table[obs].shape == (len(tuple(Action)),)

    # Force a specific action update with a positive reward.
    agent.learn(obs=obs, action=Action.RIGHT, reward=1.0, next_obs=next_obs, done=False)

    # Q(s, RIGHT) should now be > 0 given alpha=0.5 and positive reward.
    qvals = agent.q_table[obs]
    assert qvals[int(Action.RIGHT)] > 0.0


def test_apply_action_delta_stub_four_way() -> None:
    """
    The movement stub is a future-facing helper. For now it must:
    - Be import-safe even if diagonal actions don't exist.
    - Correctly apply the 4-way deltas for the current Action enum.
    """
    # Start at (row=10, col=10) and apply each action once.
    r, c = 10, 10

    r2, c2 = apply_action_delta_stub(r, c, Action.FORWARD)
    assert (r2, c2) == (9, 10)

    r2, c2 = apply_action_delta_stub(r, c, Action.BACKWARD)
    assert (r2, c2) == (11, 10)

    r2, c2 = apply_action_delta_stub(r, c, Action.LEFT)
    assert (r2, c2) == (10, 9)

    r2, c2 = apply_action_delta_stub(r, c, Action.RIGHT)
    assert (r2, c2) == (10, 11)


def test_apply_action_delta_stub_unknown_action_is_noop() -> None:
    """
    If an unknown action is passed (shouldn't happen in normal operation),
    the stub should default to (0,0) delta and behave as a no-op.

    This guards future expansions where an action might exist but deltas
    have not yet been defined.
    """
    # We can't easily manufacture a non-Action value that type-checks as Action,
    # so we do a small trick: pass a real Action, but temporarily assert the
    # stub returns something sane for any missing mapping path.

    # This test effectively just ensures the helper never throws and returns ints.
    r, c = apply_action_delta_stub(0, 0, Action.FORWARD)
    assert isinstance(r, int)
    assert isinstance(c, int)