"""
test_progressive_learning.py

Tests for progressive_learning helpers (agent_key, transfer_q_table).

Design intent:
- agent_key(cfg) returns a stable tuple for (alpha, gamma, epsilon, seed).
- transfer_q_table copies overlapping Q-values from old to new agent when grid size changes.
- We do not run Streamlit here; ensure_state is exercised via the UI.
"""

from __future__ import annotations

import numpy as np

import pytest

from rc_agents.config import TrainingUIConfig
from rc_agents.edge_ai.rcg_edge.agents.rl_agent import RLAgent, RLConfig
from rc_agents.ui.streamlit_ui.progressive_learning import agent_key, transfer_q_table


def _cfg(*, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1, seed: int = 42) -> TrainingUIConfig:
    cfg = TrainingUIConfig()
    cfg.alpha = alpha
    cfg.gamma = gamma
    cfg.epsilon = epsilon
    cfg.seed = seed
    return cfg


def test_agent_key_is_stable_for_same_config() -> None:
    """Same hyperparams -> same key (reuse agent)."""
    c1 = _cfg(alpha=0.2, seed=123)
    c2 = _cfg(alpha=0.2, seed=123)
    assert agent_key(c1) == agent_key(c2)


def test_agent_key_changes_with_alpha() -> None:
    """Different alpha -> different key (fresh agent)."""
    c1 = _cfg(alpha=0.1)
    c2 = _cfg(alpha=0.2)
    assert agent_key(c1) != agent_key(c2)


def test_agent_key_changes_with_seed() -> None:
    """Different seed -> different key."""
    c1 = _cfg(seed=1)
    c2 = _cfg(seed=2)
    assert agent_key(c1) != agent_key(c2)


def test_agent_key_handles_none_seed() -> None:
    """cfg.seed None -> key still hashable (uses 0)."""
    cfg = _cfg()
    cfg.seed = None
    key = agent_key(cfg)
    assert isinstance(key, tuple)
    assert len(key) == 4


def test_transfer_q_table_copies_overlapping_states() -> None:
    """
    transfer_q_table(old, new, rows, cols) copies Q(s,a) for (r,c) in [0,rows) x [0,cols).
    Used when resizing grid so learning is not lost on overlap.
    """
    old_agent = RLAgent(RLConfig(alpha=0.1), seed=1)
    old_agent.q_table[(0, 0)] = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    old_agent.q_table[(1, 1)] = np.array([0.5, 0.5, 0.5, 0.5], dtype=float)
    old_agent.q_table[(9, 9)] = np.array([10.0, 10.0, 10.0, 10.0], dtype=float)

    new_agent = RLAgent(RLConfig(alpha=0.1), seed=2)
    transfer_q_table(old_agent, new_agent, rows=3, cols=3)

    assert (0, 0) in new_agent.q_table
    assert (1, 1) in new_agent.q_table
    assert np.allclose(new_agent.q_table[(0, 0)], [1.0, 2.0, 3.0, 4.0])
    assert np.allclose(new_agent.q_table[(1, 1)], [0.5, 0.5, 0.5, 0.5])
    # (9,9) is outside [0,3) x [0,3) so not transferred
    assert (9, 9) not in new_agent.q_table


def test_transfer_q_table_no_op_for_agents_without_q_table() -> None:
    """transfer_q_table is a no-op if either agent lacks q_table (e.g. RandomAgent)."""
    class NoQAgent:
        pass
    old_a = NoQAgent()
    new_a = RLAgent(RLConfig(), seed=1)
    transfer_q_table(old_a, new_a, rows=5, cols=5)
    assert len(new_a.q_table) == 0
