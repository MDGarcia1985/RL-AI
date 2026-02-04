"""
test_action_selection.py

Testing file for action selection module

Phase 1: Basic Movement Control
    By Michael Garcia
    CSC370 Spring 2026
    michael@mandedesign.studio
"""

import numpy as np

from rc_agents.edge_ai.rcg_edge.agents.base import Action
from rc_agents.edge_ai.rcg_edge.agents.q_agent import QAgent, QConfig
from rc_agents.utils.logger import log_execution


def test_act_exploit_picks_argmax_action_when_epsilon_zero():
    log_execution("TEST_RUN", "test_act_exploit_picks_argmax_action_when_epsilon_zero")
    cfg = QConfig(alpha=0.1, gamma=0.9, epsilon=0.0)  # always exploit
    agent = QAgent(config=cfg, seed=42)

    s = (0, 0)

    # Force Q(s,*) so argmax is known.
    # Order is list(Action) -> [FORWARD, BACKWARD, RIGHT, LEFT]
    agent.q_table[s] = np.array([0.0, 0.0, 5.0, 1.0], dtype=float)

    result = agent.act(obs=s)

    assert result.action == Action.RIGHT
    assert result.info is not None
    assert result.info.get("policy") == "exploit"


def test_act_explore_returns_valid_action_when_epsilon_one():
    log_execution("TEST_RUN", "test_act_explore_returns_valid_action_when_epsilon_one")
    cfg = QConfig(alpha=0.1, gamma=0.9, epsilon=1.0)  # always explore
    agent = QAgent(config=cfg, seed=123)

    s = (0, 0)

    result = agent.act(obs=s)

    assert isinstance(result.action, Action)
    assert result.action in Action
    assert result.info is not None
    assert result.info.get("policy") == "explore"
