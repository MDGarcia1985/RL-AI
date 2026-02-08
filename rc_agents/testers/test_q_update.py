"""
test_q_update.py

Unit tests for Q-learning updates in the RC-Gazebo simulation.

Phase 1: Basic Movement Control
    By Michael Garcia
    CSC370 Spring 2026
    michael@mandedesign.studio
"""

import numpy as np

from rc_agents.edge_ai.rcg_edge.agents.base import Action
from rc_agents.edge_ai.rcg_edge.agents.q_agent import QAgent, QConfig
from rc_agents.utils.logger import log_execution

def test_q_update_basic_nonterminal():
    """
    Verifies the core Q-learning update:
    Q(s,a) <- Q(s,a) + alpha * (target - Q(s,a))
    where target = r + gamma * max_a' Q(s', a')
    """
    log_execution("TEST_RUN", "test_q_update_basic_nonterminal")
    cfg = QConfig(alpha=0.5, gamma=0.9, epsilon=0.0)
    agent = QAgent(config=cfg, seed=123)

    s = (0, 0)
    s2 = (0, 1)
    a = Action.RIGHT
    r = 1.0
    done = False
    #Force known Q values:
    #Q(s,*) = 0 initially
    #Q(s2,*) set mac to 2.0 to define target
    agent.q_table[s] = np.zeros(len(Action), dtype=float)
    agent.q_table[s2] = np.array([0.0, 0.0, 2.0, 0.0], dtype=float) # max = 2.0

# Execute one learning update
    agent.learn(obs=s, action=a, reward=r, next_obs=s2, done=done)

    # target = 1.0 + 0.9*2.0 = 2.8
    # old Q(s,a) = 0.0
    # new Q(s,a) = 0.0 + 0.5*(2.8 - 0.0) = 1.4
    expected = 1.4

    a_index = list(Action).index(a)
    assert agent.q_table[s][a_index] == expected


def test_q_update_terminal_sets_target_to_reward_only():
    """
    If done=True, target should be reward only (no bootstrap from next state).
    """
    log_execution("TEST_RUN", "test_q_update_terminal_sets_target_to_reward_only")
    cfg = QConfig(alpha=0.5, gamma=0.9, epsilon=0.0)
    agent = QAgent(config=cfg, seed=123)

    s = (1, 1)
    s2 = (2, 2)  # doesn't matter when done=True
    a = Action.FORWARD
    r = 1.0
    done = True

    agent.q_table[s] = np.zeros(len(Action), dtype=float)
    agent.q_table[s2] = np.array([10.0, 10.0, 10.0, 10.0], dtype=float)  # should be ignored

    agent.learn(obs=s, action=a, reward=r, next_obs=s2, done=done)

    # target = reward = 1.0
    # new Q = 0 + 0.5*(1 - 0) = 0.5
    expected = 0.5

    a_index = list(Action).index(a)
    assert agent.q_table[s][a_index] == expected