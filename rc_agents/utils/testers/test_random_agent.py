"""
test_random_agent.py

Test Suite for Random Agent
Unit tests for the baseline random agent implementation.

Copyright (c) 2026 Michael Garcia, M&E Design
https://mandedesign.studio
michael@mandedesign.studio

CSC370 Spring 2026
"""

from rc_agents.edge_ai.rcg_edge.agents.random_agent import RandomAgent
from rc_agents.edge_ai.rcg_edge.agents.base import Action, StepResult
from rc_agents.utils.logger import log_execution

def test_random_agent_act_returns_valid_action():
    log_execution("TEST_RUN", "test_random_agent_act_returns_valid_action")
    agent = RandomAgent()  # construct with no args
    result = agent.act(obs=0)  # simple observation - fixed positional/keyword argument issue
    assert isinstance(result, StepResult)  # type check verifies contract
    assert isinstance(result.action, Action)  # type check verifies enum output
    assert result.action in Action  # verifies it's one of the legal enum values
    assert result.info.get("source") == "numpy_rng"  # fixed dictionary access syntax
    assert result.info is not None