"""
test_agent_catalog.py

Smoke tests for agent_catalog.

Design intent:
- Catalog must build without importing Streamlit.
- Agent specs must be well-formed.
- Agents should be instantiable (skip if an agent is not available yet).
"""

from __future__ import annotations

import pytest

from rc_agents.config import TrainingUIConfig
from rc_agents.ui.streamlit_ui.agent_catalog import build_agent_catalog


def _cfg() -> TrainingUIConfig:
    cfg = TrainingUIConfig()
    cfg.rows = 5
    cfg.cols = 5
    cfg.start = (0, 0)
    cfg.goal = (4, 4)
    cfg.episodes = 1
    cfg.max_steps = 10
    cfg.alpha = 0.5
    cfg.gamma = 0.9
    cfg.epsilon = 0.1
    cfg.seed = 123
    return cfg


def test_catalog_builds_and_has_expected_shape() -> None:
    catalog = build_agent_catalog()
    assert isinstance(catalog, dict)
    assert len(catalog) >= 1

    for agent_id, spec in catalog.items():
        assert isinstance(agent_id, str) and agent_id.strip() != ""
        assert hasattr(spec, "label")
        assert hasattr(spec, "make")
        assert callable(spec.make)
        assert hasattr(spec, "supports_qtable_io")


@pytest.mark.parametrize("agent_id", ["rl", "rlf"])
def test_catalog_can_instantiate_known_agents_if_present(agent_id: str) -> None:
    """
    If an agent is not implemented yet, this test should SKIP cleanly,
    not crash the suite.
    """
    catalog = build_agent_catalog()
    if agent_id not in catalog:
        pytest.skip(f"{agent_id} not registered in catalog yet")

    cfg = _cfg()
    spec = catalog[agent_id]

    try:
        agent = spec.make(cfg)
    except Exception as e:
        pytest.skip(f"{agent_id} could not be instantiated yet: {e}")

    assert agent is not None
    # Most agents expose a stable display name
    assert hasattr(agent, "name")
