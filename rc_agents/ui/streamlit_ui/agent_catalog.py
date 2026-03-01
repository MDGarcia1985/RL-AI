"""
agent_catalog.py

Design intent:
    Call different agents from factory.py
    - Each has a selection check box
    - When an agent is checked, they are called and connected to the env
    - Agent parameters are set in the sidebar
    - Agents share the same parameters to test effectiveness


    Future improvements
    - Each agent has its own tab
    - Tab names are the agent names
    - Tabs contain controls for that agent's parameters
    - Agents are instantiated when their tab is first selected
    - Agents are destroyed when their tab is closed


Copyright (c) 2026 Michael Garcia, M&E Design
https://mandedesign.studio
CSC370 Spring 2026
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from rc_agents.config import TrainingUIConfig

# ---------------------------------------------------------------------------
# Agent Catalog Model
#
# Why:
# - Streamlit reruns everything; import-time failures are brutal.
# - Catalog MUST be safe to import even if an agent module is broken/WIP.
# - So: "make" functions do LOCAL imports (lazy import).
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentSpec:
    """
    Declarative spec for a selective agent.

    agent_id:
        Stable identifier used in UI + persistence keys.
        Examples: "rl", "rlf"

    lebel:
        Human-readable label for display in UI.
        Examples: "Q-Learning", "Q-Learning (Fine-Tuned)"

    make:
        Callable that istantiates the agent fromthe shared TrainingUIConfig.
        Examples:
            lambda config: QLearningAgent(config)
            lambda config: QLearningAgent(config, fine_tune=True)

    supports_qtable_io:
        If True, the UI can offer Download/upload Q-table features.
        (Requires the agent to impliment to_bytes()/from_bytes()).
    """
    agent_id: str
    label: str
    make: Callable[[TrainingUIConfig], object]
    supports_qtable_io: bool = False
    description: str = ""

def build_agent_catalog() -> Dict[str, AgentSpec]:
    """
    Return the full agent catalog.

    NOTE:
    - Keep this funtion import-safe (i.e. )
    - All heavy imports happen inside each make_* function
    """

    def make_rl(cfg: TrainingUIConfig) -> object:
        # Local import prevents import-time failures when experimenting.
        from rc_agents.edge_ai.rcg_edge.agents.rl_agent import RLAgent, RLConfig
        return RLAgent(
            RLConfig(alpha=cfg.alpha, gamma=cfg.gamma, epsilon=cfg.epsilon),
            seed=cfg.seed,
        )

    def make_rlf(cfg: TrainingUIConfig) -> object:
        from rc_agents.edge_ai.rcg_edge.agents.rlf_agent import RLFAgent, RLFConfig
        return RLFAgent(
            RLFConfig(alpha=cfg.alpha, gamma=cfg.gamma, epsilon=cfg.epsilon),
            seed=cfg.seed,
        )

    # Add more agents here as they become stable.
    # Keep them lazy-loaded like above.
    catalog: Dict[str, AgentSpec] = {
        "rl": AgentSpec(
            agent_id="rl",
            label="RL Base Agent",
            make=make_rl,
            supports_qtable_io=True,
            description="Baseline Q-learning agent.",
        ),
        "rlf": AgentSpec(
            agent_id="rlf",
            label="RL with Fractal Exploration (RLF)",
            make=make_rlf,
            supports_qtable_io=True,
            description="Exploration-augmented Q-learning agent.",
        ),
    }

    return catalog