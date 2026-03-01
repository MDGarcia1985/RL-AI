"""
progressive_learning.py

Shared helpers for:
- persistent agents across Streamlit reruns
- multi-agent training runs (RL + RLF)
- training context keys (prevents Q-table mixing)

Design intent:
- Streamlit reruns everything on any UI event.
- st.session_state is our RAM.
- We preserve progressive learning WITHOUT silently cross-contaminating agents.

Copyright (c) 2026 Michael Garcia, M&E Design
https://mandedesign.studio
CSC370 Spring 2026
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

from rc_agents.config import TrainingUIConfig
from rc_agents.edge_ai.rcg_edge.runners import run_training
from rc_agents.envs import GridEnv


def ensure_state() -> None:
    """
    Ensure Streamlit session storage exists.

    Why:
    - Prevent KeyError and keep app deterministic.
    - Gives you clean named buckets for debugging.
    """
    if "agent_store" not in st.session_state:
        st.session_state.agent_store = {}  # (agent_id, ctx_key) -> agent
    if "history_store" not in st.session_state:
        st.session_state.history_store = {}  # agent_id -> results list
    if "last_scoreboard" not in st.session_state:
        st.session_state.last_scoreboard = []  # list[dict]


def training_context_key(cfg: TrainingUIConfig, *, game_type: str) -> tuple:
    """
    Defines the training context.

    If any of these change, it should be treated like a different experiment.
    That means new agent instance (fresh Q-table), unless you intentionally transfer.

    Includes:
    - env geometry (rows, cols)
    - start/goal
    - hyperparams (alpha/gamma/epsilon)
    - seed
    - game_type (open world vs maze)
    """
    return (
        str(game_type),
        int(cfg.rows),
        int(cfg.cols),
        tuple(cfg.start),
        tuple(cfg.goal),
        float(cfg.alpha),
        float(cfg.gamma),
        float(cfg.epsilon),
        int(cfg.seed or 0),
    )


def make_agent(agent_id: str, cfg: TrainingUIConfig):
    """
    Agent factory.

    Why:
    - UI shouldn't care about constructors.
    - Keeps imports localized so broken WIP agents don't kill the whole UI.
    """
    if agent_id == "rl":
        from rc_agents.edge_ai.rcg_edge.agents.rl_agent import RLAgent, RLConfig
        return RLAgent(RLConfig(alpha=cfg.alpha, gamma=cfg.gamma, epsilon=cfg.epsilon), seed=cfg.seed)

    if agent_id == "rlf":
        from rc_agents.edge_ai.rcg_edge.agents.rlf_agent import RLFAgent, RLFConfig
        return RLFAgent(RLFConfig(alpha=cfg.alpha, gamma=cfg.gamma, epsilon=cfg.epsilon), seed=cfg.seed)

    raise ValueError(f"Unknown agent_id: {agent_id}")


def make_env(cfg: TrainingUIConfig, *, game_type: str):
    """
    Environment factory.

    Today:
    - Open World -> GridEnv
    - Maze -> MazeEnv (if present)

    Future:
    - Labyrinth -> different rules / multi-goals / hazard states
    """
    if game_type == "Open World":
        return GridEnv(cfg.to_grid_config())

    if game_type == "Maze":
        from rc_agents.envs.maze_env import MazeEnv, MazeConfig  # adjust if your path differs
        mcfg = MazeConfig(
            rows=int(cfg.rows),
            cols=int(cfg.cols),
            start=tuple(cfg.start),
            goal=tuple(cfg.goal),
        )
        return MazeEnv(mcfg)

    raise NotImplementedError("Labyrinth is a future environment type.")


def summarize_results(results) -> tuple[float, Optional[int], float]:
    """
    Summary metrics used in scoreboard.

    Returns:
        avg_steps_all: average steps across all episodes
        best_steps_on_win: best single run (min steps among wins) or None
        win_rate: wins / episodes
    """
    if not results:
        return float("nan"), None, 0.0

    avg_steps_all = float(np.mean([r.steps for r in results]))
    wins = [r for r in results if bool(r.reached_goal)]
    best_steps_on_win = min((r.steps for r in wins), default=None)
    win_rate = float(len(wins) / len(results))
    return avg_steps_all, best_steps_on_win, win_rate


def convergence_fields(agent) -> tuple[Optional[int], Optional[str], Optional[int], Optional[int]]:
    """
    Pull convergence signals from the agent.

    train_runner sets:
        agent.convergence_summary = tracker.summary()

    If agent hasn't trained yet, this will be missing.
    """
    s = getattr(agent, "convergence_summary", None)
    if s is None:
        return None, None, None, None

    return (
        getattr(s, "episode_first_saturation", None),
        getattr(s, "saturation_reason", None),
        getattr(s, "episode_first_perfect", None),
        getattr(s, "episode_first_steps_plateau", None),
    )


def run_selected_agents(
    *,
    cfg: TrainingUIConfig,
    game_type: str,
    selected_agent_ids: List[str],
) -> tuple[List[Dict[str, object]], Dict[str, object]]:
    """
    Core training driver for Streamlit.

    Why:
    - Keeps main_panel.py simple.
    - Makes "run tournament" behavior testable later.

    Returns:
        scoreboard: list of dict rows for table display
        details: dict payload for JSON export / debugging
    """
    ensure_state()

    ctx_key = training_context_key(cfg, game_type=game_type)

    scoreboard: List[Dict[str, object]] = []
    details: Dict[str, object] = {}

    for agent_id in selected_agent_ids:
        env = make_env(cfg, game_type=game_type)
        store_key = (agent_id, ctx_key)

        if store_key not in st.session_state.agent_store:
            st.session_state.agent_store[store_key] = make_agent(agent_id, cfg)

        agent = st.session_state.agent_store[store_key]

        results = run_training(env=env, agent=agent, cfg=cfg)
        st.session_state.history_store[agent_id] = results

        avg_steps, best_steps, win_rate = summarize_results(results)
        sat_ep, sat_reason, perf_ep, steps_plateau_ep = convergence_fields(agent)

        row = {
            "agent": getattr(agent, "name", agent_id),
            "win_rate": win_rate,
            "avg_steps_all": avg_steps,
            "best_steps_on_win": best_steps,
            "saturation_ep": sat_ep,
            "saturation_reason": sat_reason,
            "perfect_ep": perf_ep,
            "steps_plateau_ep": steps_plateau_ep,
        }
        scoreboard.append(row)

        details[agent_id] = {
            "agent": getattr(agent, "name", agent_id),
            "context_key": list(ctx_key),
            "cfg": {
                "episodes": int(cfg.episodes),
                "max_steps": int(cfg.max_steps),
                "rows": int(cfg.rows),
                "cols": int(cfg.cols),
                "start": list(cfg.start),
                "goal": list(cfg.goal),
                "alpha": float(cfg.alpha),
                "gamma": float(cfg.gamma),
                "epsilon": float(cfg.epsilon),
                "seed": int(cfg.seed or 0),
                "game_type": str(game_type),
            },
            "scoreboard_row": row,
            "convergence_summary": getattr(agent, "convergence_summary", None).__dict__
            if getattr(agent, "convergence_summary", None) is not None
            else None,
            "episode_results": [r.__dict__ for r in results],
        }

    st.session_state.last_scoreboard = scoreboard
    return scoreboard, details