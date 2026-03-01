"""
train_runner.py

Training Loop Runner
Wires together an environment and an agent, runs episodes, returns metrics.
Coordinates the training process for reinforcement learning agents.

Copyright (c) 2026 Michael Garcia, M&E Design
https://mandedesign.studio
michael@mandedesign.studio

CSC370 Spring 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np

from .convergence_tracker import ConvergenceTracker
from rc_agents.config.ui_config import TrainingUIConfig
from rc_agents.edge_ai.rcg_edge.agents import Action, StepResult

class Env(Protocol):
    """
    Minimal environment interface required by the training runner.

    NOTE:
    - This interface mirrors classic Gym-style environments.
    - Observation type is intentionally left generic (Any) to allow
      different environment representations.
    """

    def reset(self) -> Any:
        """
        Start a new episode and return the initial observation.
        """
        ...

    def step(self, action: int) -> tuple[Any, float, bool, Dict[str, object]]:
        """
        Apply an action and return:
        (observation, reward, done, info)
        """
        ...

@dataclass
class EpisodeResult:
    episode: int
    steps: int
    total_reward: float
    reached_goal: bool


# ---------------------------------------------------------------------------
# Trajectory recording (refactor: for "best run" trail viz in main_panel)
# ---------------------------------------------------------------------------

def _obs_to_pos(obs: Any) -> Tuple[int, int]:
    """
    Convert observation to (row, col) for trajectory recording.
    GridEnv and MazeEnv both use (row, col) as obs; support tuple, list, or
    numpy array so we stay generic for the Env protocol.
    """
    if isinstance(obs, (tuple, list)) and len(obs) >= 2:
        return (int(obs[0]), int(obs[1]))
    if isinstance(obs, np.ndarray) and obs.size >= 2:
        return (int(obs.flat[0]), int(obs.flat[1]))
    return (0, 0)


def run_training(
    env: Env,
    agent: Any,
    cfg: TrainingUIConfig,  # This loads user input for learning parameters.
) -> Tuple[List[EpisodeResult], Optional[List[Tuple[int, int]]]]:
    """
    Run episodes of interaction between env and agent.
    Returns (results, best_trajectory).
    best_trajectory: list of (row, col) for the best successful run (fewest steps
    to goal), or None if no episode reached the goal. Used by main_panel for
    the "Best run (trail)" graph.
    """
    results: List[EpisodeResult] = []
    best_trajectory: Optional[List[Tuple[int, int]]] = None
    best_steps: Optional[int] = None  # fewest steps among runs that reached_goal

    # ------------------------------------------------------------------
    # Convergence tracking
    #
    # Tracks rolling performance to detect:
    # - Perfect win-rate windows
    # - Win-rate plateau (saturation)
    # - Efficiency plateau (steps-to-goal)
    #
    # NOTE:
    # - This does NOT alter training behavior.
    # - It only observes and records metrics.
    # ------------------------------------------------------------------
    tracker = ConvergenceTracker(
        window=min(200, cfg.episodes),  # Avoid oversized window
        delta=0.005,
        patience=5,
    )

    # Optional deterministic seeding.
    # RNG seeding is applied once per training run to ensure reproducibility
    # while preserving stochastic exploration within episodes.
    #
    # NOTE:
    # - Reseeding inside episodes or per step collapses exploration by
    #   replaying identical action-selection sequences.
    # - This can silently break ε-greedy or stochastic policies by making
    #   learning appear deterministic.
    #
    # The environment may or may not support seeding; this is handled
    # defensively to keep the runner generic.
    if cfg.seed is not None:
        try:
            env.seed(cfg.seed)  # Apply seed if the environment exposes a seed() method.
        except AttributeError:
            pass

    for ep in range(1, cfg.episodes + 1):
        obs = env.reset()  # Starts a fresh episode.
        agent.reset()  # Starts a fresh episode for the agent.

        # Refactor: record full path (row, col) each step for best-run trail viz.
        # We only keep the single best trajectory (fewest steps to goal) to avoid
        # storing one path per episode.
        trajectory: List[Tuple[int, int]] = [_obs_to_pos(obs)]
        total_reward = 0.0  # Resets reward accumulator for the episode.
        steps = 0
        done = False  # Tracks whether the episode has terminated.

        for _ in range(cfg.max_steps):  # Safety limit to prevent infinite wandering.
            steps += 1

            step_result: StepResult = agent.act(obs)
            action: Action = step_result.action  # IntEnum; safe to pass as int

            next_obs, reward, done, info = env.step(int(action))
            done = bool(done)  # normalize (handles numpy.bool_ and other truthy types)
            total_reward += float(reward)
            trajectory.append(_obs_to_pos(next_obs))  # extend path for this episode

            agent.learn(
                obs=obs,
                action=action,
                reward=float(reward),
                next_obs=next_obs,
                done=done,
            )

            obs = next_obs
            if done:
                break

        # reached_goal is intentionally derived from info when available.
        # This future-proofs the runner when we add terminal failure states.
        reached_goal = bool(info.get("reached_goal", done))

        # Keep best trajectory: among episodes that reached the goal, the one
        # with fewest steps. main_panel uses this for the single "Best run (trail)" graph.
        if reached_goal and (best_steps is None or steps < best_steps):
            best_steps = steps
            best_trajectory = list(trajectory)

        results.append(
            EpisodeResult(
                episode=ep,
                steps=steps,
                total_reward=total_reward,
                reached_goal=reached_goal,
            )
        )

        tracker.update(reached_goal=reached_goal, steps=steps)

    # Expose convergence summary without breaking the current return type.
    agent.convergence_summary = tracker.summary()
    return (results, best_trajectory)
