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
from typing import Any, Dict, List, Protocol

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


def run_training(
    env: Env,
    agent: Any,
    cfg: TrainingUIConfig, # This loads user input for learning parameters.
) -> List[EpisodeResult]:
    """Run episodes of interaction between env and agent."""
    results: List[EpisodeResult] = []

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
        obs = env.reset() # Starts a fresh episode.
        agent.reset() # Starts a fresh episode for the agent.

        total_reward = 0.0 # Resets reward accumulator for the episode.
        steps = 0
        done = False # Tracks whether the episode has terminated.

        for _ in range(cfg.max_steps): # Safety limit to prevent infinite wandering.
            steps += 1

            step_result: StepResult = agent.act(obs)
            action: Action = step_result.action  # IntEnum; safe to pass as int

            next_obs, reward, done, info = env.step(int(action))
            done = bool(done)  # normalize (handles numpy.bool_ and other truthy types)           
            total_reward += float(reward)

            agent.learn(               # Order matters here.
                obs=obs,               # What it saw.
                action=action,         # What it did.
                reward=float(reward),  # What it got.
                next_obs=next_obs,     # What happened next.
                done=done,       # Whether the episode ended.
            )

            obs = next_obs # Updates observation for next iteration.
            if done:
                break # End episode early if terminal state is reached.

        # reached_goal is intentionally derived from info when available.
        # This future-proofs the runner when we add terminal failure states.
        reached_goal = bool(info.get("reached_goal", done))

        # Record episode metrics for UI display and debugging.
        results.append(
            
            EpisodeResult(
                episode=ep,
                steps=steps,
                total_reward=total_reward,
                reached_goal=reached_goal,
            )
        )

        # Update convergence tracker AFTER episode result is finalized
        tracker.update(
            reached_goal=reached_goal,
            steps=steps,
        )

    # Expose convergence summary without breaking the current return type.
    # NOTE: This is intentionally lightweight. UI can surface it later.
    agent.convergence_summary = tracker.summary()

    return results
