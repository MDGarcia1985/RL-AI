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

from rc_agents.config.ui_config import TrainingUIConfig
from rc_agents.edge_ai.rcg_edge.agents.base import Action, StepResult


class Env(Protocol):
    """Minimal env protocol required by the runner."""
    def reset(self) -> Any: ...
    # Starts a new episode
    # Returns initial observation (obs)
    # Note: the type of obs is intentionally left generic (any)
    def step(self, action: int) -> tuple[Any, float, bool, Dict[str, object]]: ...
    # Takes an action and returns (next_obs, reward, done, info)

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

    # Optional deterministic seeding
    # RNG seeding happens once per training run.
    # Reseeding inside episodes or steps collapses the exploration
    # and prevents meaningful learning.
    if cfg.seed is not None:
        try:
            env.seed(cfg.seed)  # if the env supports it
        except AttributeError:
            pass

    for ep in range(1, cfg.episodes + 1):
        obs = env.reset() # Starts a fresh episode.
        agent.reset() #Starts a fresh episode for the agent.

        total_reward = 0.0 # Resets rewaard accumulator for the episode.
        steps = 0
        done = False # Tracks whether the epside has terminated.

        for _ in range(cfg.max_steps): # Safety limit and prevent infinite wandering.
            steps += 1

            step_result: StepResult = agent.act(obs)
            action: Action = step_result.action  # IntEnum; safe to pass as int

            next_obs, reward, done, info = env.step(int(action))
            total_reward += float(reward)

            agent.learn(               # Order matters here.
                obs=obs,               # What it saw.
                action=action,         # What it did.
                reward=float(reward),  # What it got.
                next_obs=next_obs,     # What happened next.
                done=bool(done),       # Whether the episode ended.
            )

            obs = next_obs # Updates observation for next iteration.
            if done:
                break # Breaks loop if goal or max steps is reached.

        results.append(                     # 
            EpisodeResult(
                episode=ep,
                steps=steps,
                total_reward=total_reward,
                reached_goal=done,
            )
        )

    return results
