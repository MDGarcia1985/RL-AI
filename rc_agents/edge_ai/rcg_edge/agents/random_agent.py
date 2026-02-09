"""
random_agent.py

Baseline Random Agent
Agent that selects actions uniformly at random.
Used to sanity-check the environment and runner before Q-learning.

Copyright (c) 2026 Michael Garcia, M&E Design
https://mandedesign.studio
michael@mandedesign.studio

CSC370 Spring 2026
"""

# The random agent is used as a baseline.
# It shows how the system behaves with no learning or memory.
# This makes it possible to see whether learning actually improves outcomes.


from __future__ import annotations
 
import numpy as np  # numpy is a standard scientific baseline for ML
from typing import Any

from .base import Action, StepResult #imports shared actino space and results container from the base agent

_ACTIONS = tuple(Action)

# The random agent simply selects an action from the action space at random.
# Mirrors the QAgent action-selection structure for consistency.
class RandomAgent:
    """A random agent that selects actions uniformly at random."""
    name = "random"

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.last_action: Action | None = None

    def reset(self) -> None:
        self.last_action = None

    def act(self, obs: Any) -> StepResult:
        action = _ACTIONS[int(self.rng.integers(0, len(_ACTIONS)))]
        self.last_action = action
        return StepResult(action=action, info={"policy": "random"})

    def learn(
        self,
        obs: Any,
        action: Action,
        reward: float,
        next_obs: Any,
        done: bool,
    ) -> None:
        """Random agent does not learn."""
        return None