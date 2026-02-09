"""
q_agent.py

Tabular Q-Learning Agent
Implements Q-learning with epsilon-greedy action selection.
Uses Q-table stored in NumPy arrays for state-action value function.

Copyright (c) 2026 Michael Garcia, M&E Design
https://mandedesign.studio
michael@mandedesign.studio

CSC370 Spring 2026
"""

from __future__ import annotations

from dataclasses import dataclass #dataclass gives a structured return type for act()
from typing import Any, Dict, Hashable #Hashable allows us to safely key the Q-table by state.

import numpy as np #used for fast argmax, arrays, and stable math.


from .base import Action, StepResult #imports shared actino space and results container from the base agent

# Ordered action list used to map Q-table indices → Action enums.
# This centralizes action ordering and simplifies future extensions.
_ACTIONS = tuple(Action)

@dataclass
class QConfig:
    alpha: float = 0.1 #learning rate a(alpha)
    gamma: float = 0.95 #discount factor y(gamma)
    epsilon: float = 0.1 #exploration rate e(epsilon)

class QAgent:
    """
    QAgent implements a Q-learning reinforcement learning agent.

    Unlike the RandomAgent, which selects actions without memory or learning,
    QAgent maintains an internal value function (Q-table) that estimates the
    expected return of taking a given action in a given state.

    The agent observes the environment, selects actions, receives rewards,
    and updates its Q-values according to the learning rule defined in learn().
    Over time, this allows the agent to improve its policy based on experience
    rather than random exploration alone.

    ...
    This class serves as a baseline learning agent for future extensions
    """
    name = "q_agent"

    def __init__(self, config: QConfig | None = None, seed: int | None = None):
        self.config = config if config is not None else QConfig()
        self.rng = np.random.default_rng(seed)

        # Transient per-episode state (cleared in reset()).
        self.last_action: Action | None = None

        #Q-table maps state -> array of Q-values 
        #once for each possible action in that state.
        #q_table[state] will be ~ array([Q_forward, Q_backward, Q_right, Q_left]).
        self.q_table: Dict[Hashable, np.ndarray] = {}

    #helper to ensure state exists in Q-table
    #convert observation to hashable type (tuple, string, etc.)
    #for use as dictionary key in q_table
    # NOTE about tuples:
    # - A tuple is an ordered, immutable collection of values.
    # - Immutability means the contents cannot change after creation.
    # - Because tuples are immutable, they are hashable and can be used
    #   safely as dictionary keys.
    # - In this project, (row, col) tuples represent fixed grid positions
    #   and serve as stable state identifiers for the Q-table.
    # - Lists are not used here because they are mutable and not hashable.

    def _state_key(self, obs: Any) -> Hashable:
        """
        Convert an observation into a hashable key for the Q-table
        """
        if isinstance(obs, np.ndarray):
            return tuple(obs)  # Convert numpy array to tuple for hashing
        if isinstance(obs, tuple):
            return obs  # Already hashable
        return str(obs)
    
    # NOTE: Q-table size grows with the number of visited states.
    # Unvisited states are initialized lazily on first encounter.
    def _ensure_state(self, key: Hashable) -> np.ndarray:
        """
        Ensure the given state key exists in the Q-table, 
        initializing if necessary.
        """
        if key not in self.q_table:
            # Initialize Q-values for all actions to zero
            self.q_table[key] = np.zeros(len(_ACTIONS), dtype=float)

        return self.q_table[key]
    
    def reset(self) -> None:
        """
        Reset per-episode internal state.

        NOTE:
        - The Q-table persists across episodes.
        - Transient episode-level state (e.g., last action taken)
        is cleared here.
        """
        self.last_action = None


    def act(self, obs: Any) -> StepResult:
        key = self._state_key(obs)
        q_values = self._ensure_state(key)

        # epsilon-greedy selection:
        # - explore with probability epsilon
        # - otherwise exploit the current best-known action
        if self.rng.random() < self.config.epsilon:
            # Explore: choose a random action (ignore Q-values)
            action = _ACTIONS[int(self.rng.integers(0, len(_ACTIONS)))]
            self.last_action = action
            return StepResult(action=action, info={"policy": "explore"})

        # Exploit: choose the action with highest Q(s, a)
        action = _ACTIONS[int(np.argmax(q_values))]
        self.last_action = action
        return StepResult(action=action, info={"policy": "exploit"})

        
    def learn(
        self,
        obs: Any,
        action: Action,
        reward: float,
        next_obs: Any,
        done: bool,
    ) -> None:
        """Update the Q-table based on the last experience."""
        s = self._state_key(obs)
        s2 = self._state_key(next_obs)

        q_s = self._ensure_state(s)
        q_s2 = self._ensure_state(s2)

        a_index = int(action)  # Action IntEnum values map directly to Q-value array indices.
        # Q-agent only learns from non-terminal states
        # A non-terminal state is any state where the episode is still in progress
        # (done == False). In these states, the agent can still act and expects
        # future rewards, so we include the discounted value of the next state.


        # Q-learning update
        # Q(s,a) <- Q(s,a) + alpha[reward + gamma*max_a'(Q(s',a')) - Q(s,a)]
        # Source: Sutton & Barto, Reinforcement Learning: An Introduction (2018), Ch. 6
        target = reward
        if not done:
            target = reward + self.config.gamma * float(np.max(q_s2))

        q_s[a_index] = q_s[a_index] + self.config.alpha * (target - q_s[a_index])
        return None