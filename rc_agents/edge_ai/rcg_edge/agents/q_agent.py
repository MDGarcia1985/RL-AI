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

from dataclasses import dataclass  # dataclass gives a structured return type for act()
from typing import Any, Dict, Hashable  # Hashable allows us to safely key the Q-table by state.

import io  # Used for in-memory bytes objects (Streamlit upload/download).
from pathlib import Path  # Cross-platform path handling.
import numpy as np  # used for fast argmax, arrays, and stable math.

from .base import Action, StepResult  # shared action space + result container

# Ordered action list used to map Q-table indices → Action enums.
# This centralizes action ordering and simplifies future extensions.
_ACTIONS = tuple(Action)


@dataclass
class QConfig:
    alpha: float = 0.1  # learning rate a(alpha)
    gamma: float = 0.95  # discount factor y(gamma)
    epsilon: float = 0.1  # exploration rate e(epsilon)


class QAgent:
    """
    QAgent implements a Q-learning reinforcement learning agent.

    Unlike the RandomAgent, which selects actions without memory or learning,
    QAgent maintains an internal value function (Q-table) that estimates the
    expected return of taking a given action in a given state.
    """

    name = "q_agent"

    def __init__(self, config: QConfig | None = None, seed: int | None = None):
        self.config = config if config is not None else QConfig()
        self.rng = np.random.default_rng(seed)

        # Transient per-episode state (cleared in reset()).
        self.last_action: Action | None = None

        # Q-table maps state -> array of Q-values (one per action).
        # q_table[state] ~= np.array([Q_forward, Q_backward, Q_right, Q_left])
        self.q_table: Dict[Hashable, np.ndarray] = {}

    def _state_key(self, obs: Any) -> Hashable:
        """
        Convert an observation into a hashable key for the Q-table.
        Phase 1 expects (row, col) tuples as state IDs.
        """
        if isinstance(obs, np.ndarray):
            return tuple(obs.tolist())
        if isinstance(obs, tuple):
            return obs
        return str(obs)

    def _ensure_state(self, key: Hashable) -> np.ndarray:
        """
        Ensure the given state key exists in the Q-table, initializing if necessary.
        """
        if key not in self.q_table:
            self.q_table[key] = np.zeros(len(_ACTIONS), dtype=float)
        return self.q_table[key]

    def reset(self) -> None:
        """Reset per-episode internal state (Q-table persists)."""
        self.last_action = None

    def act(self, obs: Any) -> StepResult:
        key = self._state_key(obs)
        q_values = self._ensure_state(key)

        # ε-greedy:
        # - explore with probability epsilon
        # - otherwise exploit best-known action
        if self.rng.random() < self.config.epsilon:
            action = _ACTIONS[int(self.rng.integers(0, len(_ACTIONS)))]
            self.last_action = action
            return StepResult(action=action, info={"policy": "explore"})

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

        a_index = int(action)  # Action(IntEnum) maps directly to index.

        # Q-learning update:
        # Q(s,a) <- Q(s,a) + alpha[reward + gamma*max_a' Q(s',a') - Q(s,a)]
        target = float(reward)
        if not done:
            target = float(reward) + self.config.gamma * float(np.max(q_s2))

        q_s[a_index] = q_s[a_index] + self.config.alpha * (target - q_s[a_index])
        return None

    # -------------------------------------------------------------------------
    # Persistence: NPZ (portable) save/load for progressive learning across runs
    #
    # Format: .npz with arrays [states (N,2) row,col | qvals (N,num_actions)],
    # scalars alpha, gamma, epsilon, num_actions, and format tag fmt.
    # allow_pickle=False on load to avoid arbitrary code execution from untrusted files.
    # -------------------------------------------------------------------------

    def to_bytes(self) -> bytes:
        """
        Serialize Q-table + hyperparameters into a compressed .npz blob (bytes).

        Assumes Phase 1 state keys are (row, col) tuples.
        Non-(row,col) keys are ignored (future-proofing if you later add richer states).
        """
        buf = io.BytesIO()

        states: list[list[int]] = []
        qvals: list[np.ndarray] = []

        for k, v in self.q_table.items():
            if isinstance(k, tuple) and len(k) == 2:
                r, c = int(k[0]), int(k[1])
                states.append([r, c])
                qvals.append(np.asarray(v, dtype=float))
            else:
                # TODO: Support non-grid state keys with a more general serializer.
                continue

        # Empty q_table: write (0,2) and (0,num_actions) so from_bytes gets valid arrays.
        if len(states) == 0:
            states_arr = np.zeros((0, 2), dtype=int)
            qvals_arr = np.zeros((0, len(_ACTIONS)), dtype=float)
        else:
            states_arr = np.asarray(states, dtype=int)
            qvals_arr = np.stack(qvals, axis=0).astype(float, copy=False)

        np.savez_compressed(
            buf,
            states=states_arr,
            qvals=qvals_arr,
            alpha=float(self.config.alpha),
            gamma=float(self.config.gamma),
            epsilon=float(self.config.epsilon),
            num_actions=int(len(_ACTIONS)),
            fmt=np.array(["grid_tuple_v1"]),  # version tag for future format checks
        )
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes, seed: int | None = None) -> "QAgent":
        """
        Reconstruct a QAgent from bytes produced by to_bytes().
        State keys are restored as (row, col) = (states[i,0], states[i,1]) to match save order.
        """
        buf = io.BytesIO(data)
        z = np.load(buf, allow_pickle=False)  # safe: no unpickling of untrusted data

        alpha = float(z["alpha"])
        gamma = float(z["gamma"])
        epsilon = float(z["epsilon"])

        agent = cls(QConfig(alpha=alpha, gamma=gamma, epsilon=epsilon), seed=seed)

        states = z["states"]
        qvals = z["qvals"]
        # Column 0 = row, column 1 = col; qvals[i] is Q(s,a) for all actions at state (r,c).
        for i in range(states.shape[0]):
            r = int(states[i, 0])
            c = int(states[i, 1])
            agent.q_table[(r, c)] = np.asarray(qvals[i], dtype=float)

        return agent

    def save_npz(self, path: str | Path) -> None:
        """Save Q-table to disk as .npz (Streamlit download / offline reuse)."""
        Path(path).write_bytes(self.to_bytes())

    @classmethod
    def load_npz(cls, path: str | Path, seed: int | None = None) -> "QAgent":
        """Load Q-table from disk .npz; seed sets RNG for future exploration."""
        return cls.from_bytes(Path(path).read_bytes(), seed=seed)