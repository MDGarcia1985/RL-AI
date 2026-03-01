"""
rlf_agent.py

RLF Agent (Tabular Q-Learning + Fractal Exploration)
Q-learning with epsilon-greedy, but exploration actions are chosen using
Julia-set dynamics to generate a continuous heading θ(theta) ∈ [0, 2π(pi)),
then mapped into a smooth preference over the 4 discrete grid actions.

This keeps compatibility with the current 4-action GridEnv while making
exploration deterministic-chaotic rather than uniform random.

Copyright (c) 2026 Michael Garcia, M&E Design
https://mandedesign.studio
michael@mandedesign.studio

CSC370 Spring 2026
"""

from __future__ import annotations


import logging
import math  # Used for the Julia-set dynamics

from dataclasses import dataclass
from typing import Any, Dict, Hashable

import io  # Used for in-memory bytes objects (Streamlit upload/download).
from pathlib import Path  # Cross-platform path handling.
import numpy as np  # used for fast argmax, arrays, and stable math.

from .base import Action, StepResult

logger = logging.getLogger(__name__)

_ACTIONS = tuple(Action)

# ---------------------------------------------------------------------------
# Movement stub (next TODO)
#
# This is a forward-looking stub for expanding the action space to include
# diagonals (8-way movement) and/or continuous headings later.
#
# NOTE:
# - Today, Action is a 4-direction enum (FORWARD/BACKWARD/LEFT/RIGHT).
# - The diagonal actions below may not exist yet; we guard them so this file
#   remains import-safe until you extend the Action enum.
# ---------------------------------------------------------------------------

# Existing 4-way deltas (row, col) in grid coordinates.
_ACTION_DELTAS: dict[Action, tuple[int, int]] = {
    Action.FORWARD: (-1, 0),
    Action.BACKWARD: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}

# Optional diagonal deltas (safe if Action doesn't define these yet).
# TODO: When Action enum is extended, these will automatically activate.
for _name, _delta in {
    "NORTHWEST": (-1, -1),
    "NORTHEAST": (-1, 1),
    "SOUTHWEST": (1, -1),
    "SOUTHEAST": (1, 1),
}.items():
    _maybe = getattr(Action, _name, None)
    if _maybe is not None:
        _ACTION_DELTAS[_maybe] = _delta


def apply_action_delta_stub(row: int, col: int, action: Action) -> tuple[int, int]:
    """
    Stub helper for future use.

    Intended pattern:

        d_row, d_col = ACTION_DELTAS[action]
        row += d_row
        col += d_col

    For now, GridEnv owns movement. This exists to centralize deltas once we
    expand beyond the current 4-action grid world.
    """
    d_row, d_col = _ACTION_DELTAS.get(action, (0, 0))
    row += d_row
    col += d_col
    return row, col


@dataclass
class RLFConfig:  # Same baseline as rl_agent and q_agent
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon: float = 0.1

    # Fractal exploration parameters
    # This training agent uses a subset of fractal parameters that work well for 4-action grid worlds.
    # See https://www.youtube.com/watch?v=9991JlKnFmk for a video of the Julia-set dynamics.
    #
    # The purpose of this training agent is to compare and prepare for implementation of mando_agent.
    # mando_agent will use the full set of fractal parameters present in the Mandelbrot set.

    # Fractal exploration: z_{n+1} = z_n^2 + c
    # Julia parameter c = c_real + i(c_imag)
    c_real: float = -0.8
    c_imag: float = 0.156

    # Initial z0 = z0_real + i(z0_imag)
    z0_real: float = 0.0
    z0_imag: float = 0.0

    # κ(kappa): sharpness of mapping from θ(theta) to action preference
    # Higher κ(kappa) → more “nearest-direction”; lower κ(kappa) → more diffuse.
    kappa: float = 6.0

    # If True, reset z to z0 at each episode reset.
    # If False, z continues across episodes (often better coverage).
    reset_z_each_episode: bool = False


class JuliaScout:
    """
    Julia dynamics iterator:
        z_{n+1} = z_n^2 + c

    Produces a continuous θ(theta) heading from the complex plane via atan2.
    """

    # Create c as a complex number
    def __init__(self, c: complex, z0: complex):
        self.c = c
        self.z0 = z0
        self.z = z0

    # Reset the Julia dynamics iterator to z0.
    # If per_episode is True, reset z to z0 at each episode reset.
    def reset(self, *, per_episode: bool) -> None:
        if per_episode:
            self.z = self.z0

    # Step the Julia dynamics and return the resulting θ(theta) heading.
    # For more information on atan2 see:
    # https://en.wikipedia.org/wiki/Atan2
    def step_theta(self) -> float:
        self.z = self.z * self.z + self.c
        theta = math.atan2(self.z.imag, self.z.real)  # [-π(pi), π(pi)]
        return theta % (2.0 * math.pi)  # [0, 2π(pi))


def _ang_dist(a: float, b: float) -> float:
    """Shortest angular distance on a circle."""
    d = abs(a - b) % (2.0 * math.pi)
    return min(d, 2.0 * math.pi - d)


def _action_from_theta_soft(theta: float, rng: np.random.Generator, kappa: float) -> Action:
    """
    Continuous θ(theta) -> smooth probability over 4 actions.

    Headings:
      RIGHT    = 0
      FORWARD  = π(pi)/2
      LEFT     = π(pi)
      BACKWARD = 3π(pi)/2
    """
    headings = {
        Action.RIGHT: 0.0,
        Action.FORWARD: math.pi / 2.0,
        Action.LEFT: math.pi,
        Action.BACKWARD: 3.0 * math.pi / 2.0,
    }

    actions = list(headings.keys())
    dists = np.array([_ang_dist(theta, headings[a]) for a in actions], dtype=float)

    # w = exp(-κ(kappa) * d^2)
    w = np.exp(-float(kappa) * (dists ** 2))                            # Gives unnormalized weights.
    s = float(np.sum(w))                                                # Is the total weight.
    if not np.all(np.isfinite(w)) or s <= 0.0 or not np.isfinite(s):    # If any weights are invalid, fallback to uniform random.
        logger.warning(
            "Invalid weights in _action_from_theta_soft: w=%s, s=%s",
            w,
            s,
        )
        return actions[int(rng.integers(0, len(actions)))]

    p = w / s # Where p is the probability distribution over actions.

    # Sample action from the soft distribution.
    idx = int(rng.choice(len(actions), p=p))
    return actions[idx]


class RLFAgent:
    """
    RLF agent = Q-learning baseline + Julia-driven exploration.

    - Explore (prob ε(epsilon)): use JuliaScout → θ(theta) → soft 4-action draw
    - Exploit: argmax_a Q(s,a)
    """

    name = "rlf_agent"

    def __init__(self, config: RLFConfig | None = None, seed: int | None = None):
        self.config = config if config is not None else RLFConfig()
        self.rng = np.random.default_rng(seed)

        self.last_action: Action | None = None
        self.q_table: Dict[Hashable, np.ndarray] = {}

        # Initialize Julia set scout for fractal exploration
        c = complex(self.config.c_real, self.config.c_imag)
        z0 = complex(self.config.z0_real, self.config.z0_imag)
        self.scout = JuliaScout(c=c, z0=z0)

    # Convert observation to hashable key for Q-table.
    # If obs is a numpy array, convert to tuple.
    # If obs is a tuple, keep as-is.
    # Otherwise, convert to string.
    def _state_key(self, obs: Any) -> Hashable:
        if isinstance(obs, np.ndarray):
            return tuple(obs.tolist())
        if isinstance(obs, tuple):
            return obs
        return str(obs)

    # Ensure state key exists in Q-table, initializing if necessary.
    # Returns a reference to the Q-table entry for the given state.
    def _ensure_state(self, key: Hashable) -> np.ndarray:
        if key not in self.q_table:
            self.q_table[key] = np.zeros(len(_ACTIONS), dtype=float)
        return self.q_table[key]

    # Reset per-episode internal state (Q-table persists).
    def reset(self) -> None:
        self.last_action = None
        self.scout.reset(per_episode=bool(self.config.reset_z_each_episode))

    # Select an action for the given observation.
    # Returns a StepResult with the selected action and additional info.
    def act(self, obs: Any) -> StepResult:
        key = self._state_key(obs)
        q_values = self._ensure_state(key)

        if self.rng.random() < self.config.epsilon:
            theta = self.scout.step_theta()
            action = _action_from_theta_soft(theta, rng=self.rng, kappa=float(self.config.kappa))
            self.last_action = action
            # TODO: Save theta for analysis/tracing.
            # TODO: Consider returning additional scout state in info dict for analysis.
            return StepResult(action=action, info={"policy": "explore_fractal_theta", "theta": float(theta)})

        action = _ACTIONS[int(np.argmax(q_values))]
        self.last_action = action
        return StepResult(action=action, info={"policy": "exploit"})

    # Update Q-table based on the last experience. Keep same as rl_agent/q_agent.
    def learn(
        self,
        obs: Any,
        action: Action,
        reward: float,
        next_obs: Any,
        done: bool,
    ) -> None:
        s = self._state_key(obs)
        s2 = self._state_key(next_obs)

        q_s = self._ensure_state(s)
        q_s2 = self._ensure_state(s2)

        a_index = int(action)

        target = float(reward)
        if not done:
            target = float(reward) + self.config.gamma * float(np.max(q_s2))

        q_s[a_index] = q_s[a_index] + self.config.alpha * (target - q_s[a_index])
        return None


    # -------------------------------------------------------------------------
    # Persistence: NPZ (portable) save/load for progressive learning across runs
    #
    # Format: same as QAgent/RLAgent — states (N,2), qvals (N,num_actions), alpha,
    # gamma, epsilon, num_actions, fmt. allow_pickle=False on load for safety.
    # NOTE: Fractal exploration params (c_real, c_imag, z0, kappa, reset_z_each_episode)
    # are NOT saved; only Q-table and alpha/gamma/epsilon are. Loaded agents use
    # default RLFConfig for exploration, so exploitation (Q-table) is unchanged.
    # -------------------------------------------------------------------------

    def to_bytes(self) -> bytes:
        """
        Serialize Q-table + hyperparameters into a compressed .npz blob (bytes).

        Assumes Phase 1 state keys are (row, col) tuples.
        Non-(row,col) keys are ignored (future-proofing if you later add richer states).
        Fractal params (c, z0, kappa, etc.) are not persisted; add them to savez_compressed
        if you need exact exploration behavior across load/save.
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
    def from_bytes(cls, data: bytes, seed: int | None = None) -> "RLFAgent":
        """
        Reconstruct a RLFAgent from bytes produced by to_bytes().
        State keys are restored as (row, col) = (states[i,0], states[i,1]). Fractal
        exploration params (c, z0, kappa, etc.) default to RLFConfig(); only Q + α,γ,ε are restored.
        """
        buf = io.BytesIO(data)
        z = np.load(buf, allow_pickle=False)  # safe: no unpickling of untrusted data

        alpha = float(z["alpha"])
        gamma = float(z["gamma"])
        epsilon = float(z["epsilon"])

        # Only alpha, gamma, epsilon are in NPZ; fractal params use RLFConfig defaults.
        agent = cls(RLFConfig(alpha=alpha, gamma=gamma, epsilon=epsilon), seed=seed)

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
    def load_npz(cls, path: str | Path, seed: int | None = None) -> "RLFAgent":
        """Load Q-table from disk .npz; seed sets RNG. Exploration uses default Julia params."""
        return cls.from_bytes(Path(path).read_bytes(), seed=seed)