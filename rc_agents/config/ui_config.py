"""
ui_config.py

UI configuration containers for rc_agents.

Goal:
- Keep all “knobs” (episodes, alpha, epsilon, grid size, etc.) in one place
- Allow both Tkinter and Streamlit to share the same defaults
- Make it easy to expand later (maze, labyrinth, no-go zones)

Copyright (c) 2026 Michael Garcia, M&E Design
https://mandedesign.studio
michael@mandedesign.studio

CSC370 Spring 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class TrainingUIConfig:
    """
    TrainingUIConfig is the single source of truth for training parameters.

    This dataclass is populated by the UI layer (Tkinter or Streamlit),
    then passed into the environment, agent, and runner without duplication.
    """

    # ------------------------------------------------------------
    # Training loop controls
    # ------------------------------------------------------------
    episodes: int = 50        # Number of complete episodes to run.
    max_steps: int = 200      # Maximum steps per episode.

    # ------------------------------------------------------------
    # Q-learning hyperparameters
    # ------------------------------------------------------------
    alpha: float = 0.50       # Learning rate (α)
    gamma: float = 0.90       # Discount factor (γ)
    epsilon: float = 0.10     # Exploration rate (ε)

    # ------------------------------------------------------------
    # Environment layout
    # ------------------------------------------------------------
    rows: int = 5
    cols: int = 5

    # ------------------------------------------------------------
    # Environment landmarks
    # ------------------------------------------------------------
    start: Tuple[int, int] = (0, 0)
    goal:  Tuple[int, int] = (4, 4)

    # ------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------
    seed: int | None = 123

    # Optional future: unify UI state instead of storing game_type in session_state
    # game_type: str = "Open World"

    # ------------------------------------------------------------
    # Converters
    # ------------------------------------------------------------

    def to_grid_config(self) -> "GridConfig":
        """
        Convert UI settings into a GridConfig object.

        Keeping this here prevents the UI layer from duplicating
        environment setup logic.
        """
        from ..envs import GridConfig

        return GridConfig(
            rows=self.rows,
            cols=self.cols,
            start=self.start,
            goal=self.goal,
        )

    # def to_maze_config(self) -> "MazeConfig":
    #    """
    #    Convert UI settings into a MazeConfig object.
    #
    #       This is to generate different kinds of mazes from one source
    #       of truth.
    #    """
    #    from ..envs import MazeConfig
    #
    #    return MazeConfig(
    #        rows=self.rows,
    #        cols=self.cols,
    #        start=self.start,
    #        goal=self.goal,
    #        # walls gets filled by maze_runner (ascii/procedural/etc.)
    #    )

    def to_q_config(self) -> "QConfig":
        """
        Convert UI settings into a QConfig object.

        Keeping this here prevents the UI layer from duplicating
        agent setup logic.

        All agents share the same seed so as to be comparable in their performance.
        """
        from ..edge_ai.rcg_edge.agents import QConfig

        return QConfig(
            alpha=self.alpha,
            gamma=self.gamma,
            epsilon=self.epsilon,
        )