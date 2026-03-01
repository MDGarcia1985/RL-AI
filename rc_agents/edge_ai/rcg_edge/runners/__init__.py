"""
runners package

Training loop orchestration for rc_agents.

Design intent:
- Keep a single canonical training loop (run_training).
- Allow environment-specific helpers (e.g., maze runner).
- Provide a convenient entry point for MazeEnv experiments.


Copyright (c) 2026 Michael Garcia, M&E Design
https://mandedesign.studio
michael@mandedesign.studio

CSC370 Spring 2026
"""
# Package imports
from .train_runner import Env, EpisodeResult, run_training
from .maze_runner import (
    run_maze_training_from_ascii,
    run_maze_training_generated_dfs,
    generate_dfs_maze_walls,
)

__all__ = [
    "Env",
    "EpisodeResult",
    "run_training",
    "run_maze_training_from_ascii",
    "run_maze_training_generated_dfs",
    "generate_dfs_maze_walls",
]