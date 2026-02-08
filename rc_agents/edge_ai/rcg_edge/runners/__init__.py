"""
runners init file

used for package import paths

Copyright (c) 2026 Michael Garcia, M&E Design
https://mandedesign.studio
michael@mandedesign.studio

CSC370 Spring 2026
"""
# Create 'from' path for dependencies
from .train_runner import Env, EpisodeResult, run_training

__all__ = [
    "Env",
    "EpisodeResult",
    "run_training",
]

