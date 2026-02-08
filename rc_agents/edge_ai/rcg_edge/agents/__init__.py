"""
agents/__init__.py

Agent Registry

used for package import paths
and agent registration with the runner

Copyright (c) 2026 Michael Garcia, M&E Design
https://mandedesign.studio
michael@mandedesign.studio

CSC370 Spring 2026
"""
# Create 'from' path for dependencies
from .base import Action, StepResult, Agent
from .q_agent import QConfig, QAgent
from .random_agent import RandomAgent

__all__ = [
    "Action",
    "StepResult",
    "Agent",
    "QConfig",
    "QAgent",
    "RandomAgent",
]
