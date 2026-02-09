"""
Environment for rc_agents

__init__.py file creates a package for this folder as well as
package import paths.

Copyright (c) 2026 Michael Garcia, M&E Design
https://mandedesign.studio
michael@mandedesign.studio

CSC370 Spring 2026
"""

# Create 'from' path for dependencies
from .grid_env import GridEnv, GridConfig
__all__ = ["GridEnv", "GridConfig"]


# Future note: 
# Additional Environments can be added here.
# Example:
#         from .custom_env import CustomEnv
