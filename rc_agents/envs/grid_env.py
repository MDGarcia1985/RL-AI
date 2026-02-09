"""
grid_env.py

Minimal grid environment for reinforcement learning training.
Configurable 2D grid world for navigation tasks.

Copyright (c) 2026 Michael Garcia, M&E Design
https://mandedesign.studio
michael@mandedesign.studio

CSC370 Spring 2026

Actions (current descrete IDs): 
    0 = FORWARD (up)
    1 = BACKWARD (down)
    2 = RIGHT
    3 = LEFT
"""

# TODO: Unify action definitions with the shared Action enum.
# This environment currently defines its own integer action IDs.
# In larger projects, prefer importing the shared Action enum
# to avoid drift between environment logic, agents, and visualization.

# TODO: Centralize and formalize action semantics.
# Current action meanings:
#     FORWARD  = move up one row
#     BACKWARD = move down one row
#     RIGHT    = move right one column
#     LEFT     = move left one column
#
# These semantics are assumed by:
# - GridEnv.step() transition logic
# - Agent policy interpretation
# - Policy and Q-value visualization
#
# In larger systems, action semantics should be defined in a single
# shared location (e.g., the Action enum) to avoid drift.


from __future__ import annotations

from dataclasses import dataclass # @dataclass is a container for env config/state
from typing import Dict, Tuple # Type hints for dicts/tuples used in RL interfaces.

# Action definitions
ACTION_FORWARD = 0
ACTION_BACKWARD = 1
ACTION_RIGHT = 2
ACTION_LEFT = 3

#valid_actions tells the runner what actions it can accept
VALID_ACTIONS = (
    ACTION_FORWARD,
    ACTION_BACKWARD,
    ACTION_RIGHT,
    ACTION_LEFT,
)

Obs = Tuple[int, int] # observation type (row, col)
StepReturn = Tuple[Obs, float, bool, Dict[str, object]]

@dataclass # dataclass automatically generates boilerplate like __init__, __repr__, etc
class GridConfig:
    rows: int = 5 #simple container for grid settings
    cols: int = 5 #default 5x5 grid
    start: Obs = (0, 0) #initial (row, col) position (default top-left corner)
    goal: Obs = (4, 4) #Default target

class GridEnv: #defines env object
    def __init__(self, config: GridConfig | None = None): # sets up the env, config: GridConfig lets you pass a config or use defaults
        self.config = config if config is not None else GridConfig() # stores config.
        self.pos: Obs = self.config.start # stores current position. Initialize to start.
    
    def reset(self) -> Obs: # returns to start position
        self.pos = self.config.start
        return self.pos

    def step(self, action: int) -> StepReturn:
        """Apply an action to the environment and return the result."""
        if action not in VALID_ACTIONS:
            raise ValueError(f"Invalid action {action}")
        
        # unpack current position
        row, col = self.pos

        # Compute new position based on the selected action.
        #
        # NOTE on action semantics (current implementation):
        # - Grid coordinates are (row, col) with row 0 at the top.
        # - FORWARD/BACKWARD move along the row axis.
        # - LEFT/RIGHT move along the column axis.
        #
        # INVARIANT:
        # - This mapping defines the environment's transition model.
        # - Any change here alters the meaning of actions system-wide
        #   (learning, policy evaluation, and visualization).
        #
        # TODO (see action semantics TODO above):
        # - Action meanings are currently duplicated here as implicit logic.
        # - In future refactors, action semantics should be centralized
        #   (e.g., in the Action enum) and consumed here.
        #
        # NOTE on future refactor (diagonal / half-step actions):
        # - If diagonal actions are added (e.g., NW, NE, SW, SE), replace this
        #   conditional chain with an action → (d_row, d_col) mapping.
        # - Example refactor pattern:
        #       delta = ACTION_DELTAS[action]
        #       row += delta[0]
        #       col += delta[1]

        if action == ACTION_FORWARD:
            row -= 1
        elif action == ACTION_BACKWARD:
            row += 1
        elif action == ACTION_RIGHT:
            col += 1
        elif action == ACTION_LEFT:
            col -= 1

        # clip new position to grid boundaries using built-in min/max
        row = max(0, min(row, self.config.rows - 1))
        col = max(0, min(col, self.config.cols - 1))

        # update position
        self.pos = (row, col)

        # compute reward
        reward = -1.0 # default reward is -1 for each move
        # The negative starting reward creates a penalty for taking longer paths.
        # Each step costs -1.0, so unnecessary movement is discouraged.
        #
        # Reaching the goal gives a reward of 0.0.
        # This does not add points — it simply stops the penalty.
        #
        # As a result, the agent learns to minimize total negative reward,
        # which is equivalent to finding the shortest valid path.
        #
        # Example:
        # Start (0,0) -> (1,0) -> (2,0) -> (3,0) -> Goal
        # rewards: -1 + -1 + -1 + -1 + 0 = -4.0 total
        #
        # NOTE: Goal reward is 0.0 (not +N) to keep the objective "minimize steps"
        # rather than "seek high positive reward." Both work; this keeps totals intuitive.


        done = False # default state is not done

        # check for goal
        if self.pos == self.config.goal:
            reward = 0.0 # reward is 0 if you reach the goal
            done = True # done is true if you reach the goal

        info: Dict[str, object] = {"pos": self.pos} # info is a dictionary for additional info (empty for now)
        return self.pos, reward, done, info
