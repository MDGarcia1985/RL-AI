"""
Grid Environment Tester

Phase 1: Basic Movement Control
    By Michael Garcia
    CSC370 Spring 2026
    michael@mandedesign.studio
"""

from rc_agents.envs.grid_env import GridEnv, GridConfig, ACTION_FORWARD
from rc_agents.utils.logger import log_execution

#Function to test the reset function
def test_reset_return_start():
    log_execution("TEST_RUN", "test_reset_return_start")
    config = GridConfig(start=(2, 2), goal=(4, 4))
    env = GridEnv(config)
    obs = env.reset()
    assert obs == (2, 2)

#Function to test the step function
def test_forward_at_top_wall_does_not_move():
    log_execution("TEST_RUN", "test_forward_at_top_wall_does_not_move")
    env = GridEnv(GridConfig(rows=5, cols=5, start=(0, 2)))
    env.reset()
    obs, reward, done, info = env.step(ACTION_FORWARD)
    assert obs == (0, 2)  # Should not have moved
    assert done is False