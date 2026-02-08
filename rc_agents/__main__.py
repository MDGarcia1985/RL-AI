# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy>=2.4.2",
# ]
# ///
"""
__main__.py

RC Agents Package Entry Point
Main execution script for the reinforcement learning package.

Copyright (c) 2026 Michael Garcia, M&E Design
https://mandedesign.studio
michael@mandedesign.studio

CSC370 Spring 2026
"""

from __future__ import annotations

from rc_agents.envs.grid_env import GridEnv, GridConfig
from rc_agents.edge_ai.rcg_edge.agents.q_agent import QAgent, QConfig
from rc_agents.edge_ai.rcg_edge.runners.train_runner import run_training
from rc_agents.utils.logger import log_execution


def main() -> None:
    log_execution("MAIN_RUN", "Training started")

    print("RC Sentry ACTIVATED. Patrol initiated.")
    
    env = GridEnv(GridConfig(rows=20, cols=20, start=(0, 0), goal=(4, 4)))
    agent = QAgent(QConfig(alpha=0.5, gamma=0.9, epsilon=0.1), seed=123)

    from rc_agents.config.ui_config import TrainingUIConfig

    cfg = TrainingUIConfig(
        episodes=50,
        max_steps=200,
        rows=20,
        cols=20,
        start=(0, 0),
        goal=(4, 4),
        alpha=0.5,
        gamma=0.9,
        epsilon=0.1,
        seed=123,
    )

    env = GridEnv(cfg.to_grid_config())
    agent = QAgent(cfg.to_q_config(), seed=cfg.seed)

    results = run_training(env=env, agent=agent, cfg=cfg)

    wins = sum(1 for r in results if r.reached_goal)
    print(f"Reached goal: {wins}/{len(results)}")
    
    log_execution("MAIN_COMPLETE", f"Training finished: {wins}/{len(results)} wins")


if __name__ == "__main__":
    main()