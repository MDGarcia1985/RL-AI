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
    
    env = GridEnv(GridConfig(rows=5, cols=5, start=(0, 0), goal=(4, 4)))
    agent = QAgent(QConfig(alpha=0.5, gamma=0.9, epsilon=0.1), seed=123)

    results = run_training(env=env, agent=agent, episodes=50, max_steps=200)

    wins = sum(1 for r in results if r.reached_goal)
    print(f"Reached goal: {wins}/{len(results)}")
    
    log_execution("MAIN_COMPLETE", f"Training finished: {wins}/{len(results)} wins")


if __name__ == "__main__":
    main()