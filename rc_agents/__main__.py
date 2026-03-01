# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy>=2.4.2",
# ]
# ///
# NOTE: PEP 723 script metadata (used by uv) for standalone execution.

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

import logging

# Configure logging only if not already configured
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,  # or DEBUG
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

from rc_agents.envs import GridEnv
from rc_agents.edge_ai.rcg_edge.agents import QAgent
from rc_agents.edge_ai.rcg_edge.runners import run_training
from rc_agents.utils.logger import log_execution
from rc_agents.config import TrainingUIConfig

# Main execution function - minimal router for directing traffic.
def main() -> None:
    log_execution("MAIN_RUN", "Training started")
    print("RC Sentry ACTIVATED. Patrol initiated.")

    # NOTE: goal is intentionally close to start for fast smoke-test runs.
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

    # Use cfg as single source of truth.
    env = GridEnv(cfg.to_grid_config())
    agent = QAgent(cfg.to_q_config(), seed=cfg.seed)

    # Run the shared training loop (package core logic).
    # run_training returns (results, best_trajectory); CLI only uses results
    results, _ = run_training(env=env, agent=agent, cfg=cfg)
    wins = sum(1 for r in results if r.reached_goal)
    
    print(f"Reached goal: {wins}/{len(results)}")
    log_execution("MAIN_COMPLETE", f"Training finished: {wins}/{len(results)} wins")


if __name__ == "__main__":
    main()