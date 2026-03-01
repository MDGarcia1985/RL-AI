NNABL_POC/
├─ README.md
├─ pyproject.toml
├─ pytest.ini
├─ rc_agents/
│  ├─ README.md
│  ├─ __main__.py              # CLI: python -m rc_agents
│  ├─ config/
│  │  └─ ui_config.py          # TrainingUIConfig, to_grid_config(), to_q_config()
│  ├─ envs/
│  │  ├─ grid_env.py           # GridEnv, GridConfig
│  │  └─ maze_env.py           # MazeEnv, MazeConfig
│  ├─ edge_ai/
│  │  └─ rcg_edge/
│  │     ├─ agents/
│  │     │  ├─ base.py         # Action, StepResult, Agent contract
│  │     │  ├─ random_agent.py
│  │     │  ├─ q_agent.py
│  │     │  ├─ rl_agent.py
│  │     │  └─ rlf_agent.py
│  │     └─ runners/
│  │        ├─ train_runner.py # run_training() -> (results, best_trajectory)
│  │        ├─ maze_runner.py
│  │        └─ convergence_tracker.py
│  ├─ ui/
│  │  ├─ app_streamlit.py      # Streamlit entry
│  │  ├─ streamlit_ui/         # sidebar, main_panel, factory, agent_catalog, progressive_learning
│  │  └─ viz/                  # q_table_viz, trail_viz (best-run path)
│  ├─ testers/
│  │  ├─ test_trainer.py
│  │  ├─ test_factory.py
│  │  ├─ test_agent_catalog.py
│  │  └─ ...
│  └─ data/
└─ docs/
   └─ ...