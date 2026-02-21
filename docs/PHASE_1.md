rcg_agents_assignment/
├─ README.md
├─ pyproject.toml
├─ edge_ai/
│  └─ rcg_edge/
│     ├─ __init__.py
│     ├─ agents/
│     │  ├─ __init__.py
│     │  ├─ base.py
│     │  ├─ random_agent.py
│     │  └─ q_agent.py
│     └─ runners/
│        └─ train_runner.py
├─ envs/
│  └─ grid_env.py
├─ tests/
│  ├─ test_q_update.py
│  └─ test_action_selection.py
└─ data/
   └─ botmovements.txt   # optional if you reuse logs