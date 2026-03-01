# rc_agents

This package contains the reinforcement learning agents, environments, runners, and Streamlit UI for the NNABL_POC project. It is designed so that the same agents and runner work from the CLI, Tkinter GUI, or Streamlit—with no training logic inside the UI.

**Decision intent:** Keep a single training loop (`run_training` in `edge_ai/rcg_edge/runners/train_runner.py`), swappable environments (e.g. `GridEnv`, `MazeEnv`), and a catalog of agents (e.g. RL, RLF) built by the UI factory from `config`. Progressive learning (reuse and Q-table transfer when grid size changes) is handled in the Streamlit layer via `progressive_learning` and session state.

## Package layout (mental model)

- **`envs/`** — Grid and maze environments; `reset()` and `step(action)` follow the runner contract.
- **`edge_ai/rcg_edge/agents/`** — Agents (Random, Q, RL, RLF, etc.); each implements `reset`, `act(obs)`, `learn(...)`.
- **`edge_ai/rcg_edge/runners/`** — `run_training(env, agent, cfg)` and maze helpers; returns `(results, best_trajectory)`.
- **`config/`** — `TrainingUIConfig` and converters (`to_grid_config()`, `to_q_config()`); single source of truth for the UI.
- **`ui/streamlit_ui/`** — Sidebar (env dropdown, agent checkboxes, hyperparameters), main panel (per-agent panels + best-run trail), factory (`make_agent`, `make_env`, `get_env_options`), and progressive learning (agent reuse and Q-table transfer).
- **`testers/`** — Pytest tests; run with `python -m pytest rc_agents/testers -q` from the project root.

For how to run the app and tests, see the top-level **README.md** in the project root.

---

## Action deltas (future expansion)

When movement is centralized (e.g. for 8-direction or hardware mapping), the intended pattern is:

```python
ACTION_DELTAS = {
    ACTION_FORWARD: (-1, 0),
    ACTION_BACKWARD: (1, 0),
    ACTION_LEFT: (0, -1),
    ACTION_RIGHT: (0, 1),
    ACTION_NORTHWEST: (-1, -1),
    ACTION_NORTHEAST: (-1, 1),
    ACTION_SOUTHWEST: (1, -1),
    ACTION_SOUTHEAST: (1, 1),
}

d_row, d_col = ACTION_DELTAS[action]
row += d_row
col += d_col
```

## Coding conventions

This project follows standard Python naming conventions to clarify semantic roles:

- **CamelCase** for classes and type aliases
- **snake_case** for functions, methods, and variables
- **ALL_CAPS** for constants and enum-like values

These conventions are used consistently across environments, agents, runners,
and UI code to improve readability and reduce ambiguity.

These conventions are documented explicitly to support readability for students and contributors who may be new to larger Python codebases.

- Pytest: <https://docs.pytest.org/en/stable/>