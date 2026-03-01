# UI layer

The UI does not run the training loop. It collects settings from the user, builds config and (via the factory) environments and agents, then calls the shared runner. This keeps one source of truth for training logic and makes the same agents usable from CLI, Tk, or Streamlit.

- **Streamlit** — main interface for training, multi-agent comparison, and visualization (recommended).
- **Tkinter** (`gui_main.py`) — optional local GUI; same config → runner pattern.

Below is a guide to the **Streamlit** app: how to run it, what you see, and why it’s structured that way (for users at any level and for future maintainers).

---

## How to run the Streamlit app

From the **project root** (the folder that contains `rc_agents/`):

```text
python -m streamlit run rc_agents/ui/app_streamlit.py
```

Your browser will open to the app. If you are new to the project, run this once and use the UI while reading the next section.

---

## What you see: sidebar and main panel

**Sidebar (left)**

- **Environment** — Dropdown: *Open World* (plain grid) or *Maze* (DFS-generated walls). This choice is passed to the factory so the same “Run Training” flow works for both env types.
- **Agents** — One checkbox per agent in the catalog (e.g. *RL Base Agent*, *RL with Fractal Exploration*). Check one or more; only checked agents are trained and get a results section. Decision intent: as you add agents to the factory catalog, they automatically get a checkbox here.
- **Hyperparameters** — Episodes, max steps, alpha, gamma, epsilon, grid rows/cols, start and goal. Stored in a single `TrainingUIConfig` and converted to env/agent config by the factory and config layer.
- **Reset Agents** — Clears cached agents so the next “Run Training” starts from scratch. Use this when you want to abandon progressive learning for the current session.
- **Save / Load** — After at least one run with a Q-learning–style agent, *Download Q-table (.npz)* appears. Use it to save the learned Q-table. *Load Q-table* restores from a file so you can continue or inspect without retraining. Only agents that support `to_bytes`/`from_bytes` participate.

**Main panel (center)**

- **Run Training** — Builds the environment from the sidebar env choice and, for each checked agent, either reuses a cached agent, transfers its Q-table (if only grid size changed), or creates a new one. Then runs the shared `run_training(env, agent, cfg)` and shows results.
- **Per-agent sections** — Each selected agent gets an expandable block: win count, last episodes, Q-table table, state-value heatmap, and policy arrows. So you can compare agents side by side.
- **Best run (trail)** — At the bottom, a single path plot for the best successful run (fewest steps to goal) across all agents. For mazes, the path is drawn over the wall layout. Decision intent: one shared “best run” so you don’t have to hunt for the best trajectory per agent.

---

## Why it’s structured this way (decision intent)

- **Single training loop** — All training happens in `run_training` (in `edge_ai/rcg_edge/runners/`). The UI never implements learning; it only builds env and agent and calls the runner. That keeps behavior consistent between CLI, Tk, and Streamlit and makes bugs easier to trace.
- **Factory** — `streamlit_ui/factory.py` provides `make_env(cfg, game_type)` and `make_agent(agent_id, cfg)`. The UI stays thin and doesn’t need to know constructor details. New envs or agents are added in the factory (and catalog) and appear in the dropdown/checkboxes without rewriting the rest of the UI.
- **Progressive learning** — Streamlit reruns the script on every interaction. To avoid losing learned Q-tables, we cache agents in session state (per agent id, with a “context key” of alpha, gamma, epsilon, seed). If you run again with the same config and grid, the same agent is reused and keeps learning. If you only change grid size, we copy overlapping Q-values into a new agent so you don’t lose everything. See `streamlit_ui/progressive_learning.py`.
- **Best-run trajectory** — The runner returns `(results, best_trajectory)`. The UI uses `best_trajectory` for the trail plot so you get one clear “best path” without implementing path logic in the UI.

---

## Where the code lives (Streamlit)

| What you see / do        | Module / file |
|--------------------------|----------------|
| App entry, routing       | `app_streamlit.py` |
| Sidebar layout, env + agents + hyperparams | `streamlit_ui/sidebar_ui.py` |
| Main panel, Run Training, per-agent blocks, trail | `streamlit_ui/main_panel.py` |
| Building envs and agents | `streamlit_ui/factory.py` |
| List of agents (for checkboxes) | `streamlit_ui/agent_catalog.py` |
| Caching and Q-table transfer | `streamlit_ui/progressive_learning.py` |
| Q-table → tables and heatmaps | `viz/q_table_viz.py` |
| Best-run path plot      | `viz/trail_viz.py` |

Helpers (coordinates, numeric inputs, etc.) live in `streamlit_ui/` as well; the table above covers the main flow.

---

## If something doesn’t work

1. **“Run Training” does nothing** — Make sure at least one agent is checked in the sidebar.
2. **Download Q-table never appears** — Run training at least once with an agent that supports Q-table save (e.g. RL Base Agent or RLF). The button appears after a run completes.
3. **Wrong or stale results** — Try “Reset Agents” in the sidebar, then run again. If you changed grid size and want to keep learning, the transfer logic should run automatically; if you changed alpha/gamma/epsilon/seed, a new agent is created on purpose.

For deeper architecture and design notes, see the project root `README.md`, `rc_agents/README.md`, and `docs/DEV_NOTES.md`.
