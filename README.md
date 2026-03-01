## How to run

**Streamlit UI (recommended)** — training, multi-agent panels, and best-run trail:

```text
python -m streamlit run rc_agents/ui/app_streamlit.py
```

- **Sidebar:** Choose environment (Open World or Maze) from the dropdown; check one or more agents (e.g. RL Base Agent, RL with Fractal Exploration). Set episodes, max steps, and hyperparameters (alpha, gamma, epsilon). Use “Reset Agents” to clear cached Q-tables; use “Save / Load” to download or upload a learned Q-table (.npz) after training.
- **Main panel:** Click “Run Training.” Each selected agent gets its own expandable section (summary, Q-table, value heatmap, policy). At the bottom, “Best run (trail)” shows the path of the best successful run (fewest steps to goal). Progressive learning is enabled: re-running with the same config reuses the same agent; changing grid size transfers the Q-table into the new grid.

**CLI (headless)** — quick run with default grid and QAgent:

```text
python -m rc_agents
```

**Tests:**

```text
python -m pytest rc_agents/testers -q
```

---

## Runner interface contracts

This interface is documented rather than enforced to keep the code
lightweight and readable for this assignment. In a larger system—such
as the planned RC UGV control project—this contract will be formalized
using a shared `types.py` module or Python `Protocol`s once multiple
runners, agents, or environments introduce pressure for stronger static
guarantees.

The training loop (`train_runner.py`) is intentionally decoupled from
specific environment and agent implementations.

Rather than requiring concrete base classes, the runner relies on
minimal **behavioral contracts**. These contracts are documented here
to clarify system boundaries and support future extensions, without
introducing unnecessary abstraction for this assignment.

### Environment Contract

An environment used by the training runner must provide:

- `reset() -> obs`
- `step(action: int) -> (obs, reward, done, info)`

This mirrors the classic Gym-style reinforcement learning interface:
`(observation, reward, done, info)`.

### Agent Contract (AgentLike)

An agent used by the training runner is expected to implement the
following interface (illustrative, not enforced):

```python
class AgentLike:
    def reset(self) -> None:
        ...

    def act(self, obs) -> StepResult:
        ...

    def learn(
        self,
        obs,
        action,
        reward: float,
        next_obs,
        done: bool,
    ) -> None:
        ...
```

**Return value.** The runner returns `(results, best_trajectory)`: a list of `EpisodeResult` and, when at least one episode reached the goal, the path (list of `(row, col)`) for the best run (fewest steps). The UI uses `best_trajectory` for the “Best run (trail)” graph.
