python -m streamlit run .\rc_agents\ui\app_streamlit.py


This interface is documented rather than enforced to keep the code
lightweight and readable for this assignment. In a larger system—such
as the planned RC UGV control project—this contract will be formalized
using a shared `types.py` module or Python `Protocol`s once multiple
runners, agents, or environments introduce pressure for stronger static
guarantees.

## Runner Interface Contracts

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
