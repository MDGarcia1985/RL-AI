RC Agents – Development Notes & Architecture Reference

Author: Michael Garcia
CSC370 Spring 2026
M&E Design
https://mandedesign.studio

Purpose of This File

This document exists to capture design intent, architectural patterns, and “why” decisions that are easy to forget over time.

It is not user documentation.
It is not grading documentation.
It is a reference for future development across:

rc_guardian

propane hybrid UGV

lawn care / snow removal platforms

simulation + real hardware parity

If something looks “obvious” today, it won’t be in six months. Write it down.

Package Philosophy

This project is designed as a long-lived, modular robotics codebase, not a single assignment.

Key principles:

Clear separation of environment, agent, runner, and UI

No monolithic scripts

Same agents must work in:

CLI

Tkinter GUI

Streamlit UI

Headless / embedded execution

Environments should be swappable (grid → yard → real sensors)

Relative Imports (Critical Reference)

This project uses explicit relative imports to keep modules portable and refactor-safe.

Mental Model

Relative imports move through the package tree, not the filesystem.

rc_agents/
├─ envs/
├─ edge_ai/
│  └─ rcg_edge/
│     ├─ agents/
│     └─ runners/
├─ utils/
├─ config/

Relative Import Mapping
Relative Import	Resolves To
.	current package
..	parent package
..envs	rc_agents.envs
..utils	rc_agents.utils
..config	rc_agents.config
..edge_ai	rc_agents.edge_ai
..edge_ai.rcg_edge.agents	rc_agents.edge_ai.rcg_edge.agents
Examples
Importing the Grid Environment
from ..envs import GridEnv, GridConfig


Equivalent absolute import:

from rc_agents.envs import GridEnv, GridConfig

Importing the Q-Agent
from ..edge_ai.rcg_edge.agents import QAgent, QConfig

Rule of Thumb

If the file lives inside rc_agents/, use relative imports

Only __main__.py or Streamlit entry points should use absolute imports

If you need more than two .., reconsider the module location

Why dataclass Is Used Everywhere

Dataclasses are used intentionally for:

Training configuration

UI configuration

Environment configuration

Step / result containers

Reasons:

Self-documenting parameters

Safe defaults

Easy UI binding (Tkinter, Streamlit)

Serializable later (JSON / YAML)

Cleaner diffs when values change

This is preferred over long positional argument lists.

Agent Architecture
Agent (base contract)

Defines the minimum contract for all agents:

reset()

act(obs)

learn(obs, action, reward, next_obs, done)

Agents do not control the loop.
They only react to observations and feedback.

This allows:

Random agents

Q-learning agents

Future planners

Hardware-driven agents

All interchangeable.

RandomAgent

Purpose:

Baseline behavior

Sanity checking

Debugging environments

It does not learn.
It defines what is possible, not what is optimal.

QAgent

Purpose:

First learning agent

Memory-based decision making

Baseline for all future RL extensions

Key traits:

Uses epsilon-greedy policy

Stores values in a Q-table

Learns only from non-terminal states

Terminal vs Non-Terminal States

Terminal state:

Episode is over

No future reward possible

Learning target = immediate reward only

Non-terminal state:

Episode continues

Future rewards still possible

Learning target includes discounted future value

This distinction prevents the agent from hallucinating future rewards after an episode ends.

Environment Design (GridEnv)

The environment is intentionally minimal:

Deterministic movement

Bounded grid

Explicit reward shaping

Reward Model
reward = -1.0  # every step costs something


Design intent:

Penalize long paths

Encourage efficiency

Reaching goal stops the penalty (reward = 0)

The agent learns to minimize total negative reward, not chase positive reinforcement.

This mirrors real constraints:

Energy usage

Time cost

Wear on hardware

Tests Philosophy

Tests are behavioral, not cosmetic.

They answer:

Did the agent choose the correct action?

Did Q-values update correctly?

Did terminal logic behave as expected?

Tests are intentionally small and focused.

Passing tests = system integrity
Failing tests = design signal, not noise

UI Strategy

Tkinter = assignment compliance + local debugging

Streamlit = visualization + parameter tuning

UI does not contain logic

UI populates config → runner executes

No training logic should ever live in UI files.

Logging & Traceability

Execution logging exists for:

Debugging

Replay

Accountability (did this actually run?)

This will later support:

Field logs

Safety audits

Failure analysis

Final Note (Intentional)

This codebase is written for humans first.

Comments:

Explain intent

Explain tradeoffs

Explain “why”, not just “what”

Minor typos in comments are acceptable.
Opaque code is not.