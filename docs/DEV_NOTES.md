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

------------------------------------------------
Exploring math-based algorithmic learning
------------------------------------------------

To explore different trainers with quantifiable results I created:

- Agent_RL = Q-learning + classic ε(epsilon)-greedy random explore

- Agent_RLF = Q-learning + fractal-driven exploration (your Julia/θ(theta) scout)

- Agent_GA = genetic algorithm policy search (different learning loop, but can still be “an agent”)

- Agent_MANDO = Mandelbrot-parameterized explorer (or meta-controller that mutates c)

I used a tournament runner that basically:

- creates env per agent run

- reuses same cfg + seeds

- stores (agent_name → results + artifacts)

Pseudo-structure:

    AGENTS = {
        "RL": lambda: QAgent(...classic...),
        "RLF": lambda: QAgent(...fractal explore...),
        "GA": lambda: GAAgent(...),
        "MANDO": lambda: MandoAgent(...),
    }

    scoreboard = []
    details = {}

    for name, make_agent in AGENTS.items():
        agent = make_agent()
        env = GridEnv(cfg.to_grid_config())
        results = run_training(env, agent, cfg)

        wins = sum(r.reached_goal for r in results)
        avg_steps = sum(r.steps for r in results) / len(results)
        scoreboard.append({...})
        details[name] = {"agent": agent, "results": results}

        Fractal Exploration (RLF Agent)
Motivation

Classic ε(epsilon)-greedy exploration samples uniformly at random.

That works, but:

It has no structure

It has no memory

It does not scale well as state space grows

RLF introduces deterministic-chaotic exploration via Julia dynamics:

𝑧
𝑛
+
1
=
𝑧
𝑛
2
+
𝑐
z
n+1
	​

=z
n
2
	​

+c

The complex state is mapped to:

𝜃
(
𝑡
ℎ
𝑒
𝑡
𝑎
)
=
𝑎
𝑡
𝑎
𝑛
2
(
𝐼
𝑚
(
𝑧
)
,
𝑅
𝑒
(
𝑧
)
)
θ(theta)=atan2(Im(z),Re(z))

Which is normalized to:

𝜃
∈
[
0
,
2
𝜋
)
θ∈[0,2π)

This continuous heading is then softly projected onto discrete actions.

Why This Matters

Uniform random exploration treats space as flat.

Fractal exploration:

Produces structured coverage

Creates non-repeating exploration sequences

Avoids purely memoryless action sampling

Introduces tunable chaos via parameter 
𝑐
c

This creates a middle ground between:

Random noise

Fully deterministic planners

Action Mapping Strategy

Current environment supports 4 actions:

FORWARD

BACKWARD

LEFT

RIGHT

Fractal exploration generates a continuous heading.

Mapping strategy:

Compute angular distance to each action heading

Convert distance to weight via:

𝑤
=
𝑒
𝑥
𝑝
(
−
𝜅
(
𝑘
𝑎
𝑝
𝑝
𝑎
)
∗
𝑑
2
)
w=exp(−κ(kappa)∗d
2
)

Normalize:

𝑝
=
𝑤
/
∑
𝑤
p=w/∑w

Sample action from p

This preserves compatibility with GridEnv while enabling future expansion.

Movement Stub (Future Expansion)

The _ACTION_DELTAS structure exists to support:

8-direction movement

Continuous headings

Hardware-aligned motion (wheel velocity mapping)

Movement logic will eventually migrate from GridEnv into a more general motion model.

Current stub is intentionally isolated.

Tournament Framework (Comparative Evaluation)

Agents should not be evaluated emotionally.

They should be evaluated comparatively.

The tournament pattern:

Same environment

Same seed

Same config

Independent agent instances

Metrics:

Win rate

Average steps to goal

Convergence slope

Q-table entropy

Exploration diversity

This enables:

RL vs RLF comparison

RLF vs GA

MANDO vs all baselines

Comparisons must be apples-to-apples.

Logging Architecture (Critical for Long-Term Growth)

Logging is centralized at application entry points.

Modules define loggers:

logger = logging.getLogger(__name__)

Only entry points configure logging:

if not logging.getLogger().hasHandlers():
    logging.basicConfig(...)

Why:

Prevent duplicate handlers

Maintain framework compatibility (Streamlit, CLI)

Preserve clean separation of concerns

Agents never configure logging.

This ensures portability to:

Embedded systems

Headless deployments

Field logging

Hardware safety audits

Mandelbrot-Based Meta Exploration (MANDO Concept)

Future direction:

Instead of fixing parameter 
𝑐
c, allow it to evolve.

Two approaches:

Deterministic schedule across Mandelbrot parameter space

Error-correcting mutation (Reed-Solomon-inspired constraints)

Fitness-weighted parameter mutation

Conceptually:

RLF = Julia dynamics at fixed c

MANDO = exploration across Mandelbrot space of c

This becomes meta-learning over exploration patterns.

Hardware Parity Considerations

GridEnv is a controlled abstraction.

Real hardware introduces:

Sensor noise

Actuator lag

Energy constraints

Safety bounds

Exploration strategies must remain:

Bounded

Recoverable

Interruptible

Fractal exploration is acceptable only if:

Action deltas remain bounded

Fail-safe termination exists

Logging captures trajectory

This is non-negotiable in UGV deployments.

Design Guardrails

Do not:

Put learning logic in UI

Put logging configuration in agents

Allow agents to control training loop

Couple environment physics to agent internals

Always:

Keep agent stateless between episodes except Q-table

Keep exploration swappable

Keep evaluation measurable

Experimental Discipline

Before adding new math:

Define measurable hypothesis

Define control agent

Define metric

Run multiple seeds

Log results

Fractal ideas are not the goal.

Measurable improvement is the goal.

Long-Term Direction

This project is converging toward:

Unified simulation + hardware stack

Swappable exploration strategies

Structured experiment harness

Reproducible learning experiments

Fractal / GA / classical RL coexistence

This is not an assignment.

This is an extensible robotics learning framework.

Fractal Exploration (RLF Agent)
Motivation

Classic ε(epsilon)-greedy exploration samples uniformly at random.

That works, but:

It has no structure

It has no memory

It does not scale well as state space grows

RLF introduces deterministic-chaotic exploration via Julia dynamics:

𝑧
𝑛
+
1
=
𝑧
𝑛
2
+
𝑐
z
n+1
	​

=z
n
2
	​

+c

The complex state is mapped to:

𝜃
(
𝑡
ℎ
𝑒
𝑡
𝑎
)
=
𝑎
𝑡
𝑎
𝑛
2
(
𝐼
𝑚
(
𝑧
)
,
𝑅
𝑒
(
𝑧
)
)
θ(theta)=atan2(Im(z),Re(z))

Which is normalized to:

𝜃
∈
[
0
,
2
𝜋
)
θ∈[0,2π)

This continuous heading is then softly projected onto discrete actions.

Why This Matters

Uniform random exploration treats space as flat.

Fractal exploration:

Produces structured coverage

Creates non-repeating exploration sequences

Avoids purely memoryless action sampling

Introduces tunable chaos via parameter 
𝑐
c

This creates a middle ground between:

Random noise

Fully deterministic planners

Action Mapping Strategy

Current environment supports 4 actions:

FORWARD

BACKWARD

LEFT

RIGHT

Fractal exploration generates a continuous heading.

Mapping strategy:

Compute angular distance to each action heading

Convert distance to weight via:

𝑤
=
𝑒
𝑥
𝑝
(
−
𝜅
(
𝑘
𝑎
𝑝
𝑝
𝑎
)
∗
𝑑
2
)
w=exp(−κ(kappa)∗d
2
)

Normalize:

𝑝
=
𝑤
/
∑
𝑤
p=w/∑w

Sample action from p

This preserves compatibility with GridEnv while enabling future expansion.

Movement Stub (Future Expansion)

The _ACTION_DELTAS structure exists to support:

8-direction movement

Continuous headings

Hardware-aligned motion (wheel velocity mapping)

Movement logic will eventually migrate from GridEnv into a more general motion model.

Current stub is intentionally isolated.

Tournament Framework (Comparative Evaluation)

Agents should not be evaluated emotionally.

They should be evaluated comparatively.

The tournament pattern:

Same environment

Same seed

Same config

Independent agent instances

Metrics:

Win rate

Average steps to goal

Convergence slope

Q-table entropy

Exploration diversity

This enables:

RL vs RLF comparison

RLF vs GA

MANDO vs all baselines

Comparisons must be apples-to-apples.

Logging Architecture (Critical for Long-Term Growth)

Logging is centralized at application entry points.

Modules define loggers:

logger = logging.getLogger(__name__)

Only entry points configure logging:

if not logging.getLogger().hasHandlers():
    logging.basicConfig(...)

Why:

Prevent duplicate handlers

Maintain framework compatibility (Streamlit, CLI)

Preserve clean separation of concerns

Agents never configure logging.

This ensures portability to:

Embedded systems

Headless deployments

Field logging

Hardware safety audits

Mandelbrot-Based Meta Exploration (MANDO Concept)

Future direction:

Instead of fixing parameter 
𝑐
c, allow it to evolve.

Two approaches:

Deterministic schedule across Mandelbrot parameter space

Error-correcting mutation (Reed-Solomon-inspired constraints)

Fitness-weighted parameter mutation

Conceptually:

RLF = Julia dynamics at fixed c

MANDO = exploration across Mandelbrot space of c

This becomes meta-learning over exploration patterns.

Hardware Parity Considerations

GridEnv is a controlled abstraction.

Real hardware introduces:

Sensor noise

Actuator lag

Energy constraints

Safety bounds

Exploration strategies must remain:

Bounded

Recoverable

Interruptible

Fractal exploration is acceptable only if:

Action deltas remain bounded

Fail-safe termination exists

Logging captures trajectory

This is non-negotiable in UGV deployments.

Design Guardrails

Do not:

Put learning logic in UI

Put logging configuration in agents

Allow agents to control training loop

Couple environment physics to agent internals

Always:

Keep agent stateless between episodes except Q-table

Keep exploration swappable

Keep evaluation measurable

Experimental Discipline

Before adding new math:

Define measurable hypothesis

Define control agent

Define metric

Run multiple seeds

Log results

Fractal ideas are not the goal.

Measurable improvement is the goal.

Long-Term Direction

This project is converging toward:

Unified simulation + hardware stack

Swappable exploration strategies

Structured experiment harness

Reproducible learning experiments

Fractal / GA / classical RL coexistence

This is not an assignment.

This is an extensible robotics learning framework.