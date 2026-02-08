# RC Agents - Reinforcement Learning Package

**By Michael Garcia**  
CSC370 Spring 2026  
michael@mandedesign.studio

## Overview

A Python package for reinforcement learning agents designed for grid-based navigation tasks. This package serves as the foundation for the RC Guardian autonomous vehicle project, implementing Q-learning algorithms with proper software engineering practices.

## Features

- **Modular Agent Architecture**: Clean separation between agents, environments, and training logic
- **Q-Learning Implementation**: Tabular Q-learning with epsilon-greedy exploration
- **Grid Environment**: Configurable 2D grid world for navigation training
- **Professional Testing**: Comprehensive test suite with pytest
- **Type Safety**: Full type hints and protocol-based interfaces

## Project Structure

```
rc_agents/
├── edge_ai/rcg_edge/           # Core AI components
│   ├── agents/                 # Agent implementations
│   │   ├── base.py            # Agent protocol and base classes
│   │   ├── random_agent.py    # Baseline random agent
│   │   └── q_agent.py         # Q-learning agent
│   └── runners/
│       └── train_runner.py    # Training loop coordination
├── envs/
│   └── grid_env.py           # Grid environment implementation
├── testers/                   # Test suite
│   ├── test_grid_env.py      # Environment tests
│   ├── test_random_agent.py  # Random agent tests
│   ├── test__action_selection.py # Action selection tests
│   └── test_q_update.py      # Q-learning update tests
├── ui/
│   └── gui_main.py           # Graphical interface (legacy)
├── data/                     # Training outputs
│   ├── botmovements.txt      # Movement logs
│   └── q_table.npy          # Saved Q-tables
└── __main__.py              # Package entry point
```

## Quick Start

### Installation
```bash
cd rc_agents
pip install -e .
```

### Basic Usage
```python
from rc_agents.envs.grid_env import GridEnv, GridConfig
from rc_agents.edge_ai.rcg_edge.agents.q_agent import QAgent, QConfig
from rc_agents.edge_ai.rcg_edge.runners.train_runner import run_training

# Create environment
env = GridEnv(GridConfig(rows=5, cols=5, start=(0, 0), goal=(4, 4)))

# Create Q-learning agent
agent = QAgent(QConfig(alpha=0.5, gamma=0.9, epsilon=0.1), seed=123)

# Run training
results = run_training(env=env, agent=agent, episodes=50, max_steps=200)

# Check performance
wins = sum(1 for r in results if r.reached_goal)
print(f"Success rate: {wins}/{len(results)}")
```

### Run Package Directly
```bash
python -m rc_agents
```

## Agents

### RandomAgent
RandomAgent serves as a baseline and sanity check for environment-agent integration. It selects actions uniformly at random and provides a performance baseline for comparison with learning agents.

### QAgent
Implements tabular Q-learning with epsilon-greedy action selection. Features:
- Configurable learning rate (α), discount factor (γ), and exploration rate (ε)
- Automatic Q-table initialization
- Proper action-value updates following Sutton & Barto

## Q-Learning Algorithm

The Q-value update follows the standard Q-learning rule as defined by Sutton and Barto (2018):

**Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]**

Where:
- Q(s,a) = Q-value for state s and action a
- α = learning rate (how fast to learn)
- r = reward received
- γ = discount factor (importance of future rewards)
- s' = next state
- max_a' Q(s',a') = maximum Q-value for next state over all possible actions

## Environment

### GridEnv
A configurable 2D grid world where agents navigate from start to goal positions.

**Configuration Options:**
- `rows`, `cols`: Grid dimensions
- `start`: Starting position (row, col)
- `goal`: Target position (row, col)

**Actions:**
- `1` = FORWARD (up)
- `2` = BACKWARD (down)
- `3` = RIGHT
- `4` = LEFT

**Rewards:**
- `-1.0` for each step (encourages efficiency)
- `0.0` for reaching the goal

## Testing

Run the comprehensive test suite:
```bash
pytest rc_agents/testers/
```

**Test Coverage:**
- Environment mechanics and boundary conditions
- Agent action selection (exploration vs exploitation)
- Q-learning update correctness
- Integration between components

## Development Notes

**pytest tutorial video**: https://youtu.be/EgpLj86ZHFQ?si=KGs2Uu_7bR09LsnI

**heat map tutorial**
https://www.geeksforgeeks.org/machine-learning/q-learning-in-python/

**Design Principles:**
- Protocol-based interfaces for flexibility
- Type safety with comprehensive hints
- Modular architecture for extensibility
- Professional testing practices

## Future Roadmap

This package serves as the foundation for the RC Guardian project:
- Integration with real hardware (Arduino UNO Q)
- Computer vision and sensor fusion
- GPS-based outdoor navigation
- Advanced RL algorithms (DQN, PPO)
- ROS integration for robotics deployment

## References

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2nd ed.). MIT Press.