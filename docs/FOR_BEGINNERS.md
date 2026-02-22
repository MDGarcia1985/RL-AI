RC Agents – Beginner Guide to Python, NumPy, Pandas, and Math

Welcome. This project teaches reinforcement learning using clean, readable Python.
Before diving into Q-learning, you need a solid foundation in:

if statements

integers (int) and floats (float)

NumPy

Pandas

How math works in Python

This guide explains each in plain language.

1. ## What Is This Project?

The main entry point of the package is:

__main__

That file creates:

An environment (GridEnv)

An agent (QAgent)

A training loop (run_training)

The package structure is enabled by:

__init__

That file simply tells Python:
"This folder is a package. You can import from it."

If you understand this, you already understand the architecture.

2. ## int and float

Python has numeric types.

Integer (int)

Whole numbers:

x = 5
y = -3

Used for:

Grid coordinates

Episode counters

Action IDs

Example from grid world:

row = 3
col = 2
Float (float)

Decimal numbers:

alpha = 0.5
gamma = 0.9
reward = -1.0

Used for:

Learning rate α (alpha)

Discount factor γ (gamma)

Rewards

Q-values

Reinforcement learning math requires floats because:

Learning is fractional

Updates are continuous

Values are rarely whole numbers

3. ## if Statements (Decision Making)

An if statement is a conditional:

if x > 5:
    print("Large number")

In this project, if controls:

Example: epsilon-greedy exploration
if random_value < epsilon:
    # explore
else:
    # exploit

Meaning:

Sometimes explore

Otherwise use best-known action

Example: goal detection
if self.pos == self.config.goal:
    reward = 0.0
    done = True

Meaning:

If agent reaches goal

Stop episode

4. ## How Math Works in Python

Python follows normal algebra rules:

Operation	Symbol
Addition	+
Subtraction	-
Multiplication	*
Division	/
Power	**
Order of Operations

Python follows PEMDAS:

result = 2 + 3 * 4   # = 14

Multiplication happens first.

Floating Point Precision

Python uses IEEE-754 floating point.

This means:

0.1 + 0.2 != 0.3

This is normal in computing.

For ML and RL, this is acceptable.

5. ## NumPy (The Engine of Fast Math)

NumPy is the backbone of this project.

Imported like this:

import numpy as np

NumPy provides:

Fast arrays

Vector math

argmax

max

random number generators

NumPy Arrays

Instead of Python lists:

[0, 0, 0, 0]

We use:

np.zeros(4)

Why?

Faster

Better for math

Supports vectorized operations

Example Q-table row:

array([Q_forward, Q_backward, Q_right, Q_left])
np.argmax()

Finds index of largest value:

best_action = np.argmax(q_values)

If:

[1.0, 2.5, 0.3, 1.8]

Returns:

1

Because 2.5 is largest.

np.max()

Returns the largest value:

np.max(q_values)

Used in Q-learning update rule.

Random Generator
rng = np.random.default_rng(seed)

Why not random module?

Because:

NumPy RNG is modern

Reproducible

Used in ML workflows

6. ## Pandas (For Displaying Tables)

Pandas is used only in the Streamlit UI.

import pandas as pd

It converts NumPy arrays into readable tables.

Example:

df = pd.DataFrame(Q, columns=actions)

This creates a labeled table:

state	FORWARD	BACKWARD	RIGHT	LEFT

Pandas is for:

Data display

UI inspection

Debugging

It does NOT perform learning.

7. ## The Q-Learning Equation (The Core Math)

This is the update rule:

𝑄
(
𝑠
,
𝑎
)
←
𝑄
(
𝑠
,
𝑎
)
+
𝛼
(
𝑟
𝑒
𝑤
𝑎
𝑟
𝑑
+
𝛾
𝑚
𝑎
𝑥
(
𝑄
(
𝑠
′
,
𝑎
′
)
)
−
𝑄
(
𝑠
,
𝑎
)
)
Q(s,a)←Q(s,a)+α(reward+γmax(Q(s
′
,a
′
))−Q(s,a))

In Python:

target = reward + gamma * np.max(q_s2)
q_s[a_index] = q_s[a_index] + alpha * (target - q_s[a_index])

Breakdown:

reward → immediate feedback

gamma → future importance

alpha → learning speed

max(Q(s',a')) → best future value

This is iterative improvement.

Nothing mystical. Just controlled incremental updates.

8. ## Why We Use Tuples for State

Grid state is:

(row, col)

Example:

(3, 2)

Why tuple?

Because tuples are:

Immutable

Hashable

Safe dictionary keys

Lists cannot be dictionary keys.

9. ## What You Should Practice

Before modifying this project, make sure you can:

Write a basic if statement

Create a NumPy array

Use np.argmax

Create a simple Pandas DataFrame

Understand multiplication vs addition precedence

Read a dictionary access like q_table[state]

10. ## Mental Model of the System

Environment:

Defines rules

Agent:

Stores Q-table

Chooses actions

Runner:

Loops episodes

UI:

Displays results

Everything else is organization.

11. ## Rolling Windows, `deque`, and `.popleft()`

When building learning systems (like reinforcement learning agents), we often want to compute statistics over a **moving window**.

For example:

- Win rate over the last 200 episodes
- Average steps-to-goal over recent performance
- Detection of learning saturation

To do this efficiently, we use Python’s:


collections.deque


---

### What is a `deque`?

A **deque** (double-ended queue) is a data structure that allows fast:

- Add to the right → `.append(x)`
- Add to the left → `.appendleft(x)`
- Remove from the right → `.pop()`
- Remove from the left → `.popleft()`

All of these are **O(1)** operations.

That means they are constant-time, no matter how large the structure becomes.

---

### Why not use a normal list?

If you try to remove the first item of a list:

```python
lst.pop(0)

Python must shift every remaining element forward.

That is O(n) time.

For rolling statistics, this becomes slow.

deque.popleft() avoids that cost entirely.

What does .popleft() do?

It removes and returns the oldest element in the queue.

Example:

from collections import deque

d = deque([10, 20, 30])

x = d.popleft()

print(x)   # 10
print(d)   # deque([20, 30])

So:

The leftmost element (oldest) is removed

It is returned so we can update totals correctly

How This Applies to Reinforcement Learning

In our training system, we track:

Wins in the last W episodes

Steps-to-goal in the last W episodes

To maintain this window efficiently:

If the window is full:

Remove the oldest episode using .popleft()

Append the newest episode

Update running totals

This allows us to compute:

Rolling win-rate

Efficiency plateau detection

Saturation detection

Without recalculating sums over the entire window each time.

Where You Can See This in the Codebase

See:

rc_agents/rcg_edge/runners/convergence_tracker.py

Look at:

_evict_if_full()

_append_episode()

That is a real-world example of:

Using deque

Maintaining rolling statistics

Building a performance monitor for machine learning systems

Key Concept

A rolling window is like a conveyor belt:

[oldest] → → → [newest]

Each new episode pushes forward.
Old episodes fall off the left side.

deque.popleft() removes the episode that falls off.

That is how we efficiently track learning progress over time.


---

If you’d like, I can also:

- Add a small visual ASCII diagram of the rolling window math
- Add a mini coding exercise for students
- Or create a short "Why this matters for AI training" reflection section

You’re building a real framework now — documenting these primitives is the right move.

12. What To Learn Next

If you're serious about RL:

Linear algebra basics

Expected value

Bellman equation

Markov Decision Processes

Sutton & Barto (Reinforcement Learning)

Final Advice

Reinforcement learning is not magic.

It is:

Iteration

Controlled randomness

Incremental updates

Simple math repeated thousands of times

If you understand:

if

floats

arrays

max

loops

You understand the foundation.

Build from there.