Mathematical Foundations for Reinforcement Learning (RC Agents)

This document explains the core math required to understand how this project works.

## Table of Contents
- [Table of Contents](#table-of-contents)
- [

1. Numbers and Precision
Integers (int)

Whole numbers:

0, 1, 2, 10, -3

Used for:

Grid coordinates

Episode counters

Action IDs

Floating Point Numbers (float)

Decimals:

0.5

-1.0

0.95

Used for:

Rewards

Learning rate α (alpha)

Discount factor γ (gamma)

Q-values

Computers represent floats using IEEE-754 binary format.

This means:

0.1 + 0.2 != 0.3

This is normal. RL tolerates tiny floating error.

2. Algebra Refresher

Python follows standard algebra rules.

Operators
Meaning	Symbol
Add	+
Subtract	-
Multiply	*
Divide	/
Power	**
Order of Operations (PEMDAS)
result = 2 + 3 * 4

= 14
Multiplication happens before addition.

Parentheses override:

result = (2 + 3) * 4

= 20

This matters in update equations.

3. Functions

A function maps inputs to outputs.

Mathematically:

𝑓
(
𝑥
)
=
𝑥
2
f(x)=x
2

In Python:

def f(x):
    return x**2

Reinforcement learning repeatedly applies update functions.

4. Sets and States

A state space is a set of all possible states.

In a 5×5 grid:

𝑆
=
{
(
0
,
0
)
,
(
0
,
1
)
,
.
.
.
,
(
4
,
4
)
}
S={(0,0),(0,1),...,(4,4)}

Total states:

∣
𝑆
∣
=
𝑟
𝑜
𝑤
𝑠
×
𝑐
𝑜
𝑙
𝑠
∣S∣=rows×cols

If rows = 5 and cols = 5:

25
 states
25 states

Each state can have multiple actions.

5. Vectors

A vector is an ordered list of numbers.

Example:

[
1.0
,
0.5
,
−
0.2
,
3.1
]
[1.0,0.5,−0.2,3.1]

In this project:

Each state stores a vector:

𝑄
(
𝑠
)
=
[
𝑄
(
𝑠
,
𝐹
𝑂
𝑅
𝑊
𝐴
𝑅
𝐷
)
,
𝑄
(
𝑠
,
𝐵
𝐴
𝐶
𝐾
𝑊
𝐴
𝑅
𝐷
)
,
𝑄
(
𝑠
,
𝑅
𝐼
𝐺
𝐻
𝑇
)
,
𝑄
(
𝑠
,
𝐿
𝐸
𝐹
𝑇
)
]
Q(s)=[Q(s,FORWARD),Q(s,BACKWARD),Q(s,RIGHT),Q(s,LEFT)]

This is a value vector per state.

6. Maximum Function

We use:

max
⁡
(
𝑥
1
,
𝑥
2
,
.
.
.
,
𝑥
𝑛
)
max(x
1
	​

,x
2
	​

,...,x
n
	​

)

This selects the largest number.

In RL:

max
⁡
𝑎
𝑄
(
𝑠
′
,
𝑎
)
a
max
	​

Q(s
′
,a)

Means:

From the next state, what is the best action value available?

This drives greedy learning.

7. Expected Value (Core Idea)

Expected value means:

Long-term average outcome.

If a reward is random:

𝐸
[
𝑅
]
=
∑
𝑝
(
𝑟
)
⋅
𝑟
E[R]=∑p(r)⋅r

In Q-learning, Q(s,a) approximates:

𝐸
[
future total reward
]
E[future total reward]

The Q-table estimates expectation through repeated updates.

8. Discount Factor γ (gamma)

Gamma controls how much future rewards matter.

0
≤
𝛾
≤
1
0≤γ≤1

If:

γ = 0 → agent only cares about immediate reward

γ = 1 → agent values future equally

Example:

Reward = 0
Future best value = 10
γ = 0.9

0
+
0.9
×
10
=
9
0+0.9×10=9

Future is discounted slightly.

9. Learning Rate α (alpha)

Alpha controls update speed.

0
<
𝛼
≤
1
0<α≤1

If:

α = 1 → overwrite old value completely

α small → slow, stable learning

Alpha performs weighted averaging.

10. The Q-Learning Update Rule

The central equation:

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
[
𝑟
+
𝛾
max
⁡
𝑎
𝑄
(
𝑠
′
,
𝑎
)
−
𝑄
(
𝑠
,
𝑎
)
]
Q(s,a)←Q(s,a)+α[r+γ
a
max
	​

Q(s
′
,a)−Q(s,a)]

Break it down:

Step 1: Compute target
𝑡
𝑎
𝑟
𝑔
𝑒
𝑡
=
𝑟
+
𝛾
max
⁡
𝑄
(
𝑠
′
)
target=r+γmaxQ(s
′
)
Step 2: Compute error
𝑒
𝑟
𝑟
𝑜
𝑟
=
𝑡
𝑎
𝑟
𝑔
𝑒
𝑡
−
𝑐
𝑢
𝑟
𝑟
𝑒
𝑛
𝑡
_
𝑄
error=target−current_Q
Step 3: Apply fraction of error
𝑛
𝑒
𝑤
_
𝑄
=
𝑐
𝑢
𝑟
𝑟
𝑒
𝑛
𝑡
_
𝑄
+
𝛼
×
𝑒
𝑟
𝑟
𝑜
𝑟
new_Q=current_Q+α×error

This is incremental correction.

Nothing more.

11. Bellman Equation (Conceptual Form)

The Bellman optimality equation:

𝑄
∗
(
𝑠
,
𝑎
)
=
𝐸
[
𝑟
+
𝛾
max
⁡
𝑎
′
𝑄
∗
(
𝑠
′
,
𝑎
′
)
]
Q
∗
(s,a)=E[r+γ
a
′
max
	​

Q
∗
(s
′
,a
′
)]

Q-learning approximates this iteratively.

You are solving a recursive fixed-point equation through sampling.

12. Markov Property

A system is Markov if:

𝑃
(
𝑠
𝑡
+
1
∣
𝑠
𝑡
)
P(s
t+1
	​

∣s
t
	​

)

depends only on the current state.

Not the full history.

Grid world satisfies this.

That’s why Q-learning works.

13. Convergence

Q-learning converges if:

All state-action pairs are explored

Learning rate decreases or is small

Rewards are bounded

Over time:

𝑄
(
𝑠
,
𝑎
)
→
𝑄
∗
(
𝑠
,
𝑎
)
Q(s,a)→Q
∗
(s,a)

The optimal value function.

14. Geometry of the Value Landscape

The Q-table forms a discrete value surface.

Each grid cell stores:

𝑉
(
𝑠
)
=
max
⁡
𝑎
𝑄
(
𝑠
,
𝑎
)
V(s)=
a
max
	​

Q(s,a)

This produces a gradient toward the goal.

The agent climbs that gradient.

15. Why Negative Step Reward Works

Each step gives:

𝑟
=
−
1
r=−1

Goal gives:

𝑟
=
0
r=0

Total reward equals:

−
number of steps
−number of steps

Thus:

Maximizing reward = minimizing path length.

Clean. Efficient. No artificial shaping.

16. Probability Basics

Exploration rate ε:

0
≤
𝜀
≤
1
0≤ε≤1

If ε = 0.1:

10% random actions
90% greedy actions

This balances:

Exploration

Exploitation

17. Linear Algebra (What You Actually Need)

For this project, you only need:

Vectors

Maximum selection

Scalar multiplication

Weighted averaging

No matrices.
No eigenvalues.
No calculus.

18. Big Picture

Reinforcement learning in this repo is:

Discrete

Tabular

Iterative

Sample-based

Converging toward optimal value estimates

Mathematically, it is:

Repeated application of a contraction mapping toward a fixed point.

But practically:

Correct small prediction errors over time.

19. What To Study Next

If you want stronger foundations:

Probability theory basics

Expected value

Dynamic programming

Bellman equations

Markov Decision Processes