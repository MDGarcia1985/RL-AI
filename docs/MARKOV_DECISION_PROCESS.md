Markov Decision Processes (MDP) — The Mathematical Framework Behind This Project

This document explains the formal structure underlying the grid world and Q-learning agent.

Reinforcement learning is not arbitrary experimentation.
It is grounded in the theory of Markov Decision Processes.

1. What Is a Markov Decision Process?

An MDP is defined as a 5-tuple:

(
𝑆
,
𝐴
,
𝑃
,
𝑅
,
𝛾
)
(S,A,P,R,γ)

Where:

S = Set of states

A = Set of actions

P = Transition probability function

R = Reward function

γ (gamma) = Discount factor

If you understand those five components, you understand the system.

2. The Markov Property

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
,
𝑎
𝑡
)
P(s
t+1
	​

∣s
t
	​

,a
t
	​

)

depends only on:

Current state

Current action

Not the entire history.

This means:

The present fully determines the future.

Grid world satisfies this property.

Your next position depends only on:

Where you are

Which action you take

Nothing else matters.

3. State Space (S)

In this project:

A state is:

𝑠
=
(
𝑟
𝑜
𝑤
,
𝑐
𝑜
𝑙
)
s=(row,col)

For a 5×5 grid:

∣
𝑆
∣
=
25
∣S∣=25

Each coordinate pair represents a unique state.

The state space is finite and discrete.

4. Action Space (A)

Actions are discrete:

FORWARD

BACKWARD

RIGHT

LEFT

So:

∣
𝐴
∣
=
4
∣A∣=4

Every state has the same action set.

This is a stationary action space.

5. Transition Function (P)

The transition function:

𝑃
(
𝑠
′
∣
𝑠
,
𝑎
)
P(s
′
∣s,a)

Defines the probability of moving from state 
𝑠
s to 
𝑠
′
s
′
 given action 
𝑎
a.

In this grid world:

Transitions are deterministic.

Meaning:

𝑃
(
𝑠
′
∣
𝑠
,
𝑎
)
=
1
P(s
′
∣s,a)=1

for exactly one next state.

There is no randomness in movement.

6. Reward Function (R)

Reward function:

𝑅
(
𝑠
,
𝑎
,
𝑠
′
)
R(s,a,s
′
)

Defines immediate reward received after transition.

In this project:

Every step → -1

Reaching goal → 0

This creates a pressure toward shortest paths.

Reward structure defines behavior.

7. Discount Factor (γ)
0
≤
𝛾
≤
1
0≤γ≤1

Gamma determines how much future reward matters.

If:

γ = 0 → only immediate reward matters

γ close to 1 → future rewards matter strongly

In this grid:

𝛾
=
0.9
γ=0.9

This encourages long-term planning.

8. Policy (π)

A policy is a function:

𝜋
(
𝑠
)
→
𝑎
π(s)→a

It tells the agent what action to take in each state.

Two types:

Deterministic policy

Stochastic policy

Epsilon-greedy is stochastic.

9. Value Functions

There are two primary value functions.

State-Value Function
𝑉
𝜋
(
𝑠
)
V
π
(s)

Expected return from state 
𝑠
s following policy 
𝜋
π.

Action-Value Function (Q-function)
𝑄
𝜋
(
𝑠
,
𝑎
)
Q
π
(s,a)

Expected return from taking action 
𝑎
a in state 
𝑠
s.

This project learns:

𝑄
(
𝑠
,
𝑎
)
Q(s,a)

The Q-table approximates optimal action values.

10. Return (G)

Return is total discounted reward:

𝐺
𝑡
=
𝑟
𝑡
+
1
+
𝛾
𝑟
𝑡
+
2
+
𝛾
2
𝑟
𝑡
+
3
+
.
.
.
G
t
	​

=r
t+1
	​

+γr
t+2
	​

+γ
2
r
t+3
	​

+...

Q-learning estimates expected return.

11. Optimal Policy

An optimal policy satisfies:

𝜋
∗
(
𝑠
)
=
𝑎
𝑟
𝑔
𝑚
𝑎
𝑥
𝑎
𝑄
∗
(
𝑠
,
𝑎
)
π
∗
(s)=argmax
a
	​

Q
∗
(s,a)

It chooses the action with the highest expected value.

Greedy selection extracts this policy from the Q-table.

12. Bellman Optimality Equation

The optimal Q-function satisfies:

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

This is a recursive definition.

Q-learning iteratively approximates this fixed point.

13. Why This Converges

Q-learning converges if:

Every state-action pair is visited infinitely often

Learning rate is appropriate

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
14. Deterministic vs Stochastic MDP

This grid world is:

Deterministic transitions

Deterministic rewards

Finite state space

Fully observable

This is the simplest valid MDP.

More complex systems add:

Noise

Partial observability

Continuous state spaces

But the structure remains the same.

15. Relationship to Dynamic Programming

If full transition model is known:

You can solve MDP using:

Value Iteration

Policy Iteration

Q-learning differs because:

It does not require the transition model.

It learns through interaction.

16. Why MDP Matters

Without MDP structure:

No convergence guarantee

No theoretical grounding

No defined objective

MDP provides:

Formal objective

Optimality definition

Convergence theory

This is why RL works.

17. In This Repository

Your system is:

Finite MDP

Tabular solution

Off-policy learning

Model-free

Value-based

It is the foundational RL configuration.

Master this before moving to:

Function approximation

Deep Q Networks

Policy gradients

18. Conceptual Summary

An MDP is:

A structured decision system

With states

With actions

With transition rules

With rewards

And future discounting

Reinforcement learning is:

The process of discovering the optimal policy inside that structure.