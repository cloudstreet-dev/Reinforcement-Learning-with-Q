# Why Reinforcement Learning in Q? Why Not.

There are two kinds of kdb+/q developers. Those who have looked at the existing RL literature and thought "this would map beautifully to an array language", and those who haven't looked at it yet. You're reading this, so you're either in the first group or you're about to join it.

Reinforcement learning is the branch of machine learning concerned with agents that learn by doing. An agent observes a state, takes an action, receives a reward, and updates its beliefs about what actions are good. Repeat until convergence, publication, or deadline—whichever comes first.

The canonical reference is Sutton and Barto's *Reinforcement Learning: An Introduction* (second edition, 2018, freely available at [incompleteideas.net](http://incompleteideas.net/book/the-book-2nd.html)). This book follows that text chapter by chapter, implementing every algorithm in kdb+/q. If you have the S&B book open in one window, you should be able to read this and understand exactly which equation we're implementing and why.

## Why Q Is Actually Good at This

The standard argument for Python is its ecosystem. NumPy, PyTorch, Gymnasium, stable-baselines3—it's all there. The counterargument is that you are a kdb+/q developer and you don't live in Python. More substantively:

**Arrays are first-class citizens.** RL algorithms operate on tables of state-action-value estimates, batch updates to value functions, and vectorised probability distributions. In q, this is not boilerplate. It is syntax.

**The tabular methods that dominate the early S&B chapters are literally table operations.** A Q-table is a dictionary. A value function is a list. Policy evaluation is a matrix operation. Q makes these feel natural rather than imported.

**q's terseness matches the mathematical notation.** When Sutton & Barto write \\(Q(s,a) \leftarrow Q(s,a) + \alpha[R + \gamma \max_{a'} Q(s',a') - Q(s,a)]\\), the q implementation is shorter than the LaTeX. This is not a coincidence.

**Performance.** The tabular methods in this book run in microseconds per step. The q implementations are fast enough to run thousands of episodes in interactive development time.

## What This Book Is Not

It is not a gentle introduction to reinforcement learning. Read S&B for that—it's excellent. It is not an introduction to kdb+/q. If `imax`, `{x+y}/`, and `.[t;(i;`col);:;v]` require explanation before you can continue, spend some time with the q language reference first.

It is also not a comprehensive treatment of deep RL. Deep Q-Networks, policy gradient with neural networks, and transformer-based agents are beyond scope. This book covers the classical, tabular, and linear-approximation methods from S&B chapters 2–13. These are the foundations. They also happen to be complete and correct in a way that deep methods still aspire to be.

## How to Use This Book

Each chapter corresponds roughly to one or two chapters of S&B and implements the algorithms from that section. The code is meant to run. Load it into a q session and experiment. Change the learning rate. Break the algorithm. Observe what happens.

The environments are deliberately simple—GridWorlds, Blackjack, Cliff Walking. These are the same environments S&B use. If your output matches the figures in that book, you've understood the algorithm. If it doesn't, you've found something more interesting: a bug, a subtlety, or an insight.

## Setup

You need kdb+/q. The personal (32-bit) edition from [kx.com](https://kx.com/developers/download-licenses/) is sufficient for every example in this book. Each chapter is self-contained: copy the code into a `.q` file, load it with `\l filename.q`, or paste it directly into a q session.

We define one utility at the start of almost every chapter:

```q
/ Box-Muller transform: standard normal sample N(0,1)
pi:acos -1f
normal:{[] sqrt[-2f*log rand 1.0f]*cos[2f*pi*rand 1.0f]}
```

This generates normally-distributed random numbers, which appear everywhere in RL experiments. q lacks a built-in normal sampler, so we bring our own. Box-Muller is exact, not approximate, and fast enough for our purposes.

The rest is just algorithms.
