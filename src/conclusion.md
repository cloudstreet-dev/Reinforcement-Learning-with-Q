# Where to Go From Here

You've implemented reinforcement learning from scratch in kdb+/q. Epsilon-greedy bandits, dynamic programming on GridWorld, Monte Carlo Blackjack, Q-Learning on cliff edges, n-step returns with eligibility traces, tile-coded mountain car, and policy gradients on a corridor that deliberately lies about its own structure.

Each of these is directly from Sutton & Barto. Each runs in q. None of them are toy demonstrations—they're complete implementations of real algorithms that appear in papers, production systems, and textbooks studied by ML researchers worldwide.

## What You Actually Learned

The algorithms in this book aren't just implementations. They encode a set of ideas that recur at every level of modern RL:

**The value of bootstrapping.** DP uses model-based bootstrapping; TD uses bootstrapping from estimates. Both reduce variance at the cost of some bias. Every practical RL system bootstraps somewhere.

**Exploration is not optional.** The bandit algorithms established this; every subsequent chapter reinforced it. Epsilon-greedy is the simplest possible approach. UCB is principled. Softmax policies are smooth. The underlying problem—you don't know what you don't know—never disappears.

**On-policy vs off-policy matters.** SARSA vs Q-Learning isn't just an algorithmic choice; it's a statement about what you want to learn (the policy you're executing, or the optimal policy). In production RL—where your data comes from a deployed policy that may not be the one you're learning—this distinction is critical.

**Function approximation changes the guarantees but not the intuitions.** Semi-gradient TD converges to a TD fixed point, not the true value function. But the TD fixed point is usually good enough, and the architecture choices (state aggregation, tile coding, neural networks) determine how well you generalise.

**Policy gradient methods are general but noisy.** REINFORCE converges to local optima. Baselines help. The modern variants (PPO, SAC) are engineering solutions to the variance problem, built on the same theoretical foundation.

## Extending the Implementations

The code in this book is deliberately minimal. Extending it is the actual learning exercise. Some suggestions:

**Add experience replay to Q-Learning.** Store transitions in a table; sample random batches for updates. This breaks correlations between consecutive samples and dramatically stabilises learning. It's the first ingredient of DQN.

```q
/ Replay buffer: fixed-size table of (s, a, r, s2, done) transitions
initReplay:{[capacity]
  `capacity`size`idx`buffer!(capacity;0i;0i;
    ([]s:capacity#0i; a:capacity#0i; r:capacity#0f;
       s2:capacity#0i; done:capacity#0b))
  }

/ Add transition to circular buffer
addTransition:{[buf;s;a;r;s2;done]
  idx:buf`idx;
  buf[`buffer;idx]:(`s`a`r`s2`done!(s;a;r;s2;done));
  buf[`idx]:(idx+1) mod buf`capacity;
  buf[`size]:buf[`capacity]&buf[`size]+1;
  buf
  }

/ Sample random minibatch
sampleBatch:{[buf;n]
  n?buf[`buffer] til buf`size
  }
```

**Implement Double Q-Learning.** Q-Learning overestimates action values because it uses the same network to select and evaluate actions. Double Q-Learning uses two Q-tables: one selects the action, the other evaluates it.

**Add a target network.** Freeze a copy of \\(Q\\) for computing targets; update it every \\(C\\) steps. This is the second key ingredient of DQN.

**Try continuous action spaces.** Replace the softmax policy with a Gaussian: output mean \\(\mu(s;\theta)\\) and log-std \\(\log\sigma(s;\theta)\\); sample actions from \\(\mathcal{N}(\mu, \sigma^2)\\). This is the foundation of SAC, DDPG, and TD3.

## The kdb+/q Angle

The implementations in this book use standard q idioms: scan for iteration, functional amend for updates, vectorised operations for batch processing. The language forced certain clarity—when the Bellman update is one line of q, it's obvious what it means.

For production use in financial contexts, RL has specific applications that play to q/kdb+ strengths:

**Optimal execution.** Market impact models for liquidating a position: state is (remaining inventory, time left, current price), action is trade size. The Almgren-Chriss model is a solved RL problem; the reality is harder.

**Market making.** State is (spread, inventory, volatility), action is (bid, ask) offsets. The reward is P&L minus inventory risk. This is naturally a continuous-action MDP.

**Feature selection and regime detection.** Use RL to select which features to use in downstream models, conditioned on detected market regime. The state space is high-dimensional; function approximation is mandatory.

**Portfolio rebalancing.** State is (current weights, signals), action is trades to execute, reward is risk-adjusted return. Multi-asset, high-dimensional, with transaction costs.

All of these are harder than anything in this book. They share the same structure: state, action, reward, update. The algorithms are the same. The engineering is not.

## The Books You Should Read Next

**Sutton & Barto**, of course. Everything in this book is a translation of their text; the original has proofs, history, intuitions, and exercises we didn't cover. It is freely available. There is no excuse not to read it.

**Szepesvári, *Algorithms for Reinforcement Learning*.** A shorter, more mathematical treatment. Covers convergence proofs for TD methods and linear function approximation. Worth reading after S&B.

**Bertsekas, *Reinforcement Learning and Optimal Control*.** Connects RL to classical optimal control theory. More rigorous, different emphasis. Essential if you work in continuous-time or continuous-control domains.

**Spinning Up in Deep RL (OpenAI).** Code-first introduction to modern deep RL: VPG, TRPO, PPO, DDPG, TD3, SAC. In Python (sorry), but the implementations are clear and the exposition is good.

## One Last Thing

The 10-armed bandit in Chapter 2 and the short corridor in Chapter 9 are pedagogically similar: both are deceptively simple environments that reveal something real about learning under uncertainty. The bandit reveals the exploration-exploitation tradeoff. The corridor reveals that stochastic policies are sometimes strictly necessary.

The algorithms that work on these toy problems work—conceptually, modulo engineering—on everything else. Deep networks don't change what Q-Learning is; they change how \\(Q(s,a)\\) is parameterised. Continuous action spaces don't change what policy gradient is; they change how \\(\pi(a|s;\theta)\\) is represented.

The foundational ideas are stable. The implementations in q are fast and clear. The rest is scale and patience.

That, and a willingness to watch your agent fall off a cliff several thousand times before it learns to walk around.
