# Multi-Armed Bandits: Exploration, Exploitation, and Regret

*S&B Chapter 2*

The multi-armed bandit problem is RL's hello world. You have \\(k\\) slot machines (arms). Each arm \\(a\\) has a true expected reward \\(q^*(a)\\) that you don't know. Every time step, you pull an arm and observe a noisy reward. Your goal is to maximise total reward over time.

The fundamental tension: to know which arm is best, you must explore. But exploration takes time you could spend exploiting the arm you currently believe is best. Every moment spent exploring is potential reward foregone—this is *regret*. Every moment spent exploiting a suboptimal arm is also regret. You cannot win, only manage.

This tension—exploration vs. exploitation—is the central problem of reinforcement learning. It never goes away. Later chapters add states, transitions, and discount factors, but the bandit dilemma remains embedded in every algorithm.

## The 10-Armed Testbed

S&B's canonical benchmark (Figure 2.2): \\(k = 10\\) arms, true values \\(q^*(a) \sim \mathcal{N}(0, 1)\\), rewards \\(R \sim \mathcal{N}(q^*(a), 1)\\). Run for 1000 steps. Average over 2000 independent bandit problems to smooth the noise.

```q
/ ============================================================
/ S&B Chapter 2 — Multi-Armed Bandit Testbed
/ ============================================================

pi:acos -1f
normal:{[] sqrt[-2f*log rand 1.0f]*cos[2f*pi*rand 1.0f]}

/ Create a k-armed bandit: sample true values from N(0,1)
initBandit:{[k]
  qStar:{normal[]} each til k;
  `k`qStar`optimal!(k; qStar; imax qStar)
  }

/ Pull arm a: reward ~ N(q*(a), 1)
pullArm:{[bandit;a]
  bandit[`qStar][a] + normal[]
  }
```

## Epsilon-Greedy

The simplest strategy: with probability \\(\varepsilon\\), choose a random arm (explore); otherwise, choose the arm with the highest estimated value (exploit).

We track \\(Q(a)\\)—the sample mean of rewards from arm \\(a\\)—using the incremental update formula:

\\[Q_{n+1}(a) = Q_n(a) + \frac{1}{N_n(a)}\left[R_n - Q_n(a)\right]\\]

This is algebraically equivalent to computing a running average but requires only \\(O(1)\\) memory. No stored history, no growing lists.

```q
/ Agent state: estimated Q values, action counts, step counter
initAgent:{[k;eps]
  `k`eps`Q`N`t!(k; eps; k#0f; k#0i; 0i)
  }

/ Epsilon-greedy action selection
selectAction:{[agent]
  $[rand[1.0f] < agent`eps;
    first 1?agent`k;    / explore: uniform random arm
    imax agent`Q        / exploit: greedy arm
  ]}

/ Incremental mean update: Q[a] <- Q[a] + (R - Q[a]) / N[a]
updateQ:{[agent;a;r]
  n1:1i + agent[`N][a];
  dq:(r - agent[`Q][a]) % n1;
  agent,`N`Q`t!(@[agent`N;a;:;n1]; @[agent`Q;a;+;dq]; agent[`t]+1i)
  }

/ One step: act, observe, update; return augmented state
stepAgent:{[bandit;state]
  a:selectAction state;
  r:pullArm[bandit;a];
  upd:updateQ[state;a;r];
  upd,`_r`_opt!(r; a=bandit`optimal)
  }
```

> **The incremental update equation is the algebraic heart of everything that follows.** The update rule \\(Q \leftarrow Q + \alpha(R - Q)\\) says: move your estimate toward the new observation by a fraction \\(\alpha\\) of the error \\((R - Q)\\). When \\(\alpha = 1/N\\), this recovers the sample mean exactly. In later chapters, we'll use a fixed \\(\alpha\\) to weight recent rewards more heavily—useful when the environment changes over time.

```q
/ Run n steps using q's scan operator (\)
/ scan applies stepAgent repeatedly, returning all intermediate states
runExperiment:{[bandit;eps;nSteps]
  s0:initAgent[bandit`k; eps],`_r`_opt!(0f;0b);
  steps:1_ (nSteps (stepAgent[bandit;])\ s0);   / drop initial dummy state
  `rewards`optimal!({x`_r} each steps; {x`_opt} each steps)
  }

/ Average over nRuns independent bandit problems
runAveraged:{[k;eps;nSteps;nRuns]
  results:{runExperiment[initBandit[x]; y; z]}[k;;nSteps] each nRuns#eps;
  avgRewards: avg {x`rewards} each results;
  pctOptimal: avg {`float$x`optimal} each results;
  `avgRewards`pctOptimal!(avgRewards;pctOptimal)
  }
```

Running the testbed:

```q
/ Reproduce S&B Figure 2.2
k:10; nSteps:1000; nRuns:2000

r000:runAveraged[k; 0.00f; nSteps; nRuns];   / greedy
r010:runAveraged[k; 0.01f; nSteps; nRuns];   / epsilon = 0.01
r100:runAveraged[k; 0.10f; nSteps; nRuns];   / epsilon = 0.10

/ Average reward at final step
r000[`avgRewards][nSteps-1]   / ~1.0  (gets stuck on suboptimal arm)
r010[`avgRewards][nSteps-1]   / ~1.3
r100[`avgRewards][nSteps-1]   / ~1.4  (best short-term, but keeps exploring)
```

The greedy agent (\\(\varepsilon=0\\)) converges fastest initially but gets trapped. It commits to the first arm that looks good and never learns it might have been wrong. The \\(\varepsilon=0.1\\) agent explores more, learns better, and eventually outperforms—but wastes 10% of its time forever on random pulls.

## Upper Confidence Bound (UCB)

Epsilon-greedy explores randomly. UCB explores *intelligently*: prefer arms that either look good or haven't been tried much. The action selection rule (S&B Equation 2.10):

\\[A_t = \operatorname{argmax}_a \left[ Q_t(a) + c\sqrt{\frac{\ln t}{N_t(a)}} \right]\\]

The second term is the *uncertainty bonus*. An arm pulled rarely has high \\(\sqrt{\ln t / N_t(a)}\\), so it gets selected until we know enough about it. As \\(N_t(a)\\) grows, the bonus shrinks. \\(c\\) controls how aggressively we explore.

```q
/ UCB action selection
/ Handle division by zero: never-pulled arms get priority (bonus = +inf)
selectUCB:{[c;agent]
  t:agent`t;
  N:agent`N;
  Q:agent`Q;
  / Arms with N=0 must be tried first (set bonus to infinity)
  bonus:$[any N=0i;
    {$[x=0i;0w;0f]} each N;                  / 0w = positive infinity
    c*sqrt[log[`float$t] % `float$N]
    ];
  imax Q+bonus
  }

/ UCB agent step
stepUCB:{[bandit;c;state]
  a:selectUCB[c;state];
  r:pullArm[bandit;a];
  upd:updateQ[state;a;r];
  upd,`_r`_opt!(r; a=bandit`optimal)
  }

/ Run UCB experiment
runUCB:{[k;c;nSteps;nRuns]
  results:{[bandit]
    s0:initAgent[bandit`k; 0f],`_r`_opt!(0f;0b);
    steps:1_(nSteps (stepUCB[bandit;c;])\ s0);
    `rewards`optimal!({x`_r}each steps; {x`_opt}each steps)
    } each {initBandit[k]} each til nRuns;
  avgRewards:avg{x`rewards}each results;
  pctOptimal:avg{`float$x`optimal}each results;
  `avgRewards`pctOptimal!(avgRewards;pctOptimal)
  }
```

```q
/ Compare UCB c=2 with epsilon-greedy on the testbed
rUCB:runUCB[10; 2f; 1000; 2000];

/ UCB typically outperforms epsilon-greedy on stationary problems
/ by pulling suboptimal arms less often once they're understood
avg rUCB`avgRewards        / typically ~1.5
avg r100`avgRewards        / epsilon=0.1: ~1.35
```

UCB is theoretically attractive because it has *logarithmic regret*—provably near-optimal for stationary problems. The catch: it requires knowing \\(t\\) and assumes stationarity. When the environment shifts (arm values change over time), UCB's optimism about arms it once understood becomes a liability.

## Gradient Bandit

Both methods above learn action *values* and derive a policy from them. Gradient bandits do it differently: learn a *preference* \\(H_t(a)\\) for each arm and convert preferences to probabilities via softmax. Update preferences by gradient ascent on expected reward.

The update rule (S&B Equation 2.12):

\\[H_{t+1}(a) = H_t(a) + \alpha(R_t - \bar{R}_t)\left(\mathbf{1}_{a=A_t} - \pi_t(a)\right)\\]

where \\(\pi_t(a)\\) is the softmax probability of arm \\(a\\) and \\(\bar{R}_t\\) is a running reward baseline.

```q
/ Softmax of a real-valued vector
softmax:{[H]
  eH:exp H-max H;    / subtract max for numerical stability
  eH%sum eH
  }

/ Gradient bandit agent state
initGradient:{[k;alpha]
  `k`alpha`H`piBar`t!(k; alpha; k#0f; 0f; 0i)
  }

/ Select action by sampling from softmax distribution
selectGradient:{[agent]
  pi:softmax agent`H;
  / Sample from categorical distribution
  cumPi:sums pi;
  r:rand 1.0f;
  first where cumPi >= r
  }

/ Update preferences using gradient ascent
updateGradient:{[agent;a;r]
  pi:softmax agent`H;
  alpha:agent`alpha;
  Rbar:agent`piBar;
  delta:r-Rbar;
  t1:agent[`t]+1i;
  / One-hot indicator for chosen action
  oneHot:`float$(til agent`k)=a;
  / H[a] += alpha * (R - Rbar) * (1 - pi[a])
  / H[b] -= alpha * (R - Rbar) * pi[b]  for b != a
  newH:agent[`H]+alpha*delta*(oneHot-pi);
  / Update baseline: running mean of all rewards
  newRbar:Rbar+(r-Rbar)%t1;
  agent,`H`piBar`t!(newH;newRbar;t1)
  }

/ Full gradient bandit step
stepGradient:{[bandit;state]
  a:selectGradient state;
  r:pullArm[bandit;a];
  upd:updateGradient[state;a;r];
  upd,`_r`_opt!(r;a=bandit`optimal)
  }

/ Run gradient bandit experiment
runGradient:{[k;alpha;nSteps;nRuns]
  results:{[bandit]
    s0:initGradient[bandit`k;alpha],`_r`_opt!(0f;0b);
    steps:1_(nSteps (stepGradient[bandit;])\ s0);
    `rewards`optimal!({x`_r}each steps;{x`_opt}each steps)
    } each {initBandit[k]} each til nRuns;
  `avgRewards`pctOptimal!(avg{x`rewards}each results; avg{`float$x`optimal}each results)
  }
```

```q
/ alpha=0.1 typically works well; try 0.4 to see instability
rGrad:runGradient[10; 0.1f; 1000; 2000];
avg rGrad`pctOptimal    / ~75-80% optimal
```

## Which Algorithm Wins?

None of them, universally. Epsilon-greedy is robust and simple. UCB is theoretically principled for stationary problems. Gradient bandits don't require interpreting rewards as values—useful when the scale of rewards is unknown or varies. Parameter sensitivity differs across all three.

The bandit problem is elegantly tractable—we can analyse these algorithms mathematically and prove bounds on their regret. The moment we add states (i.e., the environment changes based on actions taken), the analysis becomes much harder. That's the subject of the rest of this book.

For now, observe that every algorithm encodes the same intuition: reward-seeking with uncertainty. The difference is in how precisely they quantify uncertainty and how aggressively they resolve it.

```q
/ Quick comparison summary
show ([]
  method:  `greedy`eps001`eps010`UCB`gradient;
  avgFinalReward: (
    last r000`avgRewards;
    last r010`avgRewards;
    last r100`avgRewards;
    last rUCB`avgRewards;
    last rGrad`avgRewards
  ))
```

The numbers won't surprise you, but running the code and watching them emerge will. That's the experiment.
