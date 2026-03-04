# Policy Gradient Methods: Directly Optimising What You Care About

*S&B Chapter 13*

Every method so far has learned a value function and derived a policy from it. Policy gradient methods are different: they parameterise the policy directly and optimise it directly, by gradient ascent on expected return.

This changes what you're learning. Instead of "what is the value of state \\(s\\)?", you're learning "what probability should I assign to action \\(a\\) in state \\(s\\)?". You optimise the policy's actual performance objective—no intermediary value function required.

Why would you want this?

1. **Stochastic policies**: Value-based methods produce deterministic policies (always take the greedy action). Some problems genuinely require stochastic policies—bluffing in poker, mixed strategies in games, exploration in environments where deterministic exploitation fails.

2. **Continuous actions**: Argmax over a continuous action space is expensive or impossible. Policy gradient methods can output a distribution over continuous actions directly (mean and variance of a Gaussian, for example).

3. **Convergence**: Policy gradient methods converge to local optima by construction. Value-based methods have more complex convergence properties under function approximation.

The cost: policy gradient methods typically have higher variance and require more samples than value-based methods for the same problem.

## The Policy Gradient Theorem

The performance objective is the expected return from the start state:

\\[J(\boldsymbol{\theta}) = v_{\pi_\theta}(s_0)\\]

We want \\(\nabla_\theta J(\theta)\\). The policy gradient theorem (S&B Equation 13.5) gives:

\\[\nabla J(\boldsymbol{\theta}) \propto \sum_s \mu(s) \sum_a q_\pi(s,a) \nabla \pi(a|s,\boldsymbol{\theta})\\]

where \\(\mu(s)\\) is the on-policy state distribution. The practical form, using the log-derivative trick:

\\[\nabla J(\boldsymbol{\theta}) = \mathbb{E}_\pi\left[G_t \nabla \ln \pi(A_t | S_t, \boldsymbol{\theta})\right]\\]

This is the REINFORCE estimator: for each step in an episode, the return \\(G_t\\) weights the gradient of the log-policy. High-return trajectories push the policy toward the actions that generated them.

## REINFORCE

```q
/ ============================================================
/ REINFORCE — Short Corridor Gridworld (S&B Example 13.1)
/ ============================================================
/ 4 states in a corridor; s1 and s3 appear identical to the agent
/ (both show "1 step from terminal"); only s1 is actually at
/ position 1, s3 at position 3.
/
/ The twist (S&B): from state 1, right goes left and left goes right.
/ The optimal policy must be stochastic: p(right|s1) = 0.59.
/ A deterministic policy cannot find the optimum.
/ ============================================================

pi_const:acos -1f
normal:{[] sqrt[-2f*log rand 1.0f]*cos[2f*pi_const*rand 1.0f]}

/ Corridor: states 0(terminal), 1, 2, 3; terminal at 0
/ Actions: 0=left, 1=right
/ Rewards: -1 per step; done when reaching state 0
/ Transitions: right at s1 goes left, left at s1 goes right

corridorStep:{[s;a]
  s2:$[s=1;
    $[a=1;0;2];   / state 1: actions are REVERSED
    $[a=1;s+1;s-1]
    ];
  s2:0|3&s2;           / clamp
  done:s2=0;
  `s`r`done!(s2;-1f;done)
  }

/ Policy: softmax over two actions, parameterised by theta
/ pi(right|s) = sigmoid(theta[s]) for each state
/ We use one parameter per non-terminal, non-confused state pair
/ S&B uses a single parameter theta (scalar) with x(s,right)=[1] x(s,left)=[0]
/ Let's follow S&B: scalar theta, feature h(s,a) = x(s,a)'theta

/ Feature function: h(s,right) = 1, h(s,left) = 0 for all states
/ (simplest linear softmax)
h:{[theta;s;a] $[a=1;theta;0f]}

/ Softmax probability
piProb:{[theta;s;a]
  ha:h[theta;s;a];
  hb:h[theta;s;1-a];
  exp[ha] % exp[ha]+exp[hb]
  }

/ Sample action from policy
piSample:{[theta;s]
  p:piProb[theta;s;1];   / prob of action 1 (right)
  $[rand[1.0f]<p;1;0]
  }

/ REINFORCE: policy gradient with Monte Carlo returns
/ S&B Algorithm 13.1
reinforce:{[alpha;gamma;nEpisodes]
  theta:0f;       / scalar parameter
  epReturns:();

  do[nEpisodes;
    / Generate episode
    s:3i;    / start at s3
    trajectory:();
    done:0b;
    while[(not done) and 1000>count trajectory;
      a:piSample[theta;s];
      result:corridorStep[s;a];
      trajectory,:enlist `s`a`r!(s;a;result`r);
      s:result`s;
      done:result`done
    ];

    / Compute returns G_t for each step (backwards)
    T:count trajectory;
    G:0f;
    k:T-1;
    while[k>=0;
      step:trajectory[k];
      G:step[`r]+gamma*G;
      / Policy gradient update
      / grad log pi(a|s,theta) = x(s,a) - E[x(s,*)] under pi
      / For scalar theta: grad = a - pi(right|s)
      / (since h=theta for a=1, h=0 for a=0)
      pRight:piProb[theta;step`s;1];
      gradLogPi:(`float$step`a) - pRight;   / a=1 or 0 minus prob of 1
      theta+:alpha*gamma xexp k * G*gradLogPi;
      k-:1
    ];
    epReturns,:sum trajectory`r
  ];
  `theta`returns!(theta;epReturns)
  }
```

```q
/ Run REINFORCE (S&B Figure 13.1)
/ alpha=2e-3 works well; results vary due to high variance
res_rf:reinforce[2e-3f;0.99f;1000];

/ Final theta: optimal is theta such that pi(right|s) ≈ 0.59
/ i.e. exp(theta)/(exp(theta)+1) = 0.59 => theta ≈ 0.36
show res_rf`theta
piProb[res_rf`theta;1;1]   / prob of going right from s1

/ Learning is noisy but should trend upward
avg last[100]res_rf`returns   / average episode return, last 100 eps
```

## REINFORCE with Baseline

REINFORCE has high variance because the return \\(G_t\\) can be large and volatile. Subtracting a **baseline** \\(b(s)\\) reduces variance without introducing bias (as long as the baseline doesn't depend on the action):

\\[\theta \leftarrow \theta + \alpha \left(G_t - b(S_t)\right) \nabla \ln \pi(A_t|S_t, \boldsymbol{\theta})\\]

A common choice: \\(b(s) = V(s)\\), the value function. This makes \\(G_t - b(S_t)\\) an estimate of the *advantage*: how much better was this action than average from this state?

```q
/ REINFORCE with baseline (learned state-value function)
/ Learns both theta (policy params) and w (baseline/value params)
reinforceBaseline:{[alphaTheta;alphaW;gamma;nEpisodes]
  theta:0f;          / policy parameter
  / Baseline: learned V(s) as function of state
  / Simple: one weight per state (tabular baseline)
  wBase:4#0f;        / one weight per corridor state
  epReturns:();

  do[nEpisodes;
    s:3i; trajectory:(); done:0b;
    while[(not done) and 1000>count trajectory;
      a:piSample[theta;s];
      result:corridorStep[s;a];
      trajectory,:enlist `s`a`r!(s;a;result`r);
      s:result`s;
      done:result`done
    ];

    / Compute returns and update both theta and baseline
    T:count trajectory;
    G:0f;
    k:T-1;
    while[k>=0;
      step:trajectory[k];
      G:step[`r]+gamma*G;
      st:step`s;

      / Baseline update: gradient descent on (G - wBase[s])^2
      delta:G - wBase[st];
      wBase[st]+:alphaW*delta;

      / Policy update: use advantage (G - baseline)
      pRight:piProb[theta;st;1];
      gradLogPi:(`float$step`a) - pRight;
      theta+:alphaTheta*gamma xexp k * delta*gradLogPi;
      k-:1
    ];
    epReturns,:sum trajectory`r
  ];
  `theta`wBase`returns!(theta;wBase;epReturns)
  }
```

```q
/ Compare REINFORCE vs REINFORCE with baseline (S&B Figure 13.2)
res_rf_base:reinforceBaseline[2e-3f;1e-2f;0.99f;1000];

/ Baseline version should show lower variance (smoother learning curve)
avg last[100]res_rf_base`returns        / similar final performance
avg abs deltas last[100]res_rf`returns  / variance of plain REINFORCE
avg abs deltas last[100]res_rf_base`returns  / lower variance

/ The baseline estimates: state 3 is weakest (furthest from goal)
res_rf_base`wBase
/ Expected pattern: wBase[0]=0 (terminal), wBase[3] < wBase[1] < wBase[2]
```

## Actor-Critic Methods

REINFORCE is a pure Monte Carlo method: it waits for the episode to end, computes full returns, then updates. Actor-Critic methods bootstrap: the "critic" estimates \\(V(s)\\) using TD updates; the "actor" updates policy parameters using the TD error as the advantage estimate.

```q
/ One-step Actor-Critic (S&B Algorithm 13.5)
actorCritic:{[alphaTheta;alphaW;gamma;nEpisodes]
  theta:0f;
  w:4#0f;   / tabular value function
  I:1f;     / gamma discount accumulator
  epReturns:();

  do[nEpisodes;
    s:3i; totalR:0f; done:0b;
    I:1f;

    while[(not done) and 10000>abs totalR;
      a:piSample[theta;s];
      result:corridorStep[s;a];
      s2:result`s; r:result`r; done:result`done;

      / TD error (advantage estimate)
      delta:r + gamma*$[done;0f;w[s2]] - w[s];

      / Critic update: one-step TD
      w[s]+:alphaW*delta;

      / Actor update: I * delta * grad_log_pi
      pRight:piProb[theta;s;1];
      gradLogPi:(`float$a) - pRight;
      theta+:alphaTheta*I*delta*gradLogPi;

      I*:gamma;     / discount accumulation
      totalR+:r;
      s:s2
    ];
    epReturns,:totalR
  ];
  `theta`w`returns!(theta;w;epReturns)
  }
```

```q
/ Actor-critic: online updates, no need to wait for episode end
res_ac:actorCritic[2e-3f;1e-2f;0.99f;1000];

/ Convergence comparison (rough—all three have high variance)
-3{avg last[100]x`returns}each (res_rf;res_rf_base;res_ac)
```

## The Policy Gradient Landscape

The methods in this chapter are the classical foundations. The modern policy gradient landscape has built extensively on them:

- **PPO (Proximal Policy Optimisation)**: clips the policy update to prevent large destructive steps. The most widely used algorithm in 2024.
- **A3C/A2C**: asynchronous actor-critic with parallel workers.
- **SAC (Soft Actor-Critic)**: adds entropy maximisation to the objective; naturally exploratory; very sample-efficient on continuous action spaces.
- **TRPO (Trust Region Policy Optimisation)**: uses a KL-divergence constraint to limit policy update size; theoretically cleaner than PPO but harder to implement.

All of these descend directly from REINFORCE and the policy gradient theorem. The variance problem is still there; the solutions are more sophisticated. The objective is the same: \\(\nabla J(\theta)\\). The rest is engineering.

> **The one thing to carry forward**: the policy gradient theorem is remarkably general. It says you can compute the gradient of *any* performance objective with respect to policy parameters by collecting trajectories and computing \\(G_t \nabla \ln \pi(A_t|S_t;\theta)\\). No model, no value function strictly required. Just samples, log-derivatives, and patience—which is not entirely unlike how organisms learn.
