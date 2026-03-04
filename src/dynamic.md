# Dynamic Programming: Policy Evaluation and Iteration

*S&B Chapter 4*

Dynamic programming (DP) is the class of algorithms that compute optimal policies given a *perfect model* of the environment—the full transition and reward functions \\(p(s',r|s,a)\\). In practice, you rarely have this. DP is nonetheless essential because it defines what every other algorithm is trying to approximate.

The "programming" in dynamic programming is not computer programming. It refers to Bellman's 1950s usage meaning "planning" or "scheduling." The "dynamic" refers to the time dimension. Bellman was famously good at naming things.

We'll work entirely with the 4×4 GridWorld from the MDP chapter. Load that code first, or paste it at the top of your file.

## Iterative Policy Evaluation

Given a policy \\(\pi\\), compute \\(v_\pi\\). The algorithm applies the Bellman equation as an update rule, sweeping through all states repeatedly until convergence.

**Algorithm** (S&B p. 75):
1. Initialise \\(V(s) = 0\\) for all \\(s\\)
2. For each state \\(s\\), update:
   \\[V(s) \leftarrow \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)\left[r + \gamma V(s')\right]\\]
3. Repeat until \\(\max_s |V_{\text{new}}(s) - V_{\text{old}}(s)| < \theta\\)

For the equiprobable random policy (\\(\pi(a|s) = 1/4\\) for all \\(a\\)), each state's value is the average over all actions of the reward plus discounted successor value.

```q
/ ============================================================
/ Dynamic Programming — GridWorld (S&B Chapter 4)
/ Paste GridWorld definitions from mdp.md, or \l gridworld.q
/ ============================================================

/ Equiprobable random policy: uniform over all actions
equiprobable:nStates#enlist nActions#(1f%nActions)

/ Policy evaluation: compute V_pi for policy pi
/ pi: list of (nStates) probability distributions over actions
/ gamma: discount factor
/ theta: convergence threshold
policyEval:{[pi;gamma;theta]
  V:nStates#0f;
  delta:theta+1;   / ensure at least one sweep
  while[delta>theta;
    delta:0f;
    newV:{[V;gamma;pi;s]
      $[isTerminal s; 0f;
        sum pi[s] * {[V;gamma;s;a]
          s2:transition[s;a];
          (reward[s;a;s2]) + gamma*V[s2]
          }[V;gamma;s;] each til nActions
      ]
      }[V;gamma;pi;] each til nStates;
    delta:max abs newV-V;
    V:newV
  ];
  V
  }
```

Let's replicate S&B Figure 4.1—the value function for the random policy with \\(\gamma=1\\):

```q
/ Evaluate random policy with gamma=1
V_rand:policyEval[equiprobable; 1f; 0.001f]

/ Display as 4x4 grid (S&B Figure 4.1)
show nCols cut `int$V_rand
/ Expected output (approximately):
/  0 -14 -20 -22
/ -14 -18 -20 -20
/ -20 -20 -18 -14
/ -22 -20 -14   0
```

If your numbers match S&B's figure, the algorithm is correct. The corner states (1 and 14) need 14 steps on average to reach a terminal under random policy; states 5 and 10 are even worse at 18 steps. State 6 in the centre requires 20. These are exact results, not estimates.

> **The aha moment**: The Bellman equation isn't a formula for computing \\(v_\pi\\) — it's a *fixed-point equation*. The value function that satisfies it *is* \\(v_\pi\\). Iterative policy evaluation just repeatedly applies the Bellman update until the value function stops changing. The fact that this converges to the unique fixed point \\(v_\pi\\) is the contraction mapping theorem. You don't need to care about the theorem; you just need to run the loop.

## Policy Improvement

Given \\(v_\pi\\), we can improve the policy by acting *greedily* with respect to \\(v_\pi\\):

\\[\pi'(s) = \operatorname{argmax}_a \sum_{s',r} p(s',r|s,a)\left[r + \gamma v_\pi(s')\right]\\]

The *Policy Improvement Theorem* guarantees that \\(\pi' \geq \pi\\) (weakly better everywhere). If \\(\pi' = \pi\\), both are optimal.

```q
/ Policy improvement: given V, return deterministic greedy policy
/ Returns: list of actions (one per state)
policyImprove:{[V;gamma]
  {[V;gamma;s]
    $[isTerminal s;
      0i;    / terminal: action doesn't matter
      imax {reward[s;x;transition[s;x]] + gamma*V[transition[s;x]]}
           each til nActions
    ]
  }[V;gamma;] each til nStates
  }

/ Convert deterministic policy to probability distribution format
detToProbPi:{[detPi]
  {[a] @[nActions#0f;a;:;1f]} each detPi
  }
```

## Policy Iteration

Alternate between policy evaluation and policy improvement until the policy stabilises:

```q
/ Policy iteration
/ Returns: (optimal_V; optimal_policy)
policyIteration:{[gamma;theta]
  / Start with random deterministic policy
  pi:randomPolicy[]`int;                / nStates list of actions
  stable:0b;
  while[not stable;
    / Evaluate current policy
    probPi:detToProbPi pi;
    V:policyEval[probPi;gamma;theta];
    / Improve policy
    newPi:policyImprove[V;gamma];
    / Check stability
    stable:newPi~pi;
    pi:newPi
  ];
  (V;pi)
  }
```

```q
/ Run policy iteration on GridWorld
result:policyIteration[1f; 0.001f];
V_opt:result[0]; pi_opt:result[1];

/ Optimal values (S&B Figure 4.2 left)
show nCols cut `int$V_opt
/  0  -1  -2  -3
/ -1  -2  -3  -2
/ -2  -3  -2  -1
/ -3  -2  -1   0

/ Optimal policy (0=up 1=right 2=down 3=left)
show nCols cut pi_opt
/ 0 3 3 2
/ 0 0 0 2
/ 0 0 1 2
/ 0 1 1 0  (multiple optimal actions are valid; S&B shows arrows)
```

The optimal value function is beautiful: each state's value is exactly the negative of its Manhattan distance to the nearest terminal state. This makes sense—the optimal policy takes the shortest path, so the value is -(steps to terminal).

Policy iteration converges in a small number of iterations for GridWorld. The first iteration turns the random policy into something reasonable; subsequent iterations refine the edges. Track convergence:

```q
/ Policy iteration with iteration tracking
policyIterationVerbose:{[gamma;theta]
  pi:randomPolicy[]`int;
  iter:0;
  stable:0b;
  while[not stable;
    probPi:detToProbPi pi;
    V:policyEval[probPi;gamma;theta];
    newPi:policyImprove[V;gamma];
    stable:newPi~pi;
    pi:newPi;
    iter+:1;
    -1 "Iteration ",string[iter],": policy changed=",string not stable;
  ];
  (V;pi;iter)
  }
```

## Value Iteration

Policy iteration evaluates each intermediate policy to full convergence. Value iteration is more aggressive: take just *one* Bellman update per state before improving. This converges to \\(v^*\\) directly without ever computing an intermediate policy:

**Update rule** (S&B Equation 4.10):
\\[V(s) \leftarrow \max_a \sum_{s',r} p(s',r|s,a)\left[r + \gamma V(s')\right]\\]

```q
/ Value iteration: directly find v*
valueIteration:{[gamma;theta]
  V:nStates#0f;
  delta:theta+1;
  iters:0;
  while[delta>theta;
    newV:{[V;gamma;s]
      $[isTerminal s; 0f;
        max {reward[s;x;transition[s;x]] + gamma*V[transition[s;x]]}
            each til nActions
      ]
      }[V;gamma;] each til nStates;
    delta:max abs newV-V;
    V:newV;
    iters+:1
  ];
  / Extract greedy policy
  pi:policyImprove[V;gamma];
  (V;pi;iters)
  }
```

```q
/ Value iteration converges in fewer total Bellman updates than policy iteration
result_vi:valueIteration[1f; 0.001f]
result_vi[2]   / number of sweeps to convergence: ~4

/ Compare against policy iteration result
max abs result_vi[0]-V_opt   / should be < theta (0.001)
```

Value iteration converges in far fewer sweeps because it doesn't wait for full policy evaluation. The tradeoff: each sweep uses \\(\max\\) rather than a weighted average, which is slightly more expensive per state, but the total computation is typically much less.

## Asynchronous DP

The algorithms above sweep through all states in each iteration. *Asynchronous* DP updates states in any order—even one state at a time—and still converges given sufficient updates. This matters for large state spaces where full sweeps are expensive.

```q
/ Asynchronous value iteration: update one random state per step
asyncVI:{[gamma;nUpdates]
  V:nStates#0f;
  do[nUpdates;
    s:first 1?nStates;      / random state selection
    $[not isTerminal s;
      V[s]:max {reward[s;x;transition[s;x]] + gamma*V[transition[s;x]]}
                each til nActions;
      :[]                   / skip terminals
    ]
  ];
  V
  }

/ With enough updates, async converges to same solution
V_async:asyncVI[1f; 50000];
max abs V_async-V_opt   / small residual error
```

The ordering of updates matters for convergence speed. Updating states that are frequently visited, or states whose successors are already well-estimated, is more efficient. This idea generalises to *prioritised sweeping*, which uses a priority queue to always update the state with the largest estimated update. We won't implement it here, but the principle is simple: update where learning happens fastest.

## Complexity

For GridWorld (16 states, 4 actions), DP is instantaneous. For larger problems:

- Policy evaluation: \\(O(|\mathcal{S}|^2 |\mathcal{A}|)\\) per sweep
- Value iteration: same per sweep, fewer sweeps
- Real-world state spaces: millions to billions of states—DP is infeasible

This is why the remaining chapters exist. When you can't enumerate all states, you need to estimate values from sampled experience (Monte Carlo, TD) or approximate them with parameterised functions (Chapter 8). Everything builds on the DP foundation: we're always trying to compute or approximate \\(v^*\\) or \\(q^*\\); we just have different amounts of information and computation available.
