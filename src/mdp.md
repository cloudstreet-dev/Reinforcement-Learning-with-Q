# Markov Decision Processes: States, Actions, and Rewards in Q

*S&B Chapter 3*

Bandits were stateless: each pull was independent, and the world didn't change based on what you did. Real decision problems have *state*—the world is different after you act, and what you do now affects what's available later.

A **Markov Decision Process** (MDP) formalises this structure. At each time step \\(t\\):
- The agent observes state \\(S_t \in \mathcal{S}\\)
- Selects action \\(A_t \in \mathcal{A}(S_t)\\)
- Receives reward \\(R_{t+1} \in \mathbb{R}\\)
- Transitions to \\(S_{t+1}\\)

The *Markov property*: \\(P(S_{t+1}, R_{t+1} \mid S_t, A_t) = P(S_{t+1}, R_{t+1} \mid S_0, A_0, \ldots, S_t, A_t)\\). The future depends on the present, not the history. This is simultaneously a strong assumption and the thing that makes MDPs tractable.

## Representing an MDP in Q

An MDP needs states, actions, a transition function, and a reward function. In q, these map naturally:

- **States**: integers (indices into a state space)
- **Actions**: integers (indices into action sets)
- **Transition function**: a q function `(state; action) -> next_state`
- **Reward function**: a q function `(state; action; next_state) -> reward`
- **Terminal check**: a q function `state -> bool`

This is not the only representation, but it's clean and fast.

## The GridWorld Environment

S&B uses a 4×4 GridWorld (Figure 3.2, Chapter 4) throughout the early chapters. Let's build it once and use it everywhere.

```
 0  1  2  3
 4  5  6  7
 8  9 10 11
12 13 14 15
```

States 0 and 15 are terminal. Actions: 0=up, 1=right, 2=down, 3=left. Moving into a wall returns to the current state. Reward is -1 for every non-terminal transition.

```q
/ ============================================================
/ GridWorld MDP — S&B Figure 3.2 / 4.1
/ ============================================================

nRows:4; nCols:4
nStates:nRows*nCols
nActions:4  / 0=up 1=right 2=down 3=left

/ Terminal states
terminals:0 15

/ Action deltas: (row_delta; col_delta)
actionDeltas:((-1;0);(0;1);(1;0);(0;-1))  / up right down left

/ State -> (row; col)
stateToRC:{[s] (s div nCols; s mod nCols)}

/ (row; col) -> state (clamp to grid boundary)
rcToState:{[r;c]
  r2:0|r&nRows-1;
  c2:0|c&nCols-1;
  (nCols*r2)+c2
  }

/ Transition: (state; action) -> next_state
/ Terminal states absorb (stay put)
transition:{[s;a]
  if[s in terminals; :s];
  rc:stateToRC s;
  d:actionDeltas a;
  rcToState[rc[0]+d[0]; rc[1]+d[1]]
  }

/ Reward: -1 for all non-terminal transitions
reward:{[s;a;s2] -1}

/ Is state terminal?
isTerminal:{[s] s in terminals}
```

## The Value Function

The *state-value function* \\(v_\pi(s)\\) is the expected cumulative discounted return starting from state \\(s\\) and following policy \\(\pi\\):

\\[v_\pi(s) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \;\middle|\; S_t = s\right]\\]

The discount factor \\(\gamma \in [0,1)\\) makes future rewards worth less than immediate rewards. \\(\gamma = 0\\) means the agent is completely myopic; \\(\gamma = 1\\) means it weights all future rewards equally (fine for episodic tasks).

> **The Bellman equation** rewrites this recursive definition as a consistency constraint:
> \\[v_\pi(s) = \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)\left[r + \gamma v_\pi(s')\right]\\]
> This says: the value of a state is the expected immediate reward plus the discounted value of wherever you end up. It's not a formula for computing \\(v_\pi\\)—it's a *constraint* that \\(v_\pi\\) must satisfy. The algorithms in Chapter 4 find the \\(v_\pi\\) that satisfies it.

## Generating Episodes

Before we get to dynamic programming, we need to be able to interact with the environment—generate episodes by following a policy.

```q
/ Random equiprobable policy: choose uniformly among all actions
randomPolicy:{[s] first 1?nActions}

/ One environment step: (state; action) -> (next_state; reward; done)
envStep:{[s;a]
  s2:transition[s;a];
  r:reward[s;a;s2];
  done:isTerminal s2;
  `s`r`done!(s2;r;done)
  }

/ Generate one episode following policy pi from start state s0
/ Returns table of (s, a, r, s') transitions
generateEpisode:{[policy;s0;maxSteps]
  / State for scan: (current_state; done_flag; step_count)
  / We'll collect transitions manually using while
  s:s0;
  transitions:();
  step:0;
  while[(not isTerminal s) and step<maxSteps;
    a:policy[s];
    result:envStep[s;a];
    transitions,:enlist `s`a`r`s2!(s;a;result`r;result`s);
    s:result`s;
    step+:1
  ];
  `t$transitions     / cast list of dicts to table
  }

/ Example: generate a random episode from corner state 1
ep:generateEpisode[randomPolicy;1;1000];
count ep               / episode length
sum ep`r               / total reward (all -1s, so negative of length)
```

## Representing Policies

A deterministic policy is a function from states to actions. In q, this is just a list indexed by state:

```q
/ Deterministic policy: list where policy[s] = action
/ Equiprobable random policy as a function
makeRandomPolicy:{[] {first 1?nActions} each til nStates}

/ Deterministic policy from a value table: greedy w.r.t. V
/ For each state, choose action that maximises V(next_state)
greedyPolicy:{[V;gamma]
  / For each state and action, compute immediate reward + gamma*V(s')
  {[V;gamma;s]
    $[isTerminal s; 0i;
      imax {reward[s;x;transition[s;x]] + gamma*V[transition[s;x]]}
           each til nActions
    ]
  }[V;gamma;] each til nStates
  }
```

## Action Values

The *action-value function* \\(q_\pi(s,a)\\) is the expected return from taking action \\(a\\) in state \\(s\\), then following policy \\(\pi\\):

\\[q_\pi(s,a) = \mathbb{E}_\pi\left[G_t \;\middle|\; S_t=s, A_t=a\right]\\]

In tabular methods, we often represent \\(Q\\) as a 2D structure: states × actions. In q:

```q
/ Q-table: nStates x nActions matrix, initialised to 0
initQTable:{[] nStates#enlist nActions#0f}

/ Access: Q[s][a]
/ Update Q[s][a] by delta
updateQTable:{[Q;s;a;delta]
  row:@[Q[s];a;+;delta];
  @[Q;s;:;row]
  }
```

## The Optimal Policy

The *optimal value function* \\(v^*(s) = \max_\pi v_\pi(s)\\) is the best achievable value from each state. The *Bellman optimality equation*:

\\[v^*(s) = \max_a \sum_{s',r} p(s',r|s,a)\left[r + \gamma v^*(s')\right]\\]

This is nonlinear (the \\(\max\\) breaks linearity), so we can't solve it as a system of equations directly. The algorithms in the next chapter—dynamic programming—solve it iteratively. Monte Carlo and TD methods approximate it from sampled experience. The rest of this book is various approaches to finding \\(v^*\\) without knowing the full model \\(p(s',r|s,a)\\).

```q
/ Quick demonstration: run episodes under random policy
/ and observe the average return from each state
nEpisodes:1000
returns:(nStates#enlist 0#0f);     / empty float lists per state

{[ep]
  g:0f; gamma:1f;
  / Compute returns backwards from episode end
  traj:reverse ep;
  {[gamma;row]
    g::gamma*g+row`r;             / :: for global update inside lambda
    returns[row`s],:g;            / append return to state's list
    } [gamma;] each traj;
  } each generateEpisode[randomPolicy;;500] each 1+til nStates-2;

/ Average return by state: should be negative (cost to reach terminal)
avgReturn:avg each returns
show nCols cut avgReturn    / display as 4x4 grid
```

The grid display won't win any visualisation awards, but you'll see the pattern: states near the terminals (0 and 15) have higher (less negative) average returns. States far away, like state 5 and 10 in the centre, take longer to reach a terminal under random policy, so they accumulate more −1 rewards.

This is what dynamic programming computes exactly, without sampling. That's next.
