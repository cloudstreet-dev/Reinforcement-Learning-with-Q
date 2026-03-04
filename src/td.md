# Temporal Difference Learning: TD(0), SARSA, and Q-Learning

*S&B Chapter 6*

If Monte Carlo methods are one end of the spectrum and dynamic programming is the other, temporal difference learning occupies the middle ground—and it's arguably the most important ground in all of reinforcement learning. TD methods learn from experience (like MC) but also bootstrap from estimates (like DP). They update on every step, not at episode end. They work online, in real time, without a model.

This is the chapter where the field's two canonical algorithms live: SARSA and Q-Learning. One of them learns the policy you're executing. The other learns the optimal policy regardless of what you're doing. The difference is subtle, consequential, and completely obvious in hindsight.

## TD(0): The One-Step Update

MC waits until the end of the episode to compute the return \\(G_t\\) and update \\(V(S_t)\\). TD(0) doesn't wait. After one step, it knows \\(R_{t+1}\\) and \\(V(S_{t+1})\\), and uses their combination as an estimate of the true return:

\\[V(S_t) \leftarrow V(S_t) + \alpha\left[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)\right]\\]

The quantity in brackets is the **TD error** \\(\delta_t\\): the difference between what you expected (\\(V(S_t)\\)) and what you got (\\(R_{t+1} + \gamma V(S_{t+1})\\)). Update your estimate by \\(\alpha\\) times this error.

> **The key insight**: \\(R_{t+1} + \gamma V(S_{t+1})\\) is the *TD target*—your new estimate of what \\(V(S_t)\\) should be. You're not using the actual return; you're using a one-step lookahead combined with your current estimate of \\(V(S_{t+1})\\). This is *bootstrapping*: using your own estimates to update your own estimates. It introduces bias (because \\(V(S_{t+1})\\) may be wrong), but dramatically reduces variance. The tradeoff between bias and variance is one of the deepest topics in RL.

We'll demonstrate TD(0) on the Random Walk (S&B Figure 6.2): 5 states A–E with a terminal state at each end. The left terminal gives reward 0; the right terminal gives reward 1. All transitions otherwise give reward 0.

```q
/ ============================================================
/ TD(0) Prediction — Random Walk (S&B Example 6.2)
/ ============================================================

/ States: 0=left_terminal, 1=A, 2=B, 3=C, 4=D, 5=E, 6=right_terminal
nRWStates:7
leftTerm:0; rightTerm:6

/ Random walk: uniform left/right at each step
rwStep:{[s]
  direction:$[rand[1.0f]<0.5f; -1; 1];
  s2:s+direction;
  reward:$[s2=rightTerm;1f;0f];
  done:(s2=leftTerm) or s2=rightTerm;
  `s`r`done!(s2;reward;done)
  }

/ True values under random policy: v(s) = s/6
/ (probability of reaching right terminal from state s)
trueV:7#0f;
trueV[1]:1f%6; trueV[2]:2f%6; trueV[3]:3f%6;
trueV[4]:4f%6; trueV[5]:5f%6;

/ TD(0) prediction
/ Runs nEpisodes episodes, returns V estimate after each batch
td0:{[alpha;gamma;nEpisodes]
  V:nRWStates#0f;          / initialise to 0 (true: 0.5)
  V[3]:0.5f;               / S&B uses V=0.5 for non-terminal init
  V[1]:0.5f; V[2]:0.5f; V[4]:0.5f; V[5]:0.5f;
  rmse_hist:();
  do[nEpisodes;
    s:3;   / start in centre (state C)
    done:0b;
    while[not done;
      result:rwStep[s];
      s2:result`s; r:result`r; done:result`done;
      / TD update
      target:r + gamma*$[done;0f;V[s2]];
      V[s]+:alpha*(target - V[s]);
      s:s2
    ];
    / Track RMSE against true values (non-terminal states only)
    rmse:sqrt avg (V[1 2 3 4 5]-trueV[1 2 3 4 5])xexp 2;
    rmse_hist,:rmse
  ];
  `V`rmse!(V;rmse_hist)
  }
```

```q
/ Compare alpha values (S&B Figure 6.2)
r_a01:td0[0.05f;1f;100];
r_a02:td0[0.1f;1f;100];
r_a05:td0[0.15f;1f;100];

/ Final RMSE at episode 100
r_a01[`rmse][99]   / ~0.14
r_a02[`rmse][99]   / ~0.09  (typically best)
r_a05[`rmse][99]   / ~0.12  (converges fast, noisy)

/ Value estimates after 100 episodes
r_a02`V   / should be close to 1/6, 2/6, 3/6, 4/6, 5/6
```

## SARSA: On-Policy TD Control

For control (finding a good policy), we estimate action values \\(Q(s,a)\\) instead of state values. The on-policy update uses the *next* (state, action) pair \\((S', A')\\) chosen by the current policy—hence the name **SARSA**: \\(S, A, R, S', A'\\).

\\[Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\left[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)\right]\\]

We'll implement SARSA on the **Windy Gridworld** (S&B Example 6.5): a 7×10 grid where the wind in certain columns pushes the agent upward. Start at (3,0), goal at (3,7).

```q
/ ============================================================
/ SARSA — Windy Gridworld (S&B Example 6.5)
/ ============================================================

wgRows:7; wgCols:10
wgStart:30    / row 3, col 0: state = 3*10 + 0
wgGoal:37     / row 3, col 7

/ Wind strength per column (upward push)
windStrength:0 0 0 1 1 1 2 2 1 0

/ State encoding
wgStateToRC:{[s] (s div wgCols; s mod wgCols)}
wgRCToState:{[r;c]
  r2:0|(wgRows-1)&r;
  c2:0|(wgCols-1)&c;
  (wgCols*r2)+c2
  }

/ Action deltas: 0=up 1=right 2=down 3=left
wgDeltas:((-1;0);(0;1);(1;0);(0;-1))

/ Windy gridworld step
wgStep:{[s;a]
  rc:wgStateToRC s;
  r:rc[0]; c:rc[1];
  d:wgDeltas[a];
  wind:windStrength[c];
  / Apply action then wind (wind moves agent up = decreases row)
  r2:r+d[0]-wind;
  c2:c+d[1];
  s2:wgRCToState[r2;c2];
  done:s2=wgGoal;
  `s`r`done!(s2;-1f;done)   / reward -1 per step
  }

/ Epsilon-greedy action from Q-table
wgSelectAction:{[Q;eps;s]
  $[rand[1.0f]<eps;
    first 1?4;
    imax Q[s]
  ]}

/ SARSA on-policy control
sarsa:{[alpha;gamma;eps;nEpisodes]
  nWGStates:wgRows*wgCols;
  Q:nWGStates#enlist 4#0f;
  epLengths:();
  do[nEpisodes;
    s:wgStart;
    a:wgSelectAction[Q;eps;s];
    steps:0;
    done:0b;
    while[not done;
      result:wgStep[s;a];
      s2:result`s; r:result`r; done:result`done;
      a2:wgSelectAction[Q;eps;s2];
      / SARSA update: uses next action A' from current policy
      target:r + gamma*$[done;0f;Q[s2;a2]];
      Q[s;a]+:alpha*(target - Q[s;a]);
      s:s2; a:a2;
      steps+:1
    ];
    epLengths,:steps
  ];
  `Q`epLengths!(Q;epLengths)
  }
```

```q
/ Run SARSA on windy gridworld
result_sarsa:sarsa[0.5f;1f;0.1f;170];

/ Cumulative time steps to reach goal (S&B Figure 6.3)
cumSteps:sums result_sarsa`epLengths;
/ After ~170 episodes the optimal path is ~15 steps
last result_sarsa`epLengths   / should converge toward 15-17 steps
```

## Q-Learning: Off-Policy TD Control

Q-Learning is SARSA's off-policy counterpart. Instead of using the action actually taken next (\\(A'\\)), the update uses the *greedy* action—the maximum over all possible next actions:

\\[Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\left[R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t)\right]\\]

This makes Q-Learning *off-policy*: it learns the optimal Q-function \\(q^*\\) regardless of the behaviour policy used to generate data. The behaviour can be exploratory (epsilon-greedy), but Q-learning converges to the optimal, purely greedy policy.

We'll compare SARSA and Q-Learning on the **Cliff Walking** task (S&B Example 6.6): a 4×12 grid with a cliff along the bottom edge. Falling off the cliff costs -100 and returns to start. Start at bottom-left; goal at bottom-right.

```q
/ ============================================================
/ Cliff Walking — SARSA vs Q-Learning (S&B Example 6.6)
/ ============================================================

cwRows:4; cwCols:12
cwStart:(cwRows-1)*cwCols          / bottom-left
cwGoal:(cwRows-1)*cwCols+cwCols-1  / bottom-right
/ Cliff: bottom row, columns 1-10 (not start or goal)
cwCliff:(cwRows-1)*cwCols+1+til 10

cwStateToRC:{[s](s div cwCols;s mod cwCols)}
cwRCToState:{[r;c] r*cwCols+c}
cwDeltas:((-1;0);(0;1);(1;0);(0;-1))

cwStep:{[s;a]
  rc:cwStateToRC s;
  d:cwDeltas[a];
  r2:0|(cwRows-1)&rc[0]+d[0];
  c2:0|(cwCols-1)&rc[1]+d[1];
  s2:cwRCToState[r2;c2];
  $[s2 in cwCliff;
    `s`r`done!(cwStart;-100f;0b);    / cliff: reset to start
    s2=cwGoal;
    `s`r`done!(s2;-1f;1b);           / goal reached
    `s`r`done!(s2;-1f;0b)            / normal step
  ]}

cwSelectAction:{[Q;eps;s]
  $[rand[1.0f]<eps; first 1?4; imax Q[s]]}

/ Q-Learning
qLearning:{[alpha;gamma;eps;nEpisodes]
  nCWStates:cwRows*cwCols;
  Q:nCWStates#enlist 4#0f;
  epRewards:();
  do[nEpisodes;
    s:cwStart;
    totalR:0f;
    done:0b;
    while[not done;
      a:cwSelectAction[Q;eps;s];
      result:cwStep[s;a];
      s2:result`s; r:result`r; done:result`done;
      / Q-Learning: uses max Q(s',a') NOT the actual next action
      target:r + gamma*$[done;0f;max Q[s2]];
      Q[s;a]+:alpha*(target - Q[s;a]);
      s:s2;
      totalR+:r
    ];
    epRewards,:totalR
  ];
  `Q`epRewards!(Q;epRewards)
  }

/ SARSA on cliff walking
sarsaCW:{[alpha;gamma;eps;nEpisodes]
  nCWStates:cwRows*cwCols;
  Q:nCWStates#enlist 4#0f;
  epRewards:();
  do[nEpisodes;
    s:cwStart;
    a:cwSelectAction[Q;eps;s];
    totalR:0f; done:0b;
    while[not done;
      result:cwStep[s;a];
      s2:result`s; r:result`r; done:result`done;
      a2:cwSelectAction[Q;eps;s2];
      target:r + gamma*$[done;0f;Q[s2;a2]];
      Q[s;a]+:alpha*(target - Q[s;a]);
      s:s2; a:a2; totalR+:r
    ];
    epRewards,:totalR
  ];
  `Q`epRewards!(Q;epRewards)
  }
```

```q
/ Run both on cliff walking (S&B Figure 6.4)
res_ql:qLearning[0.5f;1f;0.1f;500];
res_sarsa:sarsaCW[0.5f;1f;0.1f;500];

/ Average reward over last 100 episodes
avg neg[100]#res_ql`epRewards     / Q-Learning: ~-13 (optimal path)
avg neg[100]#res_sarsa`epRewards  / SARSA: ~-17  (safer path)
```

This result captures the SARSA vs Q-Learning distinction precisely. Q-Learning learns the optimal policy (hug the cliff edge—shortest path) but *during training* it keeps falling off because epsilon-greedy exploration takes it cliff-ward. SARSA learns to account for its own exploration and finds a safer path further from the cliff. Q-Learning is optimal in the limit; SARSA is safer in practice.

The "correct" algorithm depends on whether you care about what's optimal given perfect execution, or what's good accounting for your actual (exploratory) behaviour. In production systems—where exploration happens on live data—on-policy methods like SARSA are often preferred precisely because they don't pretend they'll always execute optimally.

## Expected SARSA

A simple but effective improvement: instead of sampling \\(A'\\) from the policy (SARSA) or using \\(\max_{a'}\\) (Q-Learning), use the *expected* value under the policy:

\\[Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\left[R_{t+1} + \gamma \sum_{a'} \pi(a'|S_{t+1}) Q(S_{t+1}, a') - Q(S_t, A_t)\right]\\]

```q
/ Expected SARSA: lower variance than SARSA, handles off-policy too
expectedSarsa:{[alpha;gamma;eps;nEpisodes]
  nCWStates:cwRows*cwCols;
  Q:nCWStates#enlist 4#0f;
  epRewards:();
  do[nEpisodes;
    s:cwStart; totalR:0f; done:0b;
    while[not done;
      a:cwSelectAction[Q;eps;s];
      result:cwStep[s;a];
      s2:result`s; r:result`r; done:result`done;
      / Expected value under epsilon-greedy policy
      / greedyA gets (1-eps + eps/nA), others get eps/nA
      nA:4;
      pi:nA#eps%nA;
      pi[imax Q[s2]]+:1f-eps;
      expQ:sum pi*Q[s2];
      target:r + gamma*$[done;0f;expQ];
      Q[s;a]+:alpha*(target - Q[s;a]);
      s:s2; totalR+:r
    ];
    epRewards,:totalR
  ];
  `Q`epRewards!(Q;epRewards)
  }
```

```q
res_esarsa:expectedSarsa[0.5f;1f;0.1f;500];
avg neg[100]#res_esarsa`epRewards   / typically between SARSA and Q-Learning
```

Expected SARSA generalises both SARSA (if you use the current policy) and Q-Learning (if you use a greedy target policy). It has lower variance than SARSA because it averages out the randomness of \\(A'\\). In practice, it often outperforms both when computational cost per step isn't a concern.

## Why TD Methods Dominate

TD methods are the workhorses of practical RL for three reasons:

1. **Online learning**: update at every step, not episode end. Essential for long or continuous tasks.
2. **No model needed**: unlike DP, we sample transitions from the environment.
3. **Lower variance than MC**: bootstrapping stabilises estimates, at the cost of some bias.

The bias from bootstrapping never fully goes away—Q-Learning converges to \\(q^*\\) only under appropriate conditions on learning rates and infinite exploration. But in practice, with reasonable hyperparameters, it works extraordinarily well. The rest of this book explores what happens when you need more: longer lookaheads (Chapter 7), approximate value functions (Chapter 8), or direct policy optimisation (Chapter 9).
