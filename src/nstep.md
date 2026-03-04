# N-Step Methods: Bridging Monte Carlo and TD

*S&B Chapter 7*

The TD(0) update uses one step of actual reward then bootstraps. Monte Carlo uses the complete return. Neither extreme is optimal for all problems. N-step methods generalise both: use \\(n\\) actual steps of reward then bootstrap from \\(V(S_{t+n})\\).

The \\(n\\)-step return:

\\[G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n})\\]

When \\(n=1\\): one-step TD. When \\(n=T\\) (episode length): Monte Carlo. Between these poles lies a continuum of algorithms, and the best value of \\(n\\) is problem-dependent—often somewhere in the middle.

> **Why this matters**: TD(0) bootstraps immediately, so errors in \\(V(S_{t+1})\\) pollute the TD target for \\(V(S_t)\\). With \\(n\\) steps, you dilute that bootstrapping error with \\(n\\) actual reward observations before hitting an estimate. MC eliminates bootstrapping error entirely but has high variance—the return \\(G_t\\) depends on \\(T - t\\) random transitions. N-step methods let you trade these off explicitly.

## N-Step TD Prediction

```q
/ ============================================================
/ N-Step TD — Random Walk (S&B Figure 7.2)
/ Same 7-state random walk as Chapter 6
/ ============================================================

pi:acos -1f
normal:{[] sqrt[-2f*log rand 1.0f]*cos[2f*pi*rand 1.0f]}

nRWStates:7
leftTerm:0; rightTerm:6
trueV:0 1 2 3 4 5 6 % 6f   / true values

rwStep:{[s]
  s2:s+$[rand[1.0f]<0.5f;-1;1];
  r:$[s2=rightTerm;1f;0f];
  `s`r`done!(s2;r;(s2=leftTerm) or s2=rightTerm)
  }

/ N-step TD prediction (S&B Algorithm, Chapter 7)
/ Uses a circular buffer of size n to store recent (s,r) pairs
nStepTD:{[n;alpha;gamma;nEpisodes]
  V:nRWStates#0.5f;
  V[leftTerm]:0f; V[rightTerm]:0f;
  rmse_hist:();

  do[nEpisodes;
    / Circular buffers for states and rewards
    states:n+1#0i;
    rewards:n+1#0f;
    states[0]:3i;   / start in centre
    T:0W;           / terminal time (infinity initially)
    t:0i;           / current time
    step:0i;        / loop counter (tau in S&B notation)

    / Run until all updates are done (t >= T + n - 1)
    while[(step-n)<T;
      if[step<T;
        result:rwStep[states[step mod (n+1)]];
        states[(step+1) mod (n+1)]:result`s;
        rewards[(step+1) mod (n+1)]:result`r;
        if[result`done; T:step+1i]
      ];
      / Update time tau: the state whose estimate we're updating
      tau:step-n+1i;
      if[tau>=0;
        / Compute n-step return
        end:tau+n;
        G:0f;
        k:min[end;T];
        while[k>tau;
          G:rewards[k mod (n+1)] + gamma*G;
          k-:1
        ];
        / Add bootstrap value if not at terminal
        if[end<T; G+:gamma xexp n * V[states[end mod (n+1)]]];
        / Update V[tau]
        s_tau:states[tau mod (n+1)];
        if[(not s_tau=leftTerm) and not s_tau=rightTerm;
          V[s_tau]+:alpha*(G - V[s_tau])
        ]
      ];
      step+:1
    ];
    rmse:sqrt avg (V[1 2 3 4 5]-trueV[1 2 3 4 5]) xexp 2;
    rmse_hist,:rmse
  ];
  `V`rmse!(V;rmse_hist)
  }
```

```q
/ Reproduce S&B Figure 7.2: RMS error vs n for various alpha
/ 10 runs, 10 episodes each, measured at episode 10
nVals:1 2 4 8 16 32 64 128 256 512
alphaVals:0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

/ For n=1 (TD) and n=512 (near-MC), compare
r_n1:nStepTD[1i;0.4f;1f;100];
r_n4:nStepTD[4i;0.4f;1f;100];
r_n16:nStepTD[16i;0.3f;1f;100];

last r_n1`rmse    / TD(0) final RMSE
last r_n4`rmse    / n=4 typically lowest error for appropriate alpha
last r_n16`rmse   / starts to behave like MC
```

The figure in S&B shows a U-shaped curve: very small \\(n\\) (pure TD) has moderate error because it bootstraps aggressively from poor estimates; very large \\(n\\) (near-MC) has moderate error from high variance. The sweet spot is often \\(n \approx 4\\) to \\(16\\), depending on the problem.

## N-Step SARSA

Extending to control is straightforward: collect \\(n\\) steps of \\((s, a, r)\\) tuples, then compute the \\(n\\)-step return and update \\(Q(s, a)\\):

\\[G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n Q(S_{t+n}, A_{t+n})\\]

```q
/ ============================================================
/ N-Step SARSA — Windy Gridworld
/ ============================================================

wgRows:7; wgCols:10
wgStart:30; wgGoal:37
windStrength:0 0 0 1 1 1 2 2 1 0
wgDeltas:((-1;0);(0;1);(1;0);(0;-1))

wgRCToState:{[r;c] ((0|(wgRows-1)&r)*wgCols)+(0|(wgCols-1)&c)}
wgStateToRC:{[s](s div wgCols;s mod wgCols)}

wgStep:{[s;a]
  rc:wgStateToRC s;
  d:wgDeltas[a];
  r2:rc[0]+d[0]-windStrength[rc[1]];
  c2:rc[1]+d[1];
  s2:wgRCToState[r2;c2];
  `s`r`done!(s2;-1f;s2=wgGoal)
  }

wgSelectAction:{[Q;eps;s]
  $[rand[1.0f]<eps; first 1?4; imax Q[s]]}

nStepSarsa:{[n;alpha;gamma;eps;nEpisodes]
  nWGStates:wgRows*wgCols;
  Q:nWGStates#enlist 4#0f;
  epLengths:();

  do[nEpisodes;
    bufSize:n+1;
    states:bufSize#wgStart;
    actions:bufSize#0i;
    rewards:bufSize#0f;
    actions[0]:wgSelectAction[Q;eps;wgStart];
    T:0W; step:0i;

    while[(step-n)<T;
      if[step<T;
        result:wgStep[states[step mod bufSize]; actions[step mod bufSize]];
        s2:result`s; r:result`r; done:result`done;
        states[(step+1) mod bufSize]:s2;
        rewards[(step+1) mod bufSize]:r;
        $[done;
          T:step+1i;
          actions[(step+1) mod bufSize]:wgSelectAction[Q;eps;s2]
        ]
      ];
      tau:step-n+1i;
      if[tau>=0;
        end:tau+n;
        G:0f;
        k:min[end;T];
        while[k>tau;
          G:rewards[k mod bufSize]+gamma*G;
          k-:1
        ];
        if[end<T;
          s_end:states[end mod bufSize];
          a_end:actions[end mod bufSize];
          G+:gamma xexp n * Q[s_end;a_end]
        ];
        s_tau:states[tau mod bufSize];
        a_tau:actions[tau mod bufSize];
        Q[s_tau;a_tau]+:alpha*(G - Q[s_tau;a_tau])
      ];
      step+:1
    ];
    epLengths,:step-n+1i  / approximate episode length
  ];
  `Q`epLengths!(Q;epLengths)
  }
```

```q
/ Compare n=1 (SARSA), n=4, n=8 on windy gridworld
r_sarsa1:nStepSarsa[1i;0.5f;1f;0.1f;200];
r_sarsa4:nStepSarsa[4i;0.5f;1f;0.1f;200];
r_sarsa8:nStepSarsa[8i;0.3f;1f;0.1f;200];

/ Convergence speed: cumulative steps to reach goal
last sums r_sarsa1`epLengths
last sums r_sarsa4`epLengths    / often converges faster
last sums r_sarsa8`epLengths
```

## TD(\\(\lambda\\)): The Elegance of Eligibility Traces

N-step methods choose a fixed \\(n\\). What if we used a *weighted combination* of all \\(n\\)-step returns, with weights decaying geometrically?

\\[G_t^\lambda = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_{t:t+n}\\]

This is the \\(\lambda\\)-return. When \\(\lambda=0\\): pure TD(0). When \\(\lambda=1\\): pure Monte Carlo. The parameter \\(\lambda\\) interpolates the full spectrum in a single algorithm.

The forward view (computing \\(G_t^\lambda\\) above) requires future rewards. The backward view, using **eligibility traces**, computes the same update online using only past information:

```q
/ TD(lambda) with eligibility traces — forward view approximate
/ (True online TD(lambda) requires the full algorithm from S&B Ch 12)
tdLambda:{[lam;alpha;gamma;nEpisodes]
  V:nRWStates#0.5f;
  V[leftTerm]:0f; V[rightTerm]:0f;
  rmse_hist:();

  do[nEpisodes;
    / Eligibility traces: one per state, initialised to 0 each episode
    e:nRWStates#0f;
    s:3i;  / start centre
    done:0b;

    while[not done;
      result:rwStep[s];
      s2:result`s; r:result`r; done:result`done;

      / TD error
      delta:r + gamma*$[done;0f;V[s2]] - V[s];

      / Update eligibility: accumulating traces
      e[s]+:1f;

      / Update all states proportional to their eligibility
      V+:alpha*delta*e;

      / Decay eligibility traces
      e*:gamma*lam;
      s:s2
    ];
    rmse:sqrt avg (V[1 2 3 4 5]-trueV[1 2 3 4 5]) xexp 2;
    rmse_hist,:rmse
  ];
  `V`rmse!(V;rmse_hist)
  }
```

```q
/ lambda=0: should match TD(0)
r_lam0:tdLambda[0f;0.4f;1f;100];

/ lambda=0.9: more like MC, slower on short problems
r_lam9:tdLambda[0.9f;0.1f;1f;100];

last r_lam0`rmse   / ~0.10
last r_lam9`rmse   / varies more

/ The vectorised update `V+:alpha*delta*e` is the key q idiom here:
/ every state's value moves by its eligibility-weighted TD error
/ This is where q's array operations shine
```

The eligibility trace update is a perfect example of where q shines. The update `V+:alpha*delta*e` is a single vectorised operation that, in Python, would require an explicit loop over all states. In q it's one line, it's fast, and it reads exactly like the mathematical expression \\(V \leftarrow V + \alpha \delta \mathbf{e}\\).

Eligibility traces generalise beautifully to function approximation (Chapter 8) and underpin many practical algorithms. For deep RL, they appear in advantage estimation (GAE) used by PPO. The underlying idea—that credit for a reward propagates backwards through recently visited states—never stops being useful.
