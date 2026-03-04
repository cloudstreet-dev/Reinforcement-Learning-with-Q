# Function Approximation: When Your State Space Won't Fit in a Table

*S&B Chapters 9–10*

Every algorithm in this book so far has maintained a table: one entry per state, or one entry per state-action pair. GridWorld has 16 states. Blackjack has 200. Windy gridworld has 70. These fit in memory, run instantly, and converge to exact solutions.

Real problems don't cooperate. A chess position has roughly \\(10^{43}\\) legal states. A continuous robot joint angle lives in \\(\mathbb{R}^n\\). kdb+/q tick data has state spaces that combine position, inventory, market regime, and a dozen other features—functionally infinite.

Function approximation replaces the table with a parameterised function \\(\hat{v}(s; \mathbf{w}) \approx v_\pi(s)\\), where \\(\mathbf{w}\\) is a weight vector of manageable size. Updating one weight can affect estimates for many states simultaneously—generalisation. The cost: we lose exact convergence guarantees (mostly) and gain approximation error. In practice, for anything interesting, it's the only option.

## Linear Function Approximation

The simplest and most tractable case: \\(\hat{v}(s; \mathbf{w}) = \mathbf{w}^\top \mathbf{x}(s)\\), where \\(\mathbf{x}(s)\\) is a feature vector for state \\(s\\). The update rule under gradient descent:

\\[\mathbf{w} \leftarrow \mathbf{w} + \alpha \left[v_\pi(s) - \hat{v}(s;\mathbf{w})\right] \mathbf{x}(s)\\]

We don't know \\(v_\pi(s)\\), so we substitute a target: for semi-gradient TD(0), the target is \\(R + \gamma \hat{v}(S';\mathbf{w})\\).

\\[\mathbf{w} \leftarrow \mathbf{w} + \alpha \left[R + \gamma \hat{v}(S';\mathbf{w}) - \hat{v}(S;\mathbf{w})\right] \mathbf{x}(S)\\]

"Semi-gradient" because we treat the target \\(R + \gamma \hat{v}(S';\mathbf{w})\\) as fixed (not differentiating through it). This is the same trick deep Q-networks use with a "target network."

## State Aggregation

The simplest feature representation: group states into clusters, and use a one-hot vector indicating which cluster \\(s\\) belongs to. States in the same group share a single weight. It's coarse approximation, but it scales.

We'll use the **1000-state random walk** (S&B Example 9.1): states 1–1000, uniform transitions \\(\pm 100\\), rewards \\(-1\\) at the left terminal and \\(+1\\) at the right. Aggregate into 10 groups of 100 states each.

```q
/ ============================================================
/ State Aggregation + Semi-Gradient TD — 1000-State Random Walk
/ S&B Example 9.1
/ ============================================================

nStates1k:1002    / 0 and 1001 are terminals; 1-1000 are real states
leftTerm1k:0; rightTerm1k:1001

/ Random walk: jump uniform in [-100, 100], clamp to [0,1001]
rw1kStep:{[s]
  delta:first[1?201]-100;  / uniform in {-100,...,100}
  s2:0|1001&s+delta;
  r:$[s2=rightTerm1k;1f;$[s2=leftTerm1k;-1f;0f]];
  done:(s2=leftTerm1k) or s2=rightTerm1k;
  `s`r`done!(s2;r;done)
  }

/ State aggregation: 10 groups of 100 states
nGroups:10; groupSize:100

/ Feature vector: one-hot over groups (length nGroups)
stateFeature:{[s]
  $[(s=leftTerm1k) or s=rightTerm1k;
    nGroups#0f;     / terminal: zero features
    [g:(s-1) div groupSize;    / group index 0-9
     @[nGroups#0f;g;:;1f]]    / one-hot
  ]}

/ Value estimate: w . x(s)
vHat:{[w;s] w dot stateFeature s}

/ Semi-gradient TD(0) with linear function approximation
semiGradTD:{[alpha;gamma;nEpisodes]
  w:nGroups#0f;   / weight vector
  rmse_hist:();
  do[nEpisodes;
    s:500i;     / start near centre
    done:0b;
    while[not done;
      x:stateFeature s;
      result:rw1kStep[s];
      s2:result`s; r:result`r; done:result`done;
      / TD target
      target:r + gamma*$[done;0f;vHat[w;s2]];
      / Semi-gradient update: w += alpha * delta * x(s)
      delta:target - vHat[w;s];
      w+:alpha*delta*x
    ];
    / RMSE against true values (linear from -1 to 1)
    trueV_1k:{-1f + (2f*x%1001)} each 1+til 1000;
    estV_1k:{vHat[w;x]} each 1+til 1000;
    rmse_hist,:sqrt avg (estV_1k-trueV_1k) xexp 2
  ];
  `w`rmse!(w;rmse_hist)
  }
```

```q
/ Run semi-gradient TD with state aggregation
res_sgTD:semiGradTD[2e-4f;1f;100000];

/ Value estimates per group (S&B Figure 9.1)
/ Each weight corresponds to one group's value estimate
show res_sgTD`w
/ Should be approximately: -0.9 -0.7 -0.5 -0.3 -0.1 0.1 0.3 0.5 0.7 0.9
/ (staircase approximation to the linear true value)
```

The staircase is the tell: state aggregation can only represent piecewise constant value functions. Each group's value is its weight; states within a group are treated identically. This is a strong assumption that's often wrong in practice, but it's mathematically clean and a useful first step.

## Polynomial and Fourier Features

Beyond one-hot encoding, we can use richer feature functions. Polynomial features: \\(\mathbf{x}(s) = (1, s, s^2, \ldots, s^k)\\). Fourier features: \\(x_i(s) = \cos(i\pi s)\\). Both expand the state into a basis that linear function approximation can use.

```q
/ Fourier basis features for the 1000-state walk
/ State normalised to [0,1]
fourierFeatures:{[order;s]
  sn:`float$s % 1001f;   / normalise to [0,1]
  cos each pi*sn*til order+1   / cos(0), cos(pi*s), cos(2*pi*s), ...
  }

/ Semi-gradient TD with Fourier features
semiGradTDFourier:{[order;alpha;gamma;nEpisodes]
  nFeats:order+1;
  w:nFeats#0f;
  do[nEpisodes;
    s:500i; done:0b;
    while[not done;
      x:fourierFeatures[order;s];
      result:rw1kStep[s];
      s2:result`s; r:result`r; done:result`done;
      target:r + gamma*$[done;0f;w dot fourierFeatures[order;s2]];
      delta:target - w dot x;
      w+:alpha*delta*x
    ]
  ];
  w
  }

/ Fourier order 5: 6 parameters to represent 1000 states
w_fourier:semiGradTDFourier[5i;5e-5f;1f;100000];

/ Plot estimated values (evaluate at each of 1000 states)
vEst:{[w;order;s] w dot fourierFeatures[order;s]}
show 50 cut {vEst[w_fourier;5i;x]} each 1+til 1000
/ Should show a smooth increasing curve from -1 to 1
```

Fourier features have a useful property: the weight for each component has a direct interpretation as the magnitude of a frequency component in the value function. Low-frequency components (\\(i\\) small) capture broad trends; high-frequency components capture fine structure. This is analogous to spectral decomposition.

## Tile Coding

The practically most successful linear feature representation for RL (before neural networks took over) is **tile coding**: multiple overlapping tilings of the state space, each tiling a regular grid. For a 2D state space, imagine laying down multiple grids, each offset by a fraction of a tile width.

```q
/ Tile coding for 1D state space [low, high]
/ nTilings: number of overlapping grids
/ nTiles: number of tiles per tiling
tileCoding:{[nTilings;nTiles;low;high;s]
  / Total features: nTilings * nTiles
  width:(high-low)%nTiles;
  / Each tiling is offset by width/nTilings from the previous
  offset:width % nTilings;
  / For each tiling, find which tile s falls into
  features:nTilings#0i;
  k:til nTilings;
  / Shift s by k*offset, then find tile index
  shifted:s - k*offset;
  tileIdx:`int$(shifted-low)%width;
  tileIdx:0|nTiles-1&tileIdx;  / clamp to valid range
  / One-hot index into flattened feature vector
  base:nTiles*k;
  base+tileIdx
  }

/ Build full binary feature vector from tile indices
tilesToVector:{[nTilings;nTiles;indices]
  v:nTilings*nTiles#0f;
  v[indices]:1f;
  v
  }

/ Tile-coded linear approximation for Mountain Car
/ State: (position, velocity) in [-1.2,0.6] x [-0.07,0.07]
nMCTilings:8; nMCTiles:8

/ Tile code a (position, velocity) pair
mcFeatures:{[pos;vel]
  posIdx:tileCoding[nMCTilings;nMCTiles;-1.2f;0.6f;pos];
  velIdx:tileCoding[nMCTilings;nMCTiles;-0.07f;0.07f;vel];
  / Combine: offset velocity tiles by nTiles to avoid collision
  combined:posIdx,velIdx+nMCTilings*nMCTiles;
  tilesToVector[nMCTilings;nMCTiles*2;combined]
  }

nMCFeats:nMCTilings*nMCTiles*2

/ Demonstrate: two similar positions should have overlapping features
x1:mcFeatures[-0.5f;0.01f];
x2:mcFeatures[-0.5f;0.015f];
sum x1 and x2   / number of shared active tiles (out of 8 tilings)
```

## The Mountain Car Environment

Mountain Car (S&B Example 10.1): a car must drive up a steep hill, but the engine is too weak to climb directly. It must first drive left to build momentum, then accelerate right. Continuous state (position, velocity), 3 discrete actions (push left, neutral, push right), reward -1 per step.

```q
/ ============================================================
/ Mountain Car Environment — S&B Section 10.1
/ ============================================================

mcMinPos:-1.2f; mcMaxPos:0.6f
mcMinVel:-0.07f; mcMaxVel:0.07f
mcGoalPos:0.5f

/ Actions: 0=reverse, 1=neutral, 2=forward
mcActionForce:-0.001 0f 0.001f

/ Step function: returns (next_pos; next_vel; reward; done)
mcStep:{[pos;vel;a]
  force:mcActionForce[a];
  / Update velocity: add force and gravity component
  vel2:mcMinVel|(mcMaxVel&vel+force+(-0.0025f*cos[3f*pos]));
  / Update position
  pos2:mcMinPos|(mcMaxPos&pos+vel2);
  / If hit left wall, zero velocity
  vel2:$[pos2=mcMinPos;0f;vel2];
  done:pos2>=mcGoalPos;
  `pos`vel`r`done!(pos2;vel2;-1f;done)
  }

/ Semi-gradient SARSA for Mountain Car
/ Uses tile-coded linear Q-function approximation
mcSarsa:{[alpha;gamma;eps;nEpisodes]
  / Weight vector: nFeats * nActions
  W:nMCFeats*3#0f;   / flat: W[a*nMCFeats + i] = weight for action a, feature i

  / Approximate Q(s,a) = w_a . x(s)
  qHat:{[W;pos;vel;a]
    x:mcFeatures[pos;vel];
    wA:W[a*nMCFeats + til nMCFeats];
    wA dot x
    };

  / Epsilon-greedy over 3 actions
  selectMC:{[W;eps;pos;vel]
    $[rand[1.0f]<eps;
      first 1?3;
      imax {qHat[W;pos;vel;x]} each 0 1 2
    ]};

  epSteps:();
  do[nEpisodes;
    pos:mcMinPos+rand[mcMaxPos-mcMinPos];  / random start position
    vel:0f;                                 / zero initial velocity
    a:selectMC[W;eps;pos;vel];
    steps:0;
    done:0b;
    while[(not done) and steps<5000;        / max 5000 steps per episode
      x:mcFeatures[pos;vel];
      result:mcStep[pos;vel;a];
      pos2:result`pos; vel2:result`vel;
      r:result`r; done:result`done;
      a2:selectMC[W;eps;pos2;vel2];
      / Semi-gradient SARSA update
      target:r + gamma*$[done;0f;qHat[W;pos2;vel2;a2]];
      delta:target - qHat[W;pos;vel;a];
      / Update weights for action a only
      base:a*nMCFeats;
      W[base+til nMCFeats]+:alpha*delta*x;
      pos:pos2; vel:vel2; a:a2; steps+:1
    ];
    epSteps,:steps
  ];
  `W`epSteps!(W;epSteps)
  }
```

```q
/ Train: convergence is slow—this is a harder problem
/ alpha per-tiling: S&B uses alpha/8 where 8 = nTilings
res_mc:mcSarsa[0.5f%nMCTilings;1f;0.0f;500];  / eps=0 (greedy)

/ Episode lengths should decrease from ~5000 to ~100-200
res_mc[`epSteps][0]     / first episode: usually max steps (5000)
last res_mc`epSteps     / last episode: ~100-200 if converged

/ Show learning curve
show 50 cut res_mc`epSteps   / display as 10x50 grid for readability
```

## Why Function Approximation Changes Everything

With tabular methods, every update is exact and local. Changing \\(V(s)\\) for state \\(s\\) affects only that state. With function approximation, updating weights \\(\mathbf{w}\\) changes the estimates for *all* states that share features—which is most of them.

This generalisation is the whole point: we're making the algorithm smarter about states it hasn't seen. It's also what makes convergence analysis hard. Semi-gradient TD doesn't necessarily converge to the global optimum of any loss function; it converges to a *TD fixed point* that trades off Bellman error and function approximation error.

The practical implications: start with the simplest feature representation that captures the structure you need. Tile coding is remarkably effective and well-understood. Neural networks are powerful but introduce non-convex optimisation, instability, and the need for replay buffers and target networks. If tile coding solves your problem, use tile coding. If it doesn't—that's what deep RL is for.
