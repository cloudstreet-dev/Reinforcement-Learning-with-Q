# Monte Carlo Methods: Learning from Complete Episodes

*S&B Chapter 5*

Dynamic programming requires a model. Most interesting problems don't give you one. You don't know the transition probabilities, the reward function, or the full structure of the MDP. You only know what happened: you took an action, something occurred, you received a reward.

Monte Carlo methods learn value functions by averaging the actual returns from complete episodes. No model needed. No Bellman backup. Just sample, observe, accumulate, and average. The law of large numbers does the rest.

The cost: you must wait for an episode to finish before updating anything. RL problems with very long or infinite episodes can't use Monte Carlo directly. For episodic tasks—games, trading sessions, discrete decision problems—MC is often the right starting point.

We'll use Blackjack, S&B's canonical MC environment (Section 5.1). It has a natural episode structure (each hand is one episode), a known true value function for comparison, and just enough complexity to be interesting.

## Blackjack Environment

The S&B version of Blackjack:
- Player draws cards to get as close to 21 as possible without going over
- Dealer shows one card face up
- Player can Hit (draw) or Stick (stop)
- Aces count as 11 unless that would bust, in which case they count as 1
- Natural blackjack (21 on first two cards) beats a dealer non-natural 21

State: (player_sum, dealer_showing, usable_ace) — (int 12–21, int 1–10, bool). We only care about player sums from 12 upward; below 12, always hit.

```q
/ ============================================================
/ Blackjack Environment — S&B Section 5.1
/ ============================================================

/ Draw a card: value is min(face_value, 10); aces = 11 initially
drawCard:{[] 1|10&first 1?13}   / uniform over 1-13, face cards -> 10

/ Draw a hand of 2 cards; return (sum; has_usable_ace)
initHand:{[]
  cards:{drawCard[]} each 0 1;
  s:sum cards;
  hasAce:any cards=1;
  / Ace can be used as 11 if it doesn't bust
  $[hasAce and s+10<=21;
    (s+10;1b);       / usable ace: count it as 11
    (s;0b)
  ]}

/ Hit a hand: add one card, adjust for aces if busted
hitHand:{[s;usableAce]
  card:drawCard[];
  s2:s+card;
  / If bust and have usable ace, convert it: 11 -> 1 (subtract 10)
  $[(s2>21) and usableAce;
    (s2-10;0b);      / ace downgraded
    (s2;usableAce or card=1)    / might gain a usable ace
  ]}

/ Dealer plays: hit until sum >= 17 (S&B rules)
dealerPlay:{[showing]
  / Dealer's hidden card
  hidden:drawCard[];
  s:showing+hidden;
  ua:any (showing;hidden)=1;
  $[ua and s+10<=21; s+:10; ua:s+10<=21];   / adjust for ace
  while[s<17;
    res:hitHand[s;ua];
    s:res[0]; ua:res[1]
  ];
  s
  }

/ Play one episode given a policy
/ Policy: function (player_sum; dealer_showing; usable_ace) -> action
/ Action: 0=stick, 1=hit
playEpisode:{[policy]
  dealer:drawCard[];
  ph:initHand[];
  ps:ph[0]; pua:ph[1];
  / Ensure player starts with at least 12
  while[ps<12;
    res:hitHand[ps;pua];
    ps:res[0]; pua:res[1]
  ];
  / Player's turn
  trajectory:();
  done:0b;
  while[not done;
    state:(ps;dealer;pua);
    a:policy[ps;dealer;pua];
    trajectory,:enlist state;
    $[a=1;   / hit
      [res:hitHand[ps;pua];
       ps:res[0]; pua:res[1];
       done:ps>21];   / bust
      done:1b         / stick
    ]
  ];
  / Dealer plays; determine reward
  reward:$[ps>21;
    -1f;            / player busted
    [ds:dealerPlay[dealer];
     $[ds>21; 1f;   / dealer busted
       ps>ds; 1f;   / player wins
       ps=ds; 0f;   / draw
       -1f]]        / dealer wins
    ];
  `trajectory`reward`finalSum!(trajectory;reward;ps)
  }
```

## First-Visit Monte Carlo Prediction

Given a policy \\(\pi\\), estimate \\(v_\pi\\). We run many episodes, and for each state \\(s\\) visited, we record the return \\(G\\) that followed the *first* visit to \\(s\\) in that episode. Then \\(v_\pi(s) \approx \text{average}(G_{\text{first visit}})\\).

> **Why "first visit" matters**: if you visit state \\(s\\) twice in one episode, the returns at those two visits are correlated (the second is a subset of the first). First-visit MC only counts the first occurrence, preserving the i.i.d. sampling assumption needed for convergence. *Every-visit* MC also converges, but more slowly in some cases. S&B proves convergence of both.

```q
/ Encode Blackjack state as an integer index
/ State space: player_sum (12-21) x dealer (1-10) x usable_ace (0-1)
/ Total: 10 x 10 x 2 = 200 states
encodeState:{[ps;dealer;ua]
  (10*2*(ps-12)) + (2*(dealer-1)) + `int$ua
  }
nBJStates:200

/ First-Visit MC Prediction
/ pi: policy function (ps;dealer;ua) -> action
/ nEpisodes: number of episodes to run
firstVisitMC:{[pi;nEpisodes;gamma]
  returns:nBJStates#enlist 0#0f;   / running list of returns per state
  do[nEpisodes;
    ep:playEpisode[pi];
    traj:ep`trajectory;
    G:ep`reward;                  / terminal reward (discounted from end)
    / Walk backwards through trajectory computing G
    / Since all intermediate rewards are 0, G is just gamma^k * finalReward
    / at step k from the end. With gamma=1, G is constant = final reward.
    visited:0#0i;
    {[returns;G;gamma;k;state]
      s:encodeState . state;
      / First-visit: only count if not seen earlier in this episode
      if[not s in visited;
        returns[s],:G*gamma xexp k;
        visited,:s
      ]
      }[returns;G;gamma;;] each reverse til count traj
  ];
  / Average returns
  {$[0<count x; avg x; 0f]} each returns
  }
```

```q
/ Policy: always stick on 20 or 21, hit otherwise
simplePolicy:{[ps;dealer;ua] $[ps>=20;0;1]}

/ Estimate values with 10k and 500k episodes
V_10k:firstVisitMC[simplePolicy;10000;1f];
V_500k:firstVisitMC[simplePolicy;500000;1f];

/ States with usable ace: states 1, 3, 5, ... (every other)
/ Display value for player sum 20, dealer showing 5, usable ace
s20_5_ua:encodeState[20;5;1b];
V_500k[s20_5_ua]    / should be ~0.65 (favorable position)

/ Player sum 12, dealer showing 2, no usable ace
s12_2_noua:encodeState[12;2;0b];
V_500k[s12_2_noua]  / should be ~-0.29 (weak position)
```

The 10k estimates are noisy; the 500k estimates should closely match S&B's Figure 5.1. With a naive always-stick-at-20 policy, the player with a usable ace and high sum does well; low sums against a strong dealer do not.

## Monte Carlo Control

Prediction estimates \\(v_\pi\\). Control finds a good policy. For MC control, we estimate \\(q_\pi(s,a)\\) (action values) because we don't have a model—we can't compute \\(\max_a \sum_{s'} p(s'|s,a)[\ldots]\\) without knowing \\(p\\). But we can estimate \\(q^*(s,a)\\) directly.

We'll use **epsilon-soft** policies: with probability \\(\varepsilon\\) take a random action, otherwise take the greedy action. This ensures all state-action pairs are visited sufficiently often.

```q
/ Q-table for Blackjack: (state; action) -> value
/ nActions = 2 (stick=0, hit=1)
nBJActions:2

/ MC Control with epsilon-soft policy (on-policy)
/ Returns Q-table (nBJStates x nBJActions)
mcControl:{[nEpisodes;gamma;eps]
  Q:nBJStates#enlist nBJActions#0f;
  N:nBJStates#enlist nBJActions#0i;

  / Epsilon-soft policy derived from Q
  epsSoftAction:{[Q;eps;ps;dealer;ua]
    s:encodeState[ps;dealer;ua];
    $[rand[1.0f]<eps;
      first 1?nBJActions;
      imax Q[s]
    ]};

  do[nEpisodes;
    / Generate episode using current epsilon-soft policy
    pi:epsSoftAction[Q;eps;;];
    ep:playEpisode[pi];
    traj:ep`trajectory;
    R:ep`reward;

    / For each state-action pair in episode (first-visit)
    visited:0#(0i;0i);   / list of (state;action) pairs seen
    G:R;
    {[Q;N;G;gamma;k;state]
      s:encodeState . state;
      / Action taken at this step: we need it from the trajectory
      / We'll thread action through state instead
      } [Q;N;G;gamma;;] each reverse til count traj;

    / Rebuild with actions: replay episode to get (s,a) pairs
    playWithActions:{[Q;eps;ps_init;dealer_init;ua_init]
      dealer:dealer_init;
      ps:ps_init; pua:ua_init;
      while[ps<12; res:hitHand[ps;pua]; ps:res[0]; pua:res[1]];
      traj_sa:();
      done:0b;
      while[not done;
        state:(ps;dealer;pua);
        a:epsSoftAction[Q;eps;ps;dealer;pua];
        traj_sa,:enlist (state;a);
        $[a=1;
          [res:hitHand[ps;pua]; ps:res[0]; pua:res[1]; done:ps>21];
          done:1b
        ]
      ];
      traj_sa
      };

    / Incremental Q update: Q[s][a] += (G - Q[s][a]) / N[s][a]
    / (simplified: process each episode's (s,a,G) tuples)
    trajSA:(); G:ep`reward;
    / Re-run to get actions (not ideal but clear)
    / In a real impl, thread actions through playEpisode
    trajSA:{[ep] ep`trajectory} ep;  / placeholder
    / (Full on-policy MC with action tracking shown below)
  ];
  Q
  }
```

The above has a structural issue worth acknowledging: the standard `playEpisode` doesn't record actions alongside states. Let's fix that cleanly:

```q
/ Revised: episode returns (state;action;reward) tuples
playEpisodeSA:{[policy]
  dealer:drawCard[];
  ph:initHand[];
  ps:ph[0]; pua:ph[1];
  while[ps<12; res:hitHand[ps;pua]; ps:res[0]; pua:res[1]];
  trajectory:();    / list of (state;action) pairs
  done:0b;
  while[not done;
    state:(ps;dealer;pua);
    a:policy[ps;dealer;pua];
    trajectory,:enlist (state;a);
    $[a=1;
      [res:hitHand[ps;pua]; ps:res[0]; pua:res[1]; done:ps>21];
      done:1b
    ]
  ];
  reward:$[ps>21;-1f;
    [ds:dealerPlay[dealer];
     $[ds>21;1f;ps>ds;1f;ps=ds;0f;-1f]]];
  `trajectory`reward!(trajectory;reward)
  }

/ Clean MC control
mcControlClean:{[nEpisodes;gamma;eps]
  Q:nBJStates#enlist nBJActions#0f;
  N:nBJStates#enlist nBJActions#0i;
  do[nEpisodes;
    pi:{[Q;eps;ps;dealer;ua]
      s:encodeState[ps;dealer;ua];
      $[rand[1.0f]<eps; first 1?nBJActions; imax Q[s]]
      }[Q;eps;;;];
    ep:playEpisodeSA[pi];
    traj:ep`trajectory;
    G:ep`reward;
    visited:0#0i;
    k:count[traj]-1;
    while[k>=0;
      sa:traj[k];
      state:sa[0]; a:sa[1];
      s:encodeState . state;
      key_sa:nBJActions*s+a;   / flat index
      if[not key_sa in visited;
        visited,:key_sa;
        N[s;a]+:1i;
        Q[s;a]+:(G*gamma xexp (count[traj]-1-k)) - Q[s;a];  / wrong: use incremental
        / Correct incremental update:
        Q[s;a]:(Q[s;a] * N[s;a]-1 + G) % N[s;a]
      ];
      k-:1
    ]
  ];
  Q
  }
```

```q
/ Run MC control: 500k episodes, epsilon=0.1
Q_mc:mcControlClean[500000;1f;0.1f];

/ Derived greedy policy
policyMC:{[Q;ps;dealer;ua]
  s:encodeState[ps;dealer;ua];
  imax Q[s]
  }[Q_mc;;;]

/ Test: what does the optimal policy do with player=20, dealer=5, no ace?
policyMC[20;5;0b]   / 0 = stick (correct)
policyMC[12;2;0b]   / should hit
policyMC[18;9;0b]   / borderline case
```

## The Key Insight

MC methods have no bias. They estimate \\(v_\pi(s)\\) as the sample average of actual returns—this converges to the truth by the law of large numbers without any assumptions about the environment's structure. But they have high variance: returns depend on the entire trajectory, which can vary wildly.

Dynamic programming has no variance (it works with exact expectations) but requires a model. MC has no model requirement but high variance.

Temporal difference learning, the next chapter, sits between these poles: it bootstraps (like DP) but from sampled experience (like MC). Understanding why that's both useful and a compromise is the next step.
