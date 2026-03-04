# Reinforcement Learning with Q

*Sutton & Barto's canonical RL text, implemented in kdb+/q.*

**Read online**: [cloudstreet-dev.github.io/Reinforcement-Learning-with-Q](https://cloudstreet-dev.github.io/Reinforcement-Learning-with-Q/)

---

There are many reinforcement learning tutorials. All of them are in Python. This one is not.

This book follows [Sutton & Barto's *Reinforcement Learning: An Introduction* (2nd ed.)](http://incompleteideas.net/book/the-book-2nd.html) chapter by chapter, implementing every algorithm in kdb+/q. The reader is a q developer who wants to understand RL properly, not port someone else's NumPy code. The code runs. The math is right. The environments match S&B so you can cross-reference.

## Contents

| Chapter | Topic | S&B Reference |
|---|---|---|
| [Introduction](src/introduction.md) | Why RL in Q? Why Not. | — |
| [Bandits](src/bandits.md) | Epsilon-greedy, UCB, gradient bandit | Ch. 2 |
| [MDPs](src/mdp.md) | States, actions, rewards, GridWorld | Ch. 3 |
| [Dynamic Programming](src/dynamic.md) | Policy eval/iteration, value iteration | Ch. 4 |
| [Monte Carlo](src/montecarlo.md) | Blackjack, first-visit MC, MC control | Ch. 5 |
| [Temporal Difference](src/td.md) | TD(0), SARSA, Q-Learning, cliff walking | Ch. 6 |
| [N-Step Methods](src/nstep.md) | N-step TD/SARSA, eligibility traces | Ch. 7 |
| [Function Approximation](src/approximation.md) | Tile coding, semi-gradient TD, mountain car | Ch. 9–10 |
| [Policy Gradient](src/policy.md) | REINFORCE, baseline, actor-critic | Ch. 13 |
| [Conclusion](src/conclusion.md) | Where to go from here | — |

## Prerequisites

- kdb+/q — the [personal edition](https://kx.com/developers/download-licenses/) covers everything in this book
- Familiarity with q syntax (`imax`, `{x+y}/`, functional amend)
- Basic ML literacy helps; deep RL background is not required

## Building Locally

```bash
# Install mdBook (https://rust-lang.github.io/mdBook/)
cargo install mdbook

# Build
mdbook build

# Serve with live reload
mdbook serve --open
```

## License

See [LICENSE](LICENSE).
