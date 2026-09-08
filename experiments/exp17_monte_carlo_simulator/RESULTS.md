# Experiment 17 results — validating `MonteCarloSimulator` against experiment 16's already-established finding

Ran: `PYTHONPATH=. python experiments/exp17_monte_carlo_simulator/run.py`.
12 seeds (the same seeds experiment 16 used) × 5 starting states each = 60
policy comparisons. See `docs/research-agenda.md` #7n for the hypothesis
and `docs/related-work.md` §3k for the grounding theory. This closes
Phase 6's third and final remaining gap (`Simulator`) — nonlinear
dynamics (experiment 15) and regime-adaptation + multi-step planning
(experiment 16) were the first two.

## The question

`transintelligence/simulation/`'s `MonteCarloSimulator` is new code
(hand-verified against a deterministic canonical case first —
`tests/test_simulation.py`). Rather than trusting it on a fresh,
unrelated claim, this experiment asks a narrower, pre-registered
question: does it **independently recover** a ranking this program
already established by a completely different methodology? Experiment
16 found, via full multi-seed environment rollouts averaged over 400
episodes each, that `greedy_cusum_adapts` beats `mpc_beam_cusum_adapts`
post-shift (-4.52 vs. -33.06). This experiment reuses experiment 16's own
`Agent` class, `DelayedRegimeShiftControlEnv`, and every constant
(imported directly, not reimplemented), trains the same two agent
configurations to the same post-shift, CUSUM-adapted state, then freezes
them (learning switched off — `agent.observe()` is never called again)
and asks `MonteCarloSimulator.compare_policies` to rank them using
nothing but short, stochastic Monte Carlo rollouts from fresh starting
states, via the environment's true post-shift dynamics formula plus its
real observation noise.

This was a genuine risk, not a formality: the answer was already known,
so a mismatch would have been a real, reportable finding — either a bug
in `MonteCarloSimulator`, or evidence that experiment 16's
full-environment result doesn't generalize to arbitrary starting states.

## Result

| n_rollouts | greedy_cusum_adapts wins | win rate | mean(greedy) | mean(mpc) |
|---|---|---|---|---|
| 200 | 58/60 | 96.7% | -2.29 | -32.98 |
| 5 (sensitivity control) | 53/60 | 88.3% | — | — |

`MonteCarloSimulator` recovers experiment 16's ranking cleanly: greedy
wins 58 of 60 comparisons, and the two policies' mean simulated rewards
(-2.29 vs. -32.98) land close to experiment 16's own full-environment
numbers (-4.52 vs. -33.06) despite using a completely different
methodology — short rollouts from arbitrary starting states, not full
400-episode averages from the environment's own reset distribution.

**Sensitivity control**: dropping `n_rollouts` from 200 to 5 (same
trained agents, same start states, same environment) reduces the win
rate from 96.7% to 88.3% — noisier, as expected from fewer samples, but
still correctly favoring greedy in the large majority of cases, since
the true effect size is large relative to the per-rollout noise
(`noise_sigma=0.1`). This confirms `n_rollouts` is doing real
work — the verdict is not just as reliable at 5 samples as at 200 — while
also showing the effect is large enough that even a small sample rarely
gets the ranking backwards.

The two comparisons `mpc` won (both at `n_rollouts=200`) were both
close calls (e.g. seed 6, start position -2.16: -0.0843 vs. -0.0502) —
starting states already very close to the target, where a single
extra planning step's small advantage can occasionally outweigh its
usual multi-step cost, not a contradiction of the general finding.

## What this establishes

- `MonteCarloSimulator.simulate`/`compare_policies` behave correctly
  on a real, previously-unseen combination of policies and dynamics,
  not just the small deterministic case they were hand-verified against.
- Experiment 16's finding is robust to starting state — it was not an
  artifact of averaging over the environment's own `reset()` distribution
  specifically.
- Phase 6's three remaining gaps listed after experiment 14 (nonlinear
  dynamics, regime-adaptation + multi-step planning, `Simulator`) are
  now all closed; every reasoning protocol stub in
  `transintelligence/reasoning/interfaces.py` from before Phase 4 now has
  a real implementation except `Verifier`.

## What this does not establish

- **Only one pair of policies, one environment, and one severity were
  tested** — this validates `MonteCarloSimulator` on the specific case
  experiment 16 already characterized in detail, not on a broad range of
  policy comparisons.
- **`n_rollouts` sensitivity was checked at only two values (5 and
  200)** — the point at which the verdict becomes unreliable, if it
  exists at all for this effect size, was not located.
- **The post-shift regime only** — pre-shift, where experiment 16 found
  both agents perform near-identically (-0.02 to -0.03), was not
  separately validated here; the interesting, decisive ranking is
  post-shift, so that is what this experiment targeted.
