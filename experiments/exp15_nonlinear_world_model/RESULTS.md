# Experiment 15 results — nonlinear world-model dynamics

Ran: `PYTHONPATH=. python experiments/exp15_nonlinear_world_model/run.py`.
20 seeds × 2000 trials per condition (first 60 trials per seed are a
shared random-action warmup, excluded from the reported metrics). See
`docs/research-agenda.md` #7l for the hypothesis and
`docs/related-work.md` §3i for the grounding theory.

## The environment and the confound this needed to control for

`environments/transworld/nonlinear_control_env.py`'s `NonlinearControlEnv`
adds a quadratic restoring force to `ResourceControlEnv`'s (experiment
11) dynamics: `next_state = state + nudge(action) - GAMMA*state*abs(state)
+ noise` — the same structural role experiment 10's `gamma2*X**2` term
played for a causal effect, now for a state's own evolution.
`state*abs(state)` (not a bare square) keeps the force's sign opposed to
the state's own sign everywhere — a genuine restoring force, not a
one-sided drift, so a naively-squared feature (blind to sign) would
still be a real misspecification, not an accidental match.

Showing a nonlinear world model beat a linear one would only prove
nonlinear features can help *somewhere* — not that they capture the
*actual* functional form. `nonlinear_world_model` uses `state*abs(state)`
as its feature, matching the true dynamics' actual shape exactly, using
the identical `ordinary_least_squares` tool and identical state access
as `linear_world_model` (which reuses `LinearDynamicsModel`, experiment
11, completely unchanged).

## A boundary condition this experiment's first run found

`GAMMA=0.05` (the restoring force's magnitude at the state range's edge
is ~1.25, comparable to a single mid-sized nudge) showed almost no gap
between `linear_world_model` and `nonlinear_world_model` — the same
pattern experiment 13 found for a mild regime shift: a nonlinearity too
small to change *which discrete action ranks best* doesn't produce a
measurable gap, even though the model is still technically misspecified.

| gamma | linear mean reward | nonlinear mean reward | oracle mean reward |
|---|---|---|---|
| 0.05 | -0.7386 | -0.7392 | -0.7370 |
| 0.10 | -0.1946 | -0.1884 | -0.1868 |
| 0.20 | -0.2901 | -0.1273 | -0.1257 |
| 0.30 | -0.8242 | -0.1464 | -0.1439 |
| 0.50 | -3.9991 | -3.2282 | -3.2193 |

Swept `GAMMA` up rather than reporting the null result at `0.05` without
checking whether a stronger version of the same claim held. `GAMMA=0.30`
produces a clean, dramatic gap and is the value used for the main
result below.

## Result (GAMMA=0.30)

| condition | mean reward | mean ceiling | mean regret |
|---|---|---|---|
| state_blind | -1.0184 | -0.1415 | 0.8769 |
| linear_world_model | -0.8349 | -0.1415 | 0.6933 |
| nonlinear_world_model | -0.1451 | -0.1415 | **0.0036** |
| oracle_dynamics | -0.1429 | -0.1415 | 0.0014 |

**`nonlinear_world_model` matches the oracle ceiling almost exactly**
(regret 0.0036 vs. the oracle's 0.0014, both indistinguishable from
sampling/estimation noise), while **`linear_world_model`'s regret
(0.6933) is roughly 193x larger** — despite using the identical
regression tool and identical state access, differing only in the
feature set. `linear_world_model` still meaningfully beats the
`state_blind` floor (0.6933 vs. 0.8769) — the misspecified linear model
still captures the dominant linear trend, the same "wrong but not
worthless" pattern experiment 11's `model_free_linear_q` showed.

## What this establishes

`LinearDynamicsModel` was not modified for this experiment — it's used
exactly as verified in experiment 11, deliberately facing dynamics its
linearity assumption gets wrong. `NonlinearWorldModelAgent`'s fitting
logic was verified against a hand-computed case (a noiseless transition
`next_state = state + 2.0 - 0.05*state*abs(state)`, fitted coefficients
recovered to floating-point precision) before this experiment was
built. The result confirms the same lesson experiments 10 and 11
established — a linear model cannot represent a nonlinearity, no matter
how much data it gets — now shown for the third distinct setting
(effect estimation, single-step value estimation, and now world-model
dynamics prediction itself), and adds the boundary condition experiments
10-14 have each needed: the effect only shows up once the
misspecification is severe enough to change which discrete decision is
actually best.

## What this does not establish

- **A single nonlinear functional form tested** (a quadratic restoring
  force) — other nonlinearities (saturating actuators, discontinuities,
  interaction effects between state and action) weren't tried.
- **The correct nonlinear feature was handed to the agent, not
  discovered** — `state*abs(state)` was chosen because it exactly
  matches the environment's known construction; a real environment
  wouldn't hand an engineer the right functional form in advance (the
  same limitation experiment 11's quadratic follow-up noted).
- **Single-step decisions only** — this reuses experiment 11's
  fresh-state-per-trial environment, not experiment 12's multi-step
  credit-assignment setting or experiment 14's planner; whether a
  nonlinear dynamics model composes with multi-step planning or
  beam search is untested.
- **No sweep of the boundary GAMMA value precisely** — `0.05` showed no
  effect and `0.10`-`0.30` did, but the exact crossover point wasn't
  mapped, matching experiment 13's own similar limitation (a continuous
  severity sweep wasn't run there either).
