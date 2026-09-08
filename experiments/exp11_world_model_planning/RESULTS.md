# Experiment 11 results — does explicit dynamics modeling beat model-free value estimation?

Ran: `PYTHONPATH=. python experiments/exp11_world_model_planning/run.py`.
20 seeds × 2000 trials per condition (first 60 trials per seed are a
shared random-action warmup, excluded from the reported metrics). See
`docs/research-agenda.md` #7h for the hypothesis and
`docs/master-context.md` §13/§19 for Phase 6's own formalization
(`M(S_t, A_t) → S_{t+1}`), and `docs/related-work.md` §3e for the
grounding theory (Sutton's Dyna architecture, Ha & Schmidhuber's *World
Models*).

## The environment and the confound this needed to control for

`environments/transworld/resource_control_env.py`'s `ResourceControlEnv`:
each trial draws a fresh state `s`, the agent picks a discrete "nudge"
action, and `reward = -(s + nudge(action) + noise - target)²`. The true
**dynamics** are linear in state given the action; the true **reward**
is a quadratic function of the resulting state. Showing a state-aware
agent beat a state-blind one would only prove using context helps at
all — trivial, and not Phase 6's actual claim. Two state-aware
conditions, with identical state access and the identical regression
tool (`ordinary_least_squares`), isolate the real claim instead:
`model_free_linear_q` fits reward directly as a linear function of
state per action; `world_model` fits the state *transition* linearly per
action (`LinearDynamicsModel`), then applies the known reward formula to
score candidates.

## Result

| condition | mean reward | mean ceiling | mean regret |
|---|---|---|---|
| state_blind | -8.7295 | -1.9015 | 6.8280 |
| model_free_linear_q | -5.4436 | -1.9015 | 3.5421 |
| world_model | -1.9013 | -1.9015 | **-0.0002** |
| oracle_dynamics | -1.9013 | -1.9015 | -0.0002 |

("mean ceiling" is the true expected reward of the optimal action from
each trial's actual state, computed from the known dynamics — the
theoretical best any policy could achieve in expectation, including the
irreducible noise floor. "regret" = ceiling − reward.)

**`world_model` matches the `oracle_dynamics` ceiling almost exactly**
(regret -0.0002, indistinguishable from floating-point/sampling noise) —
learning the (correctly-specified, linear) dynamics from data costs
essentially nothing relative to knowing them exactly. **`state_blind`
has by far the worst regret** (6.83), as expected for a policy that
can't tell current states apart at all. **`model_free_linear_q` sits in
between** (regret 3.54) — meaningfully better than ignoring state
entirely, but roughly 3.5x further from optimal than the world-model
agent, despite seeing identical states and using the identical
regression tool.

## Why the identically-tooled model-free agent still falls far short

Investigated directly rather than left as "well, it's linear so it's
worse" — the fitted `model_free_linear_q` coefficients (one seed, full
run) are genuinely informative, not degenerate or near-zero:

| action | true nudge | fitted intercept | fitted slope |
|---|---|---|---|
| large_down | -2.0 | -18.34 | 2.80 |
| large_up | 2.0 | -11.70 | -1.26 |
| tiny_up | 0.3 | -12.21 | 1.03 |

The mechanism: each action's true reward is a **downward parabola in
state**, peaking exactly where that action lands the state on target
(e.g. `large_down`'s peak is at `s=2.0`, since a -2.0 nudge from `s=2.0`
reaches the target exactly). A linear fit is forced to be **monotonic**
in state — it can match the parabola's rising side reasonably well (e.g.
`large_down`'s positive slope of 2.80 does track the true reward
increasing as `s` rises toward 2.0), but it has no way to represent the
peak-then-decline: past `s=2.0`, the true reward starts falling again
while the linear fit keeps extrapolating upward, systematically
misranking `large_down` against other actions for states beyond its
optimal zone. **This is the same lesson experiment 10 established for
effect estimation — a linear coefficient cannot represent a relationship
that depends on where a unit starts — now shown in a planning setting**:
the value-function surface here isn't just "nonlinear," it's
non-monotonic, which a linear model can never represent regardless of
how much data it gets. `world_model` doesn't have this problem because
it never has to fit the reward's curvature at all — it fits the (truly
linear) *transition*, then plugs the result into the exact, known reward
formula.

## What this establishes

`LinearDynamicsModel` (`transintelligence/world_models/model.py`) was
verified against a hand-computed exact-linear-dynamics case (two actions
with different known intercepts/slopes, recovered to floating-point
precision) *before* this experiment was built, in
`tests/test_world_models.py` — the same discipline as every other kernel
mechanism in this codebase. The experiment confirms the qualitative
claim survives sampling noise and a genuinely fair, identically-tooled
alternative: explicit dynamics modeling isn't just "using more
information," it's a specific structural decomposition (model the easy
part exactly, apply the known formula for the hard part) that a
direct model-free fit of the same complexity class cannot match.

## What this does not establish

- **Single-step lookahead only, not multi-step planning** — the
  environment resets to a fresh random state every trial rather than
  chaining into a rollout; `Simulator`/`Planner` remain empty stubs. This
  tests whether a learned model helps *pick the next action*, not
  whether it helps plan a *sequence* of actions toward a distant goal.
- **The dynamics are genuinely linear here** — the entire advantage
  demonstrated rests on the transition being easy to model exactly with
  a linear fit while the reward is not; a nonlinear (e.g. saturating)
  true transition would need `StructuralEquation`-style nonlinear
  extensions (Phase 5, experiment 10) to keep this advantage, untested
  in combination here.
- **Discrete, small, fixed action set** (6 actions) — no continuous
  action space, no action-space search/optimization beyond exhaustive
  enumeration over 6 candidates.
- **No comparison against a nonlinear model-free baseline** (e.g. a
  quadratic-feature regression, which *could* represent the true reward
  surface exactly) — the claim here is specifically about a
  linear-vs-linear comparison with different targets (dynamics vs.
  reward), not a claim that model-free methods can never represent this
  reward surface with a richer function class.
- **Stationary environment, no regime changes** — unlike Experiments 4/5
  (Phase 3/4), the dynamics never shift mid-experiment; combining
  learned world models with regime-change detection is untested.
