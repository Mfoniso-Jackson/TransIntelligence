# Experiment 9 results — instrumental variables (2SLS) and front-door adjustment

Ran: `PYTHONPATH=. python experiments/exp09_iv_and_frontdoor/run.py`.
30 seeds × 500 samples per condition. See `docs/research-agenda.md` #7f
for the hypothesis and `docs/related-work.md` §3d for the grounding
theory (Wright 1928, Wright 1934, Pearl 1995, Bound/Jaeger/Baker 1995).

Both parts test identification strategies for the case experiments 6 and
8 couldn't handle: a confounder that is genuinely **unobserved** — not in
the dataset at all, so no adjustment set (valid or invalid) can be
formed. Both parts also each add a confound control built in from the
start, not discovered after a clean-looking number.

## Part 1: 2SLS vs. instrument strength

True effect of `X` on `Y` is 0.6, with unobserved confounder `U` driving
both. `Z` is a valid instrument whose strength (its effect on `X`) is
swept from 0.9 down to exactly 0.0.

| instrument strength | mean naive OLS | mean 2SLS | 2SLS stdev | naive abs bias | 2SLS abs bias |
|---|---|---|---|---|---|
| 0.9 | 1.0118 | 0.5869 | 0.0476 | 0.4118 | 0.0131 |
| 0.5 | 1.2504 | 0.5733 | 0.0885 | 0.6504 | 0.0267 |
| 0.2 | 1.4298 | 0.5006 | 0.2599 | 0.8298 | 0.0994 |
| 0.05 | 1.4732 | -0.7561 | 18.6777 | 0.8732 | 1.3561 |
| 0.0 | 1.4765 | 1.9150 | 3.0852 | 0.8765 | 1.3150 |

**With a strong-to-moderate instrument (0.9 down to 0.2), 2SLS is
dramatically less biased than naive OLS** — naive is off by 0.41-0.83
throughout (it never uses `Z` at all, so `U`'s confounding never gets
removed regardless of instrument strength), while 2SLS's bias stays
under 0.10 down to strength 0.2.

**Below that, 2SLS doesn't just degrade — it becomes actively worse and
wildly unstable, the textbook weak-instrument pathology (Bound, Jaeger,
Baker 1995).** At strength 0.05, individual-seed 2SLS estimates ranged
from **-96.3 to +29.4** (stdev 18.7, vs. 0.048 at strength 0.9 — a
~390x increase), and the mean estimate's bias (1.36) is now *worse* than
naive's (0.87). This is not a smooth, forgiving degradation: 2SLS is a
ratio estimator (reduced-form covariance divided by the first-stage
coefficient), and dividing by a first-stage coefficient that's
statistically indistinguishable from zero produces exactly this kind of
explosive, heavy-tailed instability — a stronger and more dramatic
version of Bound/Jaeger/Baker's warning than "IV bias merely approaches
OLS bias," not a contradiction of it (their result is an asymptotic
bias-direction claim for a specific correlated-error setup; this
simulation's iid errors instead surface the more visible finite-sample
variance blow-up, arguably the more practically dangerous failure mode
of the two).

## Part 2: front-door adjustment, valid vs. violated assumption

True total effect (`X` on `Y` via `M`) is 0.35 (`0.7 × 0.5`). Unobserved
confounder `U` drives `X` and `Y` directly in both conditions; the only
difference is whether `U` also affects the mediator `M` — which
front-door identification requires it must not.

| condition | mean naive OLS | mean front-door | naive abs bias | front-door abs bias |
|---|---|---|---|---|
| valid (U does not affect M) | 1.2307 | 0.3406 | 0.8807 | 0.0094 |
| violated (U affects M, strength 0.6) | 1.5606 | 1.2804 | 1.2106 | 0.9304 |

**When its assumption holds, front-door adjustment recovers the true
effect almost exactly (bias 0.0094) against a naive estimate that's off
by 0.88.** **When the assumption is violated — `U` given a direct 0.6
effect on `M`, using the identical `front_door_adjustment` call on the
now-invalid data — the estimate is nearly as biased as naive (0.93 vs.
1.21), a ~99x jump in bias from the valid condition.** This is the same
"don't just show the positive case, show what happens when the method's
own precondition is violated" discipline experiment 6's collider
adjustment used: front-door adjustment is not a general-purpose fix for
unobserved confounding, it's a fix that depends on a specific structural
assumption (the confounder doesn't reach the mediator) actually holding,
and this experiment demonstrates the failure directly rather than
stating it as a caveat.

## What this establishes about the two mechanisms

`two_stage_least_squares` and `front_door_adjustment` were verified
against hand-computed cases *before* this experiment was built (see
`tests/test_causal_reasoning.py`): both recover close to the true effect
against a heavily biased naive estimate at moderate sample sizes and
strong instrument/valid-assumption settings. This experiment confirms
both mechanisms' *known theoretical failure modes* are real and
reproduce here specifically, not just their success cases:
weak-instrument instability for 2SLS, and mediator-confounding bias for
front-door adjustment when its precondition breaks.

## What this does not establish

- **Just-identified 2SLS only** (one instrument, one endogenous
  regressor) — overidentified systems (more instruments than endogenous
  regressors), which allow overidentification tests (e.g. the Sargan
  test) that can *detect* instrument invalidity from data, are not
  implemented.
- **No formal weak-instrument diagnostic** (e.g. the first-stage F-
  statistic rule of thumb) — this experiment demonstrates the pathology
  exists and is severe, not a way to detect it automatically before
  trusting a 2SLS estimate on real data.
- **Front-door adjustment's other two conditions weren't separately
  stress-tested** — only "U also affects M" was violated here; a direct
  X→Y edge (breaking full mediation) or an unblocked backdoor path from
  M to Y not through X would be different, untested violations.
- **Linear structural equations only**, same simplification as the rest
  of `reasoning/causal/` and `reasoning/counterfactual/`.
- **No comparison to sensitivity-analysis approaches** for unobserved
  confounding (e.g. Rosenbaum bounds) that don't require a valid
  instrument or mediator at all — a different family of methods for the
  same underlying problem, not attempted here.
