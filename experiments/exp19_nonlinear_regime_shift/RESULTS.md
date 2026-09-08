# Experiment 19 results — extending regime-adaptation to a nonlinear dynamics model

Ran: `PYTHONPATH=. python experiments/exp19_nonlinear_regime_shift/run.py`.
15 seeds × 3000 trials per condition per severity (first 60 trials are a
shared random-action warmup). See `docs/research-agenda.md` #7p for the
hypothesis and `docs/related-work.md` §3m for the grounding theory. This
is a synthesis, structurally identical to experiment 13 (`CUSUMTemporalReasoner`
+ a learned dynamics model) with one substitution: `NonlinearDynamicsModel`
(generalized from experiment 15 into `transintelligence/world_models/`)
in place of `LinearDynamicsModel`.

## The question, and why the answer wasn't obvious

Experiment 13 found detection-triggered adaptation composes cleanly with
a *linear* dynamics model: no benefit under a mild shift (a real,
investigated null result), a clear benefit under a severe one.
`CUSUMTemporalReasoner` monitors a model's own prediction residuals, not
the raw state — it doesn't know or care whether the model producing
those residuals is linear or nonlinear. But `NonlinearDynamicsModel` has
one more coefficient to estimate from the same refit window than
`LinearDynamicsModel` (3 observations minimum per action, not 2) — there
was no a priori guarantee a detection/adaptation pipeline calibrated
implicitly around a linear model's behavior would transfer unchanged.

## Result

**`mild_attenuation` (`post_shift_scale=0.4`):**

| condition | pre-shift reward | post-shift reward |
|---|---|---|
| never_adapts | -0.1515 | **-0.3630** |
| oracle_adapts | -0.1515 | -0.5278 |
| sliding_window_baseline | -0.1939 | -0.4108 |
| cusum_detects_and_adapts | -0.1515 | -0.3630 |

**`sign_flip` (`post_shift_scale=-1.0`):**

| condition | pre-shift reward | post-shift reward |
|---|---|---|
| never_adapts | -0.1515 | -1.6340 |
| oracle_adapts | -0.1515 | **-0.4151** |
| sliding_window_baseline | -0.1939 | -0.4255 |
| cusum_detects_and_adapts | -0.1515 | -1.3364 |

Detection (`sign_flip`): 9/15 seeds detected the true shift (vs.
experiment 13's 15/15 under its linear model), mean latency 125.2 trials
(vs. experiment 13's 23.0), 0/15 false positives (vs. experiment 13's
3/15).

## Finding 1: under a mild shift, adaptation actively HURTS — not just fails to help

`oracle_adapts`, given the true shift trial exactly, performs **worse**
than `never_adapts` (-0.5278 vs. -0.3630) — a qualitatively different
result from experiment 13's analogous case, where `oracle_adapts` was
only *marginally* worse than `never_adapts` (-5.0996 vs. -5.0823, a
~0.3% gap) and reported as "statistically indistinguishable." Here the
gap is a real ~45% relative degradation.

**Investigated and confirmed directly, not just inferred**: `OracleAdaptsAgent`
discards ALL pre-shift data the instant `_training_transitions()`
switches to post-shift-only, but the model itself only updates on the
next scheduled refit boundary (`REFIT_INTERVAL=20`), and
`NonlinearDynamicsModel` needs 3 fresh observations per action before it
can fit *any* action at all. Direct instrumentation of a fresh
post-shift agent confirms `known_actions()` stays completely empty for
the entire first 19-trial refit window — forcing pure random action
selection the whole time (`tests/test_exp19_nonlinear_regime_shift.py`)
— and even after the first refit, only 4 of 6 actions were known in one
traced seed, not the full set. `LinearDynamicsModel`'s lower
per-action data requirement (2 vs. 3) gives it a shorter, cheaper
version of this same cold-start gap, which is why experiment 13 never
surfaced this as a distinct, sign-flipping effect: under a mild shift,
the stale-but-still-roughly-right old model was cheap enough to keep
using that a brief, mild cold start could plausibly wash out; a nonlinear
model's longer cold start cannot.

## Finding 2: under a severe shift, adaptation still wins overall — but detection is measurably less reliable

`oracle_adapts` still clearly beats `never_adapts` under `sign_flip`
(-0.4151 vs. -1.6340) — the stale model's harm is severe enough that
even a real cold-start cost is worth paying, mirroring experiment 13's
own conclusion. But `cusum_detects_and_adapts` (-1.3364) barely beats
`never_adapts` and does **not** beat `sliding_window_baseline`
(-0.4255) — the confound-controlled comparison experiment 13's own
design was built to pass, and did pass there, **fails here**: explicit
CUSUM detection is not earning its complexity over the simpler recency
heuristic when the underlying model is nonlinear.

Detection itself is measurably worse: only 9/15 seeds detected the true
shift at all (vs. 15/15 for the linear model in experiment 13), and mean
latency more than quintupled (125.2 vs. 23.0 trials). **One natural
explanation was checked directly and refuted**: pre-shift residual
standard deviation is essentially identical between the linear and
nonlinear settings (~0.30–0.31 in both, across 5 matched seeds) — the
nonlinear model is not simply noisier in steady state. The true cause of
slower, less reliable detection under a nonlinear model is **not
established** by this experiment; it remains an open question.

`sliding_window_baseline`, which never fully empties its training set
(always the most recent 200 transitions, aging out old data gradually
rather than discarding everything at once), avoids Finding 1's cold-start
cliff entirely and ends up the most practically competitive adaptive
strategy in both severities here — a genuine reversal of experiment 13's
own preference for CUSUM-triggered adaptation, specific to combining
detection with a data-hungrier nonlinear model.

## What this establishes

- **Regime-adaptation does not generalize for free from a linear to a
  nonlinear dynamics model.** The higher per-action data requirement a
  correctly-specified nonlinear model needs introduces a real,
  measurable cold-start cost that a full-reset adaptation strategy pays
  every time it resets — large enough to flip a mild shift's honest null
  result (experiment 13) into an actively negative one here.
- **CUSUM detection reliability itself degrades under a nonlinear
  model** (9/15 vs. 15/15 detected, ~5.4x slower latency) for a reason
  this experiment ruled out (steady-state residual noise) but did not
  identify.
- **A gradual, never-fully-empties adaptation strategy (sliding window)
  is more robust to this cold-start cost than either a hard-reset oracle
  or CUSUM-triggered detection**, reversing experiment 13's own
  preference ordering — visible only by testing the actual nonlinear
  combination, not predictable from experiments 13 or 15 individually.

## What this does not establish

- **The cause of degraded CUSUM detection reliability under a nonlinear
  model is unconfirmed** — the natural "noisier residuals" hypothesis
  was checked directly and refuted; no alternative mechanism was
  identified or tested.
- **Only one nonlinearity (experiment 15's quadratic restoring force)
  and one GAMMA value (0.3) were tested** — whether the cold-start cost
  scales with nonlinearity strength, or is specific to this functional
  form's higher parameter count, is untested.
- **The cold-start cost itself was not mitigated or engineered around**
  (e.g. a softer transition that retains some pre-shift data temporarily,
  or an immediate forced refit at detection rather than waiting for the
  next scheduled boundary) — this experiment reports the cost as found,
  it does not attempt to fix it.
- **Not combined with multi-step planning or beam search** (experiments
  12/14/16) — whether this cold-start cost compounds further with
  multi-step lookahead is untested.
