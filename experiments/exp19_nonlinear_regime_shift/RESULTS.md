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

## Follow-up: what actually causes the detection degradation?

Ran: `PYTHONPATH=. python experiments/exp19_nonlinear_regime_shift/detection_diagnosis.py`.
10 seeds, `sign_flip` severity (the case the detection gap was found
in). Three hypotheses checked, all against matched linear/nonlinear
agents on identical seeds:

1. **Steady-state (pre-shift) residual noise** — the original
   hypothesis above, reproduced as code instead of an ad-hoc check:
   REFUTED again, 0.3045 vs. 0.3118 mean stdev across 10 seeds
   (previously 5).
2. **Residual autocorrelation (lag-1, pre-shift)** — a serially
   correlated residual stream could slow CUSUM's cumulative-sum
   statistic without needing higher variance at all. Also REFUTED:
   -0.0035 (nonlinear) vs. 0.0027 (linear), both indistinguishable from
   zero.
3. **Post-shift (pre-detection) residual variability, using the agent's
   own real greedy action selection rather than a forced/uniform one** —
   a genuine, measured difference: nonlinear post-shift residual stdev
   averaged 1.42 across 10 seeds vs. linear's 0.91, and the *spread*
   across seeds was also much larger for nonlinear (1.74 vs. 1.10, max
   observed 2.08 vs. 1.52) — several nonlinear seeds show markedly
   noisier post-shift residual behavior than any linear seed does.

**This narrows the explanation without fully resolving it.** Pre-shift,
the two models are statistically indistinguishable in both variance and
autocorrelation — the difference is not an intrinsic property of the
nonlinear functional form's fitting or residual behavior in a stable
regime. It emerges specifically post-shift, before detection, while both
agents are still acting on their stale (pre-shift-fit) model — plausibly
because the stale nonlinear model's own greedy action choices interact
with the now-shifted dynamics more erratically than the stale linear
model's do, feeding CUSUM a genuinely noisier, more seed-dependent
signal to detect against, not just a smaller one. **The exact mechanism
generating that extra post-shift variability was not identified** — this
is a real, verified narrowing (from "unexplained" to "post-shift
action-driven interaction, not steady-state model noise"), not a
complete account.

## Follow-up: mitigating the cold-start cost (Finding 1)

Ran: `PYTHONPATH=. python experiments/exp19_nonlinear_regime_shift/cold_start_mitigation.py`.
10 seeds, both severities. Rather than only reporting Finding 1's
cold-start cost, this attempts to engineer around it — with an honest
account of what failed before what worked.

**Attempt 1 (FAILED): shrink `refit_interval`.** `_training_transitions()`'s
post-reset pool grows continuously regardless of `refit_interval` —
shrinking it should only help or be neutral, never hurt, by catching
sufficient new data sooner. **Directly tested and refuted**: shrinking
`refit_interval` from 20 to 2 made the mild-shift reversal substantially
*worse* (`oracle - never` went from -0.14 to -0.71), and even flipped
the *severe*-shift case from a clear adaptation benefit (+1.25) to a net
cost (-0.43). **Investigated rather than left as a puzzling negative
result**: instrumentation shows the first action to reach 3 observations
(from the initial random-fallback phase) triggers immediate, permanent
greedy exploitation of that one action's noisy, barely-identified
coefficient — `choose_action` has no exploration bonus once *any* action
is known. Measured directly: at `refit_interval=20`, no single action
exceeds ~38% of post-reset choices in the first 60 trials; at
`refit_interval=2`, one action dominates 88-97% of choices almost
immediately — premature lock-in onto an unreliable estimate, worse than
continued random exploration. A genuine, deeper finding this experiment's
main run didn't surface: the cold-start cost isn't only about
insufficient data to fit anything — it's compounded by a purely-greedy
policy's total lack of exploration once *any* action becomes "known."

**Attempt 2 (partially succeeded): require full action coverage before
going greedy.** `FullCoverageOracleAgent`/`FullCoverageCUSUMAgent` keep
exploring uniformly at random until *every* action has a fitted
coefficient, not just one, avoiding lock-in onto a single early estimate.

| condition | mild_attenuation oracle−never | sign_flip oracle−never | sign_flip cusum−never |
|---|---|---|---|
| original (greedy on first known action) | -0.1441 | +1.2460 | +0.3130 |
| full-coverage-before-greedy | **+0.0521** | **+1.4023** | **+0.6238** |

**This closes Finding 1 outright**: `oracle_adapts` no longer
underperforms `never_adapts` under a mild shift (-0.14 → +0.05) — and
also *improves* the severe-shift case, both for the oracle (+1.25 →
+1.40) and for `cusum_detects_and_adapts` (+0.31 → +0.62, roughly
doubling the adaptation benefit over `never_adapts`).

**It does not fully close Finding 2**: `full_coverage_cusum`'s post-shift
reward (-1.0376) still trails `sliding_window_baseline`'s (-0.4375) under
`sign_flip` — the confound-controlled comparison experiment 19's design
needed still fails, though the gap narrows substantially from the
original (-1.3484 vs. the same -0.4375 baseline). A purely-greedy
CUSUM-adaptive agent, even with the lock-in problem fixed, is still not
earning its complexity over the simpler recency heuristic here.

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

- **The exact mechanism generating degraded CUSUM detection reliability
  under a nonlinear model is still not fully identified** — the
  follow-up above ruled out two candidate explanations (steady-state
  residual noise, residual autocorrelation) and localized the effect to
  post-shift, pre-detection behavior specifically, showing it's
  plausibly driven by the stale model's own action choices interacting
  with the shifted environment rather than an intrinsic property of the
  nonlinear fit — but did not pin down precisely why that interaction is
  noisier for a nonlinear model than a linear one.
- **Only one nonlinearity (experiment 15's quadratic restoring force)
  and one GAMMA value (0.3) were tested** — whether the cold-start cost
  scales with nonlinearity strength, or is specific to this functional
  form's higher parameter count, is untested.
- **The cold-start cost's mitigation is partial, not complete** (see the
  follow-up above): full-coverage-before-greedy closes Finding 1 (the
  mild-shift reversal) and improves the severe-shift oracle and CUSUM
  cases, but `cusum_detects_and_adapts` still doesn't beat
  `sliding_window_baseline` under `sign_flip` — Finding 2's confound-
  controlled comparison still fails, just by a smaller margin. A softer
  data-retention transition (rather than a hard reset) was not tried.
- **Not combined with multi-step planning or beam search** (experiments
  12/14/16) — whether this cold-start cost compounds further with
  multi-step lookahead is untested.
