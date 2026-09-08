# Experiment 13 results — combining world models with regime-change detection

Ran: `PYTHONPATH=. python experiments/exp13_regime_shift_world_model/run.py`.
15 seeds × 3000 trials per condition per severity (first 60 trials are a
shared random-action warmup). See `docs/research-agenda.md` #7j for the
hypothesis and `docs/related-work.md` §3g for the grounding theory (this
is a synthesis of experiment 5's `CUSUMTemporalReasoner` and experiment
11's `LinearDynamicsModel`, not a new mechanism).

## This is a two-part story, not one number — and the first part wasn't the plan

The original design tested one fixed regime shift. That run found
`oracle_adapts` (given the true shift trial exactly, zero detection
delay) performed **statistically indistinguishably from `never_adapts`**
— the opposite of the hypothesis. Investigated rather than reported as a
clean positive result: with this environment's state range wide relative
to its nudge magnitudes, most trials start far enough from target that
both a stale and a correctly-calibrated model pick the *same*
largest-available nudge regardless of the exact scale factor. A mild
miscalibration rarely changes *which action looks best* — only a shift
severe enough to change the *ranking* of actions should matter. That
turned this into a deliberate two-severity sweep.

## Part 1 — `mild_attenuation` (`post_shift_scale=0.4`): the honest null case

Nudges get weaker after the shift but keep the same direction.

| condition | pre-shift reward | post-shift reward |
|---|---|---|
| never_adapts | -1.8866 | -5.0823 |
| oracle_adapts | -1.8866 | -5.0996 |
| sliding_window_baseline | -2.1405 | -5.3411 |
| cusum_detects_and_adapts | -2.4639 | -5.4131 |

**All four conditions perform similarly post-shift — `oracle_adapts`
does not beat `never_adapts`, and the two adaptive conditions
(`sliding_window_baseline`, `cusum_detects_and_adapts`) are actually
slightly *worse*.** This is a real, reported negative finding, not
smoothed over: discarding a large body of already-converged pre-shift
data and re-fitting from a small post-shift sample has a genuine cost
(confirmed by inspecting the post-shift reward trajectory in 100-trial
chunks — `oracle_adapts` starts *worse* than `never_adapts` in the first
chunk after the shift, before the two converge), and in this environment
that cost isn't clearly repaid, because the miscalibration itself rarely
changes which action is locally best.

## Part 2 — `sign_flip` (`post_shift_scale=-1.0`): where adaptation earns its keep

Nudges reverse direction entirely after the shift — a stale model's
choices become actively counterproductive, not merely suboptimal.

| condition | pre-shift reward | post-shift reward |
|---|---|---|
| never_adapts | -1.8866 | **-15.4894** |
| oracle_adapts | -1.8866 | -2.3595 |
| sliding_window_baseline | -2.1405 | -3.0027 |
| cusum_detects_and_adapts | -2.4639 | **-2.6199** |

**`never_adapts` collapses (post-shift reward ~8x worse than pre-shift)
— its stale model now actively pushes the state away from target.**
`oracle_adapts` recovers almost completely (-2.3595, close to pre-shift
performance). **`cusum_detects_and_adapts` (-2.6199) beats
`sliding_window_baseline` (-3.0027) — the confound-controlled result
this experiment needed**: explicit detection-triggered adaptation isn't
just "using recent data," it measurably outperforms that simpler
heuristic, landing closer to the oracle ceiling than the sliding window
does.

**Detection behavior**: 15/15 seeds detected the true shift (vs. 13/15
under the milder attenuation), with a faster mean latency (23.0 trials
after the true shift, vs. 36.5 under attenuation) — a more severe shift
produces a larger, easier-to-detect residual signal, exactly as CUSUM's
mechanism predicts. **3/15 seeds (20%) showed a false-positive detection
before the true shift, in both severities** — notably higher than
experiment 5's own ~6% baseline false-positive rate under the same
default calibration. Investigated rather than left unexplained: unlike
experiment 5's raw, stationary monitored signal, this experiment's
residual stream comes from a model that itself refits periodically
(`REFIT_INTERVAL=20` trials) — each refit introduces small, genuine
finite-sample jumps in the residual series even during a truly stationary
regime, which CUSUM can occasionally mistake for a real change. This is
a real, reported limitation of monitoring a co-evolving model's own
residuals, not a bug.

## What this establishes

`CUSUMTemporalReasoner` and `LinearDynamicsModel` were each independently
verified before this experiment (experiments 5 and 11 respectively); this
experiment tests only whether *combining* them works as expected, and
found that it does, but conditionally — the value of detection-triggered
adaptation depends on whether the regime shift is severe enough to change
which action is actually best, not merely on whether the model's
coefficients become numerically wrong. Discarding valuable prior data
has a real cost that a mild miscalibration may not repay.

## What this does not establish

- **Only two severities tested** (a 0.4x attenuation and a full sign
  reversal) — the transition between "adaptation doesn't help" and
  "adaptation clearly helps" wasn't mapped as a continuous sweep; where
  exactly it crosses over is unknown.
- **A single, fixed regime-shift trial and a single post-shift window
  length** — how quickly the pre-shift/post-shift reward gap needs to
  close, or whether `never_adapts` would eventually catch up given a
  much longer post-shift window (its mixed-data model keeps absorbing
  more post-shift signal over time), wasn't tested.
- **The elevated false-positive rate (20% vs. experiment 5's ~6%) is
  reported, not fixed** — no attempt was made to recalibrate CUSUM
  specifically for monitoring a periodically-refit model's residuals
  rather than a raw stationary signal.
- **Single dynamics-model architecture (`LinearDynamicsModel`)** — this
  says nothing about whether detection-triggered adaptation composes as
  well with a nonlinear dynamics model (experiment 10's generalization)
  or a multi-step planner (experiment 12).
- **A single sliding-window size** (`SLIDING_WINDOW_SIZE=200`) — not
  swept, so the specific margin by which CUSUM beats it may depend on
  how that hyperparameter is chosen.
