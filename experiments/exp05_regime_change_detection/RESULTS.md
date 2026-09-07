# Experiment 5 results — regime-change detection

Ran: `PYTHONPATH=. python experiments/exp05_regime_change_detection/run.py`.
5 synthetic regimes (means `[0.3, 0.7, 0.3, 0.6, 0.4]`), 60 steps each, 20
seeds per configuration. `CUSUMTemporalReasoner`'s hyperparameters held
**fixed at the class defaults** (`burn_in=30`, `h_sigma=8.0`) across every
sweep — mirroring experiments 1/4's discipline of testing a
once-calibrated detector outside its calibration point, not
recalibrating per condition. See `docs/research-agenda.md` #7b for the
hypothesis and `docs/related-work.md` §9a for the established theory
(Page 1954) this implements.

## Noise sweep: degrades gracefully — unlike experiment 4's trigger

| noise σ | recall | precision | mean detection delay (steps) |
|---|---|---|---|
| 0.01 | 0.988 | 0.970 | 0.00 |
| 0.02 | 0.988 | 0.970 | 0.01 |
| 0.05 | 0.988 | 0.970 | 0.82 |
| 0.10 | 0.988 | 0.970 | 2.02 |
| 0.20 | 0.887 | 0.887 | 5.80 |
| 0.30 | 0.637 | 0.688 | — |
| 0.40 | 0.412 | 0.500 | — |
| 0.50 | 0.237 | 0.467 | — |
| 0.70 | 0.125 | 0.500 | — |

**This is a materially different failure pattern than experiment 4's
binomial trigger, and the reason is mechanistic, not incidental.**
Experiment 4's `min_accuracy=0.85` threshold was a single fixed number,
uninformed by locally observed noise — its false-discovery rate
*exploded* once real noise diverged from the calibration point (0.000 →
0.763 across the same rough noise range). CUSUM instead re-estimates σ
from each burn-in window, so its effective threshold (`h_sigma * σ`)
automatically scales with whatever noise is actually present. Recall and
precision degrade *smoothly* — from ~0.99 down to ~0.89 across a
10x increase in noise (0.01→0.10... →0.20), only crossing into serious
degradation past σ=0.3, roughly 6x the well-behaved regime's noise level.
**This is direct evidence for the fix experiment 4's RESULTS.md named but
didn't build**: "a trigger that adapts its null accuracy to
locally-observed performance rather than a value fixed at design time."
Built here, independently, for a different mechanism — and it behaves
exactly as that suggestion predicted.

Detection delay also degrades gracefully and interpretably: near-zero at
low noise, rising to ~2 steps at σ=0.10 and ~6 steps at σ=0.20 — a real,
expected cost (more samples needed to accumulate enough evidence through
noise), not a discontinuity.

## Regime-length sweep: a sharp, structural threshold at `burn_in`

| regime length | recall | precision |
|---|---|---|
| 10 | **0.000** | n/a (nothing ever detected) |
| 20 | 0.312 | 1.000 |
| 30 (= `burn_in`) | **1.000** | 1.000 |
| 40 | 1.000 | 1.000 |
| 60 | 0.988 | 0.970 |
| 100 | 0.988 | 0.922 |

**This is not a bug — it's the mechanism's defining structural
constraint, and the sharp transition exactly at `regime_length == burn_in`
confirms the mechanism works as designed, not by accident.** A regime
shorter than `burn_in` can never be used to calibrate a stable mean/σ
before the *next* change happens — the burn-in window simply spans the
boundary and blends two regimes into one noisy "calibration," and the
detector has no chance to notice anything. Recall is exactly zero at
`regime_length=10` (regimes a third of `burn_in`), partial at 20
(two-thirds), and jumps to a clean 1.0 the moment regimes reach or exceed
`burn_in`. **This directly couples the detector's minimum usable
regime length to its `burn_in` parameter — state this explicitly as a
precondition, not bury it**: this detector is only meaningful for regimes
expected to persist for at least `burn_in` observations, a real
limitation for any application with fast-changing regimes.

## What this changes going forward

- This is the first genuinely reusable kernel primitive from this
  research program — `CUSUMTemporalReasoner` lives in
  `transintelligence/reasoning/temporal/`, not in `experiments/`, and is
  tested as kernel code (`tests/test_temporal_reasoning.py`), unlike
  experiments 1-4's RL research scripts.
- The self-calibration lesson generalizes: any future detection-triggered
  mechanism in this codebase (including a revisit of experiment 4's
  frame-discovery trigger) should calibrate its threshold from local data
  the way this one does, not fix it at design time.
- **Both boundaries found here are real limits to state plainly, not
  just "future work" filler**: this detector needs regimes to persist for
  at least `burn_in` steps, and its recall/precision degrade past roughly
  6x its well-behaved noise level. Neither invalidates the mechanism —
  they define where it's actually applicable.
- ~~Not yet tested: `regime_segments()`'s per-segment mean accuracy under
  noise~~ — run below. Still not tested: more than one tracked `key`
  simultaneously; non-Gaussian regime transitions (gradual drift rather
  than a step change, which CUSUM is not designed to detect cleanly).

## Per-segment accuracy: a different question than detection recall/precision

Ran: `PYTHONPATH=. python experiments/exp05_regime_change_detection/sweep_segment_accuracy.py`.
The noise sweep above benchmarks `change_points()` — did detection happen
at roughly the right steps. This benchmarks `regime_segments()`'s actual
*output* — are the resulting segment mean estimates any good — which is a
different question a detector could fail at even with decent recall (if
boundaries are off by a few steps) or succeed at despite occasional
missed/spurious detections (if the resulting merged/split segments still
average out close to the truth).

Mean absolute error (MAE) between each state's assigned segment mean and
its true generating regime's mean, against two reference points: a
"no segmentation" floor (one global mean, ignoring regimes entirely) and
a "ground-truth segmentation" ceiling (means computed from the *true*
regime boundaries, not detected ones).

| noise σ | no segmentation (floor) | detected segments | ground-truth segments (ceiling) |
|---|---|---|---|
| 0.01 | 0.1520 | 0.0024 | 0.0011 |
| 0.02 | 0.1521 | 0.0035 | 0.0021 |
| 0.05 | 0.1522 | 0.0106 | 0.0053 |
| 0.10 | 0.1524 | 0.0254 | 0.0106 |
| 0.20 | 0.1527 | 0.0555 | 0.0213 |

**A clean, stable result: detected segmentation stays consistently
~2-2.6x worse than the ground-truth ceiling across every noise level
tested, rather than diverging from it as noise grows.** The ratio
(0.0024/0.0011≈2.2, 0.0035/0.0021≈1.7, 0.0106/0.0053≈2.0,
0.0254/0.0106≈2.4, 0.0555/0.0213≈2.6) drifts only mildly upward — the
detector's imperfections (missed detections, off-by-a-few-steps
boundaries) cost a roughly constant multiplicative penalty, not a
noise-compounding one. And segmentation is dramatically better than not
segmenting at all throughout: 60x+ better at low noise, still ~2.75x
better even at the noisiest level tested (0.1527 / 0.0555). Detection
recall/precision degrading under noise (the sweep above) does not, on
this evidence, translate into segment-mean estimates that are
*disproportionately* worse — a meaningfully more reassuring result than
the recall/precision numbers alone would suggest.

## Temporal comparison via DTW: the motivating case, constructed directly

Implemented `dynamic_time_warp()` (Sakoe & Chiba 1978, basic symmetric
form, no slope constraint) and `CUSUMTemporalReasoner.trajectory_distance()`
on top of it. Ran:
`PYTHONPATH=. python experiments/exp05_regime_change_detection/dtw_comparison.py`
— the canonical motivating case for DTW, built directly rather than
asserted from the literature: three length-30 trajectories, a `template`
(flat, bump to a high value for 10 steps, flat again), a `shifted` version
(the *identical* bump shape, delayed onset by `shift` steps), and a
`different` version (a smaller/different-height bump at the *same*
position as `template` — genuinely a different shape, not a warped one).
The correct ranking, by shape, is `template ~ shifted << different`.

| shift | DTW(shifted) | DTW(different) | DTW ranks correctly | naive(shifted) | naive(different) | naive ranks correctly |
|---|---|---|---|---|---|---|
| 0 | 0.000 | 2.000 | ✓ | 0.000 | 2.000 | ✓ |
| 1 | 0.000 | 2.000 | ✓ | 1.200 | 2.000 | ✓ |
| 2 | 0.000 | 2.000 | ✓ | 2.400 | 2.000 | **✗** |
| 3-9 | 0.000 | 2.000 | ✓ | 3.6-10.8 | 2.000 | **✗** |
| 10-15 | 6.000 | 2.000 | ✗ (see below) | 9.0-12.0 | 2.000 | ✗ |

**Naive same-index comparison starts ranking backwards at shift=2 — a
gap of only two steps is enough to fool it — and stays wrong through
shift=9, an 8-value window where DTW is unambiguously correct and naive
is unambiguously not.** DTW recognizes the shifted bump as a perfect
match (distance 0.0) regardless of delay, exactly the property it's
designed to have; naive comparison's distance grows roughly linearly with
the shift because it's comparing the bump against flat baseline at
increasingly misaligned indices, and by shift=5 it's already ranking the
*wrong-shaped* trajectory as more than 3x closer than the *identical*
one.

**DTW's own apparent "failure" at shift≥10 is a construction artifact,
not a real limitation, and worth being precise about rather than letting
the table imply otherwise:** at `shift=10`, `BUMP_START + shift + BUMP_LEN
== LENGTH` exactly — the trailing flat segment after the bump runs out
entirely, so `shifted` at that point is genuinely a different waveform
(missing the "return to baseline" tail), not merely a delayed version of
`template`. DTW correctly detects that this is now a real shape
difference; it isn't failing to handle a large shift, the fixed-length
test harness stopped being able to represent a pure shift beyond that
point.

## Full updated status

Two follow-ups to the original noise/regime-length sweeps, both
confirming or extending the original finding rather than overturning it:
segmentation quality degrades gracefully in proportion to detection
quality (not catastrophically), and the DTW mechanism this module now
also provides does exactly what the established literature says it
should, demonstrated with a constructed counterexample against naive
comparison rather than taken on faith.
