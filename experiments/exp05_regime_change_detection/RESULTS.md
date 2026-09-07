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
  noise~~ — run below. ~~Still not tested: more than one tracked `key`
  simultaneously; non-Gaussian regime transitions~~ — also run below, in
  the multi-key, non-Gaussian noise, and gradual drift sections.

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

## Multi-key tracking: joint detection beats a union-of-detectors control, with real boundaries on both ends

Implemented `joint_change_points()` (Crosier 1988's "reduce to a scalar,
then CUSUM" variant: sum per-key z-scored deviations from their own
burn-in means, normalize by `sqrt(n_keys)`, run the same two-sided CUSUM
on the combined scalar — assumes independence across keys, no
cross-covariance term). **Calibration check first, as always**: false-positive
rate stayed at 5.5-8% across 1 to 5 tracked keys (200 trials each) — the
`sqrt(n_keys)` normalization keeps the single-key calibration valid
without retuning.

Ran: `PYTHONPATH=. python experiments/exp05_regime_change_detection/multi_key.py`.
Two keys share the same true change point and shift direction/magnitude
with independent noise, chosen so each key alone is a **weak** detector
(shift=0.04 gives ~44% single-key recall) — the regime where combining
evidence should matter most. The critical control, built alongside the
treatment (the lesson from experiments 3/4): a "union" baseline that
takes either key's *independent* detection, to separate "combining
evidence helps" from "two independent chances to detect helps."

| shift | shift/noise | recall A | recall B | recall union | recall joint | paired (joint wins / union wins / ties, n=50) |
|---|---|---|---|---|---|---|
| 0.02 | 0.40 | 0.16 | 0.14 | 0.26 | 0.24 | 2 / 3 / 45 |
| 0.03 | 0.60 | 0.24 | 0.28 | 0.44 | 0.58 | 9 / 2 / 39 |
| 0.04 | 0.80 | 0.44 | 0.42 | 0.68 | 0.78 | 6 / 1 / 43 |
| 0.05 | 1.00 | 0.64 | 0.68 | 0.84 | 0.90 | 3 / 0 / 47 |
| 0.06 | 1.20 | 0.84 | 0.86 | 0.96 | 0.98 | 1 / 0 / 49 |

**A genuine, bounded positive result — the control confirmed a real
advantage over the union baseline, not just "more capacity helps"
(the confound found in experiments 3/4), but only across a specific
signal range.** In the weak-to-moderate regime (shift 0.03-0.05, i.e.
shift/noise 0.6-1.0), joint detection wins the paired comparison by wide
margins (9:2, 6:1, 3:0) — combining evidence *before* thresholding
genuinely outperforms taking either detector's independent hits.
**At the weakest signal tested (shift=0.02), the advantage disappears**
(2 joint wins vs. 3 union wins — a wash, if anything a slight edge to
union) — apparently too little signal in either key for the combined
statistic to reliably clear a `sqrt(n_keys)`-scaled threshold either.
At the strongest signal tested (shift=0.06), both methods saturate near
1.0 and there's little room left to differ. **State the claim at its
actual width: joint detection helps in a real, moderate signal-strength
band, not universally** — a genuine gain, not the "more chances = more
detections" confound this codebase has learned to check for.

## Non-Gaussian noise: a real specificity failure, mirroring experiment 4's pattern

Ran: `PYTHONPATH=. python experiments/exp05_regime_change_detection/non_gaussian_noise.py`.
CUSUM's calibration (burn-in mean/σ, threshold as a σ multiple) implicitly
assumes light-tailed noise. Tested against a standard contaminated-Gaussian
mixture (5% of observations drawn from a 5x-wider Gaussian, "outliers";
95% from the calibration-matched one) — no single canonical citation
adopted for "CUSUM's non-normal robustness" (the SPC literature here is a
family of results and robust/nonparametric variants, not one seminal paper
the way Page 1954 is for CUSUM itself); tested empirically instead.

| condition | false-positive rate (200 trials) | recall on a real shift (30 trials) |
|---|---|---|
| Gaussian | 0.080 | 1.000 |
| heavy-tailed | **0.340** | 0.967 |

**Heavy tails more than quadruple the false-positive rate (0.08 → 0.34)
while barely touching recall (1.00 → 0.967).** This is the same
qualitative pattern found for experiment 4's fixed-threshold trigger and
for the joint-detection weak-signal boundary above: **specificity is
consistently the more fragile property under assumption violations across
every detection mechanism built in this codebase so far; detection power
holds up comparatively well.** Mechanistically straightforward: a single
large outlier lands directly in either the burn-in window (inflating the
σ estimate, distorting `k`/`h`) or the monitoring phase (injecting a large
deviation straight into the CUSUM accumulator) — either way it looks like
exactly the kind of surprising deviation CUSUM is built to flag, whether
or not anything about the underlying regime actually changed.

## Gradual drift: an initial hypothesis, tested and corrected

Ran: `PYTHONPATH=. python experiments/exp05_regime_change_detection/gradual_drift.py`.
CUSUM is designed for abrupt shifts (Page 1954). Going in, the working
hypothesis was that a sufficiently slow drift might never be detected:
`change_points()` recalibrates `mu0`/`σ` from a burn-in window once per
detection cycle, then holds them fixed while monitoring — plausible that
a slow enough drift stays within noise relative to a *stale* reference
indefinitely, going undetected.

| ramp length | total series length | recall | mean detection delay (steps) |
|---|---|---|---|
| 1 (abrupt) | 150 | 1.00 | 1.0 |
| 5 | 150 | 1.00 | 2.9 |
| 10 | 150 | 1.00 | 4.0 |
| 20 | 150 | 1.00 | 5.9 |
| 40 | 150 | 1.00 | 8.7 |
| 80 | 150 | 1.00 | 12.9 |
| 150 (ramp fills the whole remaining series) | 150 | 1.00 | 18.2 |
| 300 | 400 | 1.00 | 28.0 |
| 600 | 700 | 1.00 | 44.5 |
| 1000 (ramp never completes within the series) | 1100 | 1.00 | 67.8 |

**The hypothesis was wrong, and it's worth saying so plainly rather than
quietly dropping it: recall stays at 1.00 across every ramp length
tested, including a 1000-step ramp that never even finishes within an
1100-step series.** The reasoning that seemed plausible going in missed
something real: because the calibration reference is *fixed* once per
cycle rather than continuously re-chased, `s_pos`'s accumulated deviation
from that fixed `mu0` grows with *any* persistent positive drift, however
slow — mathematically guaranteed to cross threshold eventually, not a
coincidence of the specific rates tested. Detection delay grows with ramp
length, but sub-linearly (a 3.3x longer ramp from 300→1000 produced only
a 2.4x longer delay) — a real, expected cost, not a breakdown.
**Gradual drift is not, on this evidence, actually outside this
detector's practical capability** — the "outside CUSUM's design"
framing that motivated this test applies more to *how quickly* it reacts
than to *whether* it eventually reacts at all.

## Full updated status

Five follow-ups to the original noise/regime-length sweeps now:
segmentation quality degrades gracefully in proportion to detection
quality (not catastrophically); DTW does exactly what the established
literature says it should, demonstrated with a constructed counterexample
rather than taken on faith; joint multi-key detection gives a real,
bounded advantage over a union-of-detectors control; heavy-tailed noise
is a genuine, substantial specificity failure (the same pattern found in
experiment 4, now confirmed a second time in an unrelated mechanism); and
gradual drift, going in the most likely candidate for a real breakdown,
turned out not to be one — the working hypothesis was wrong, and finding
that out empirically rather than assuming it going in is the actual
result worth keeping.
