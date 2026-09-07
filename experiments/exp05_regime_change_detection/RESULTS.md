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
- Not yet tested: `regime_segments()`'s per-segment mean accuracy under
  noise (only `change_points()` was benchmarked directly here); more than
  one tracked `key` simultaneously; non-Gaussian regime transitions
  (gradual drift rather than a step change, which CUSUM is not designed
  to detect cleanly).
