# Experiment 4 results — frame discovery

Ran: `PYTHONPATH=. python experiments/exp04_frame_discovery/run.py`. Reuses
experiment 1's exact held-out-frame setup (`ALL_FRAMES`/`TRAIN_FRAMES`,
`switch_period=40±10`, `noise_sigma=0.05`, 10 seeds, 3000 steps) so these
numbers are directly comparable to
[experiments/exp01_frame_conditioning/RESULTS.md](../exp01_frame_conditioning/RESULTS.md).
See `docs/research-agenda.md` #7a for the design and `docs/related-work.md`
§8a for the established theory this borrows a crude heuristic from.

Design as scoped, with two changes made in review before implementing (see
the conversation this came from): the trigger's null accuracy was
calibrated at `p=0.85` (a margin below experiment 1's observed 0.906
steady-state accuracy, not the internal `error_rate=0.05` naively), and
the trigger checks non-overlapping (tumbling) windows once every 40 steps
rather than a sliding window every step, avoiding a repeated/correlated
hypothesis-testing problem.

## Result — a real, substantial, structure-specific effect

| agent | held-out acc. BEFORE first discovery | held-out acc. AFTER first discovery | seeds that discovered |
|---|---|---|---|
| static (`rf_aware`, experiment 1 baseline) | 0.742 (whole run) | never discovers | 0/10 |
| `discovering_rf` | 0.701 | **0.914** | 10/10 |
| `random_discovery` | 0.701 | 0.786 | 10/10 |

Mean false-discovery rate (fraction of triggers firing during a
known-frame window, not the held-out one): **0.000 for both agents**,
across 10 total triggers (`discovering_rf`) and 22 total triggers
(`random_discovery`). Mean final candidate-list size: 4.00 for
`discovering_rf` (started at 3, i.e. exactly one discovery per seed, every
seed), 4.50 for `random_discovery` (some seeds needed a second attempt).

## This is the cleanest result of the four experiments — and it was derived, not tuned

Unlike experiments 1 and 3, where every control shrank the headline
number, here the confound control *confirms* a real effect rather than
mostly explaining it away: `discovering_rf`'s post-discovery accuracy
(0.914) is **+0.128 above `random_discovery`'s** (0.786), and both are
well above the pre-discovery baseline (0.701) — so there is a genuine
"any extra candidate helps a little" effect (0.701→0.786, consistent with
the confound experiment 3 found), *and* a much larger effect specifically
from fitting the *right* frame (0.786→0.914).

**Checked directly, not just inferred from the accuracy numbers**: the
frames `discovering_rf` actually fits are close to the true held-out frame
(`baseline=0.7`, `direction=lower_is_better`) — spot-checked across 5
seeds, fitted baselines were `0.70, 0.70, 0.80, 0.75, 0.70`, all with the
correct direction. This is why `discovering_rf` needed only one discovery
per seed (final frame count exactly 4.00, no variance) while
`random_discovery` sometimes needed a second attempt (4.50 on average) —
a random frame doesn't reliably close the gap on the first try, a fitted
one does.

The hyperparameters (window=40, `min_accuracy=0.85`, `alpha=0.01`) were
derived analytically before running anything — an exact binomial tail
probability gave a predicted false-positive rate of ~4.6e-5/window and
~21% per-window detection power at the true held-out accuracy (0.742),
which compounds to a ~98.6% chance of at least one detection somewhere in
a 3000-step run given how often the held-out regime recurs. The observed
result (10/10 seeds discovered, 0 false positives) matches that prediction
almost exactly — this was not tuned to produce a good-looking number after
the fact.

## What this does not establish

- **Narrow discovery mechanism.** The fit is a grid search over a single
  `(baseline, direction)` pair — it works because the true underlying
  regime family is exactly that two-parameter family. It would not
  generalize to a richer frame structure (multiple properties, composed
  frames) without a materially more expensive search or a different
  mechanism (see the Dirichlet-process-style upgrade path in
  `docs/related-work.md` §8a).
- ~~**Single held-out regime.**~~ Run below (two simultaneously missing) —
  and it surfaced a more precise, more important characterization of the
  mechanism than the single-regime result alone could: detectability
  depends on behavioral distinguishability from current belief, not
  parameter distance. Still untested: a continuously drifting
  (non-discrete) regime, which the fixed-family fit can't represent at
  all regardless of detectability.
- **False-discovery rate is a strong but thin result.** Zero false
  triggers across ~75 windows/seed × 10 seeds is consistent with the
  ~4.6e-5/window design prediction, but the absolute count of trigger
  events is small (10 and 22 respectively) — a much longer run would be
  needed to bound the false-positive rate tightly rather than just
  observe zero of it.
- ~~**No sweep.**~~ Run below — and it found a sharper failure mode than
  predicted.

## Noise sweep: specificity collapses faster than usefulness does

Ran: `PYTHONPATH=. python experiments/exp04_frame_discovery/sweep_noise.py`,
same noise levels as Experiment 2, `min_accuracy=0.85` held **fixed**
across the sweep on purpose — the question is whether a trigger calibrated
at one noise level (0.05) degrades gracefully outside it, not whether some
per-level recalibration could be found that works everywhere.

| sigma | agent | discovery rate | acc before | acc after | total triggers | false-discovery rate |
|---|---|---|---|---|---|---|
| 0.00 | discovering_rf | 0.90 | 0.699 | 0.953 | 9 | 0.000 |
| 0.00 | random_discovery | 0.90 | 0.699 | 0.783 | 21 | 0.000 |
| 0.02 | discovering_rf | 0.90 | 0.699 | 0.918 | 9 | 0.000 |
| 0.02 | random_discovery | 0.90 | 0.699 | 0.782 | 21 | 0.000 |
| 0.05 | discovering_rf | 1.00 | 0.701 | 0.914 | 10 | 0.000 |
| 0.05 | random_discovery | 1.00 | 0.701 | 0.786 | 22 | 0.000 |
| 0.10 | discovering_rf | 1.00 | 0.699 | 0.853 | 12 | **0.250** |
| 0.10 | random_discovery | 1.00 | 0.699 | 0.778 | 34 | 0.118 |
| 0.20 | discovering_rf | 1.00 | 0.755 | 0.751 | 93 | **0.645** |
| 0.20 | random_discovery | 1.00 | 0.755 | 0.722 | 117 | 0.564 |
| 0.40 | discovering_rf | 1.00 | 0.590 | 0.642 | 465 | **0.763** |
| 0.40 | random_discovery | 1.00 | 0.590 | 0.613 | 493 | 0.728 |

**At and below the calibrated noise level (σ≤0.05), the result holds up
and even improves at lower noise**: false-discovery rate is exactly 0.000
in all three low-noise rows, and `discovering_rf`'s post-discovery
accuracy (0.918-0.953) is meaningfully higher at lower noise, as expected.

**But the fixed threshold doesn't degrade gracefully — it breaks sharply,
and the dominant failure mode is not the one predicted.** The original
scoping predicted "higher noise would lower [detection] power further,"
i.e. a fewer-discoveries problem. That's not what happened — discovery
*rate* actually stays at or near 1.00 throughout. **What actually breaks
is specificity**: false-discovery rate jumps from 0.000 (σ≤0.05) to 0.250
(σ=0.10) to 0.645 (σ=0.20) to 0.763 (σ=0.40). At σ=0.20, `discovering_rf`
triggered 93 times across 10 seeds (vs. 10 at the calibrated level) and
the *majority* of those triggers fired during ordinary known-frame
windows, not the held-out regime — because at higher noise, even a
correctly-identified known frame's accuracy dips below the fixed 0.85
threshold from noise alone often enough to look statistically
"surprising" under a null calibrated for a quieter environment.

**This is consequential, not cosmetic: once false triggers dominate, the
mechanism stops working even at what it's nominally supposed to be
doing.** By σ=0.20, `discovering_rf`'s post-discovery accuracy (0.751) is
statistically indistinguishable from its pre-discovery accuracy (0.755) —
the discovery mechanism has stopped helping entirely, at almost exactly
the noise level (σ≈0.2) where Experiment 1 separately found the underlying
structural advantage of `rf_aware` over flat/opaque baselines also
vanishes. The candidate list fills with noise-driven junk frames (the
`max_frames` cap is reached almost immediately once triggers start firing
dozens of times per seed), leaving no room left for a frame that would
actually help, and diluting belief across mostly-useless candidates. The
`discovering_rf` vs. `random_discovery` gap also collapses at high noise
(0.642 vs. 0.613 at σ=0.4, down from 0.128 at σ=0.05) — once the mechanism
is dominated by false triggers, whether the fitted value is any good
barely matters anymore, since most of what's being fit is noise, not
signal.

**Scope of the claim, restated precisely**: frame discovery via this
fixed-threshold binomial trigger works within the same bounded noise
regime experiment 1 found for the underlying structural claim (roughly
σ≲0.1), and its failure mode outside that regime is a genuine, actively
harmful specificity collapse — not just reduced usefulness. A trigger that
adapts its null accuracy to locally-observed noise (rather than a value
fixed at design time) is the natural fix, not implemented here.

## Two simultaneously-missing regimes: detectability depends on behavior, not parameter distance

Ran: `PYTHONPATH=. python experiments/exp04_frame_discovery/two_missing_regimes.py`.
5 total frames, 3 known, 2 held out — chosen to be mutually distinct in
parameter space (different baselines *and* different directions from each
other, not just from the known set): `f4` (baseline=0.7,
`lower_is_better`, the same frame used in the single-held-out
experiments) and `f5` (baseline=0.2, `higher_is_better`). 4500 steps
(longer than the single-regime run, to give two discoveries room), 10
seeds, otherwise identical configuration.

| held-out frame | discovered | acc. before match | acc. after match |
|---|---|---|---|
| `f4` (0.7, lower_is_better) | **9/10 seeds** | 0.699 | 0.900 |
| `f5` (0.2, higher_is_better) | **2/10 seeds** | 0.862 | 0.904 |

Both regimes discovered in the same seed: 2/10. False-discovery rate:
0/12 = 0.000 (specificity held up, as expected at this noise level per
the sweep above).

**This is not "discovery generalizes to two regimes" — it's a sharp,
informative asymmetry that reveals what the trigger actually detects.**
`f4` replicates the single-held-out result closely (9/10 seeds, 0.699→0.900,
matching the original experiment's 0.701→0.914). `f5` is discovered in
only 2/10 seeds, **despite being parametrically just as distinct from the
known frames as `f4` is** (different baseline, different direction from
every known frame). The reason isn't a weaker signal or bad luck — it's
that the trigger only ever reacts to *observed reward*, not to parameter
distance, and `f5`'s true rule happens to produce nearly the same
predictions as the known frames already do.

Checked directly: simulating the known-frames-only belief's predictions
against each held-out frame's true rule across the raw-value range
`[0,1]` gives an agreement rate of **40% with `f4`'s true rule** (the
known frames are usually wrong when `f4` is active — a clear,
detectable failure) versus **90% with `f5`'s true rule** (the known
frames coincidentally agree with `f5`'s rule most of the time, since two
of the three known frames already share `f5`'s `higher_is_better`
direction and `f5`'s low baseline of 0.2 makes its rule's predictions
resemble theirs across most of the raw-value range). With ~90% behavioral
agreement, `f5`-active windows rarely dip persistently below the 0.85
trigger threshold — there's usually nothing to detect, even though the
agent's belief is, strictly, still wrong.

**This is a genuine, non-obvious limitation, and a more precise
characterization of the mechanism than "it discovers novel regimes":
it discovers regimes that are *behaviorally* distinguishable from current
belief under the reward signal actually available, not regimes that are
merely parametrically distinct.** A regime that happens to blend in with
existing predictions can go undetected indefinitely, not because the
mechanism failed, but because there was never a large enough visible
symptom to react to. This is a real, sound limitation of any
detection-triggered-by-observed-performance approach (the same is true of
the established changepoint-detection and open-set-recognition literature
this borrows from, docs/related-work.md §8a — detectability there is
also fundamentally a function of the observation distribution, not
parameter distance) — not a defect specific to this crude implementation.

## What this changes going forward

- The held-out-frame limitation found in experiment 1 is **not
  fundamental** — a genuinely crude, hand-designed discovery mechanism
  recovers most of the lost accuracy (0.742 static → 0.914 after
  discovery, actually exceeding the static held-out baseline, though
  still short of `true_oracle`'s ceiling from experiment 1). This is
  worth stating plainly as a positive result for the overall research
  program, distinct from the caveats attached to experiments 1 and 3.
- The one open question this raises: `discovering_rf`'s post-discovery
  accuracy (0.914) is close to but not quite `rf_aware`'s accuracy on
  *known* frames in experiment 1 (0.906-0.923 depending on which number
  is compared) — worth a direct comparison in a future pass rather than
  treating 0.914 as self-evidently "as good as knowing the frame from the
  start."
- **The noise sweep changed the honest scope of the headline claim.** "Frame
  discovery recovers most of the lost accuracy" is true within roughly the
  same noise regime experiment 1's structural claim holds (σ≲0.1), and
  degrades into an actively harmful specificity collapse outside it, not a
  graceful decline. Any future write-up of this result must carry that
  qualifier — the pattern of "the number holds up less well than the
  single-configuration headline suggests" from experiments 1 and 3 turned
  out to apply here too, just in a different way (specificity failure
  rather than a shrinking effect size).
- The natural fix — a trigger that adapts its null accuracy to
  locally-observed performance rather than a value fixed at design time —
  is not implemented. This is the concrete next step if this mechanism is
  pursued further, not a bigger version of the current sweep.
- **The two-missing-regimes test is the most important addition to how
  this result should be described.** The honest characterization is no
  longer "detects and fits novel regimes" — it's "detects and fits novel
  regimes whose predictions are actually distinguishable, under the
  reward signal, from what's currently believed." A regime that happens
  to blend into existing predictions can go undetected indefinitely, and
  this isn't fixable by tuning the trigger threshold — it's a structural
  property of any performance-triggered detector, shared with the
  established literature this borrows from. Any future extension (more
  frames, real-world data) needs to account for this rather than assume
  "distinct parameters" implies "detectable."
