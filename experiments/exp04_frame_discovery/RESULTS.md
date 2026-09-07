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
- **Single held-out regime.** Only one frame was ever missing from the
  known set. Untested: what happens with two or more simultaneously
  missing regimes, or a continuously drifting (non-discrete) regime the
  fixed-family fit can't represent at all.
- **False-discovery rate is a strong but thin result.** Zero false
  triggers across ~75 windows/seed × 10 seeds is consistent with the
  ~4.6e-5/window design prediction, but the absolute count of trigger
  events is small (10 and 22 respectively) — a much longer run would be
  needed to bound the false-positive rate tightly rather than just
  observe zero of it.
- **No sweep.** Unlike experiments 1 and 2, this wasn't run across
  multiple noise levels or switch periods. The ~21% per-window power
  number is specific to `noise_sigma=0.05`; higher noise would lower
  power further (a smaller true accuracy gap from the 0.85 null) and
  likely require a longer run or a larger window to compensate.

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
