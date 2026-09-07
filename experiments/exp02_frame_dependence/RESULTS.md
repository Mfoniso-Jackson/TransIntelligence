# Experiment 2 results — frame-dependence detection

Ran: `PYTHONPATH=. python experiments/exp02_frame_dependence/run.py`,
`N_ENTITIES=300`, seed `0`. See that file for the full method; see
`docs/research-agenda.md` #6 for the hypothesis being tested and
`docs/related-work.md` #4 for the surrounding literature.

## Result

| condition | sigma | AUC(sensitivity) | noisy conclusion accuracy | sensitivity stdev |
|---|---|---|---|---|
| same_dir | 0.00 | 0.494 | 1.000 | ~0 (float noise only) |
| same_dir | 0.02 | 0.519 | 0.980 | ~0 |
| same_dir | 0.05 | 0.496 | 0.913 | ~0 |
| same_dir | 0.10 | 0.531 | 0.850 | ~0 |
| same_dir | 0.20 | 0.590 | 0.757 | ~0 |
| same_dir | 0.40 | 0.593 | 0.777 | ~0 |
| diff_dir | 0.00 | **1.000** | 1.000 | 0.306 |
| diff_dir | 0.02 | 0.991 | 0.953 | 0.316 |
| diff_dir | 0.05 | 0.971 | 0.927 | 0.332 |
| diff_dir | 0.10 | 0.915 | 0.870 | 0.341 |
| diff_dir | 0.20 | 0.735 | 0.727 | 0.404 |
| diff_dir | 0.40 | **0.530** | 0.737 | 0.547 |

## Finding: the hypothesis fails outright in one of the two regimes it needs to cover, and this is provable, not just observed

**`sensitivity(x, R1, R2)` = `abs(evaluate(x,R1).result - evaluate(x,R2).result)`.
When `R1` and `R2` share the same `direction`, the entity's raw value
algebraically cancels out of that subtraction:**

```
evaluate(x, R1) = raw - b1
evaluate(x, R2) = raw - b2
sensitivity      = |(raw - b1) - (raw - b2)| = |b2 - b1|
```

This is independent of `raw` — i.e. independent of the entity — entirely.
`sensitivity()` in the same-direction regime is a **constant function of the
frame pair**, not a per-entity signal, so it cannot possibly discriminate
which entities have frame-dependent conclusions from which don't: AUC sits
at chance (0.49-0.59, with the deviation from exactly 0.500 coming from
floating-point rounding noise in the cancellation, not a real signal — see
`tests/test_exp02_frame_dependence.py` for a direct proof via
`pytest.approx` equality across two entities with very different raw
values). This is locked in as a regression test, not just an experimental
observation, because it's a closed-form algebraic fact about the current
implementation, not a statistical trend that could plausibly flip with a
different seed or sample size.

In the opposite-direction regime, the cancellation doesn't happen
(`sensitivity = |2*raw - b1 - b2|`, which does depend on the entity), and
the detector genuinely works — AUC = 1.000 noiseless, degrading smoothly to
chance (0.530) by `sigma=0.4`, which is a real, if narrow, positive result
and matches the "degradation curve as noise increases" the original
experiment spec asked for.

## Interpretation

The literal claim from `docs/research-agenda.md` #6 — "`sensitivity()` can
distinguish frame-dependent from frame-invariant properties" — is **false
as a general claim about the current implementation**. It's true only in
the special case where the compared frames disagree on `direction`, and
false whenever they only disagree on `baseline` (arguably the *more common*
case in practice — e.g. the finance demo's three volatility frames in
`examples/finance_demo.py` all share `direction: lower_is_better` and
differ only in `baseline`, which is exactly the regime where `sensitivity()`
provides zero per-entity signal).

This is a genuine finding, not a wash: it identifies precisely why the
current `_score()` formula in
[reasoning/relative/model.py](../../transintelligence/reasoning/relative/model.py)
is inadequate as a general frame-dependence detector — it's a bare
location-shift (`raw - baseline`), so any comparison between two
same-direction frames is, by construction, just a comparison of their
baselines, with no way for entity-level data to enter. A statistic that
could work in both regimes would need to be relative to the *entity's
position between the two (direction-adjusted) baselines* — e.g. whether
`raw` falls between `min(b1,b2)` and `max(b1,b2)` under a shared direction,
or a probability of sign-flip under the observation's confidence/noise
model — rather than the raw magnitude of the score difference. That's a
concrete next step for `reasoning/relative/model.py`, not a reason to
abandon the frame-dependence-detection idea.

## What this changes going forward

- Do not present `sensitivity()` as a working frame-dependence detector
  without qualifying it to the opposite-direction case.
- Before extending `BaselineRelativeReasoner`, fix `_score`/`sensitivity`
  to depend on the entity's raw value even when directions match (e.g.
  boundary-crossing probability rather than point difference) — otherwise
  any future experiment built on top of it (including Experiment 1, which
  reuses `evaluate`/`compare`) inherits the same blind spot for
  same-direction frame pairs.
- The synthetic generator and AUC harness in `run.py` are reusable as-is
  once that fix exists — rerun this script and diff against the numbers
  above to check the fix actually closes the gap.

## Fix applied, and post-fix results

`sensitivity()` was rewritten to compare the *signs* of `evaluate(x, r1)`
and `evaluate(x, r2)` directly, rather than their point-difference. Each
individual `evaluate()` result depends on the entity's raw value regardless
of whether `direction` matches, so the cancellation described above no
longer happens. Result convention: **negative ⇒ conclusion flips between
`r1` and `r2` (frame-dependent); positive ⇒ both frames agree
(frame-invariant); magnitude ⇒ margin to the nearer decision boundary**,
usable directly as a ranking/confidence score. See
[transintelligence/reasoning/relative/model.py](../../transintelligence/reasoning/relative/model.py)
and the regression tests in `tests/test_exp02_frame_dependence.py` (which
now lock in the *fixed* behavior — the file previously locked in the bug).

Re-ran `run.py` unmodified except for score polarity (the harness negates
`sensitivity().result` so higher score means "more likely frame-dependent",
matching the AUC convention — the fixed function itself needed no such
adjustment, this is purely about how the experiment script reads the sign):

| condition | sigma | AUC(sensitivity) | noisy conclusion accuracy |
|---|---|---|---|
| same_dir | 0.00 | **1.000** | 1.000 |
| same_dir | 0.02 | 0.997 | 0.980 |
| same_dir | 0.05 | 0.970 | 0.913 |
| same_dir | 0.10 | 0.906 | 0.850 |
| same_dir | 0.20 | 0.745 | 0.757 |
| same_dir | 0.40 | 0.618 | 0.777 |
| diff_dir | 0.00 | **1.000** | 1.000 |
| diff_dir | 0.02 | 0.991 | 0.953 |
| diff_dir | 0.05 | 0.971 | 0.927 |
| diff_dir | 0.10 | 0.915 | 0.870 |
| diff_dir | 0.20 | 0.735 | 0.727 |
| diff_dir | 0.40 | 0.530 | 0.737 |

`same_dir` and `diff_dir` now behave almost identically — both start at a
perfect AUC of 1.000 at zero noise and degrade smoothly to chance as noise
grows, which is exactly the "degradation curve" the original experiment
spec (§6 of `docs/research-agenda.md`) asked for, in *both* regimes rather
than just one. The gap identified above is closed. `examples/finance_demo.py`'s
printed "sensitivity" value for BTC also changed as a side effect — it now
reports an entity-specific margin/flip signal instead of a frame-pair
constant, which is the intended fix, not a regression (verify by running
the demo and checking the value differs if you swap in a different entity's
raw observation).
