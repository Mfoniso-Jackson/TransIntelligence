# Experiment 1 results — explicit frame structure vs. flat, learned-embedding, and oracle baselines

Ran: `PYTHONPATH=. python experiments/exp01_frame_conditioning/run.py`.
4 candidate frames (mixed baselines and directions — see `FRAMES` in
`run.py`), 3000 steps, switch every 40±10 steps, observation noise
σ=0.05, 10 seeds. See `docs/research-agenda.md` #5 for the hypothesis and
`docs/related-work.md` #2 for why condition A alone doesn't earn the full
claim.

## Result — all five conditions (final, after all three fixes below)

| agent | overall acc. | stdev (across seeds) | recovery acc. (steps 1-10 post-switch) | steady acc. (steps 30-40 post-switch) | belief in true frame (steady) |
|---|---|---|---|---|---|
| flat | 0.700 | 0.020 | 0.586 | 0.709 | n/a |
| learned_embedding | 0.837 | 0.013 | 0.702 | 0.834 | n/a |
| rf_aware | 0.918 | 0.005 | 0.780 | 0.906 | 0.817 |
| flat_oracle | 0.886 | 0.006 | 0.882 | 0.899 | n/a |
| true_oracle | **0.961** | 0.002 | **0.962** | **0.963** | n/a |

Clean monotonic ordering: `flat` (0.700) < `learned_embedding` (0.837) <
`flat_oracle` (0.886) < `rf_aware` (0.918) < `true_oracle` (0.961).

Paired, per-seed (same env/seed for both agents in each pair):

```
overall_acc(rf_aware) - overall_acc(flat):               mean=+0.218  stdev=0.016  wins=10/10 seeds
overall_acc(rf_aware) - overall_acc(learned_embedding):   mean=+0.080  stdev=0.010  wins=10/10 seeds
overall_acc(learned_embedding) - overall_acc(flat):       mean=+0.137  stdev=0.016  wins=10/10 seeds
overall_acc(rf_aware) - overall_acc(flat_oracle):         mean=+0.032  stdev=0.007  wins=10/10 seeds
overall_acc(true_oracle) - overall_acc(rf_aware):         mean=+0.043  stdev=0.004  wins=10/10 seeds
```

(The sections below document the three fixes that got here, in order:
random initialization to break a symmetry bug, hard responsibility
assignment to fix an underperformance-vs-flat problem, then a proper
logistic-regression gradient step in place of the perceptron mistake-rule
to close more of the gap to `rf_aware`. All three were real findings at
the time, not detours.)

## A methodological bug worth reporting on its own: naive symmetric multi-hypothesis tracking degenerates to a flat rule

The first version of `LearnedEmbeddingAgent` initialized all `n_slots`
linear-rule "experts" at exactly `(w, b) = (0, 0)` and updated each slot's
weights by `lr * belief[slot] * target * raw`. That run produced results
for `learned_embedding` that were **numerically identical to `flat`** —
same overall accuracy (0.700), same stdev (0.020), same recovery (0.586),
same steady-state accuracy (0.709), to three decimal places, across all
10 seeds.

That's not a coincidence, it's a real degeneracy, provable directly: since
`sign(c·x) == sign(x)` for any `c > 0`, and every slot starts at the exact
same point and receives an update that only ever scales the *same* shared
direction by the *same* shared belief weight (because belief itself starts
uniform and — since every slot always predicts identically to every other
slot, by induction — never has any signal to differentiate on), all
`n_slots` slots stay proportionally identical to each other forever. The
whole ensemble collapses into a single effective rule, scaled by a smaller
learning rate (`lr / n_slots`), which produces exactly the same prediction
sequence as `flat` because predictions only depend on the *sign* of the
linear score, and scaling by a positive constant never changes a sign.

Fixed by initializing each slot with small random `(w, b)` (`N(0, 0.05)`,
seeded per-run for reproducibility) instead of zero, which breaks the
symmetry so slots can actually specialize. Regression-locked in
`tests/test_exp01_frame_conditioning.py::test_learned_embedding_slots_break_symmetry`.
This is a known failure mode in mixture models generally (symmetric
initialization + a symmetric update rule never breaks symmetry on its
own) — worth remembering before building any other multi-hypothesis
tracker in this codebase.

## First fix wasn't enough on its own: soft-weighted updates underperformed flat

After the random-init fix alone, `learned_embedding` (0.632) was **worse**
than `flat` (0.700), losing in 9 of 10 seeds, with over twice the variance
(0.050 vs. 0.020). This did not contradict the multi-task RL literature
cited in `related-work.md` §2 (arXiv:2102.06177, arXiv:2207.02249) — those
use encoder-decoder architectures trained on reward/dynamics prediction, a
materially stronger mechanism than a soft-EM mixture of 4 linear experts.
The likely cause: with ~40 steps between switches and 4 slots competing
for a small, noisy reward signal, the *soft* (belief-weighted) update
spread learning across all 4 slots every step instead of letting one
commit — slower to specialize than a single rule that commits fully to
whatever it's currently seeing.

## Second fix: hard (argmax) responsibility assignment for weight updates

Changed the weight-update rule only (the *belief* used for prediction and
for choosing which slot to update stayed the same soft Bayesian filter as
`RFAwareAgent`): each step, only the single currently-highest-belief slot
receives the full-learning-rate update, rather than all `n_slots` slots
receiving a belief-weighted fraction of it. This is the standard hard-EM
fix for exactly this failure mode (soft/fractional assignment dilutes
specialization when slots are competing for a scarce, noisy signal).

**Result: `learned_embedding` now decisively beats `flat`** (0.805 vs.
0.700, +0.105, 10/10 seeds, stdev 0.012 — both the margin and its
consistency improved over the diluted first attempt). This is now a
legitimate, non-degenerate, capacity-matched opaque baseline — the fair
test `related-work.md` §2 asked for.

**`rf_aware` still beats it** (+0.113, 10/10 seeds, stdev 0.012) — smaller
than the margin over plain `flat` (+0.218) but real and consistent. This
*is* now citable as evidence for the narrowed hypothesis: explicit,
human-legible `ReferenceFrame` structure beats a reasonably strong opaque
multi-hypothesis baseline that itself clearly beats a flat single rule —
not just "explicit beats nothing."

## Third fix: a real gradient loss instead of a perceptron mistake-rule

The remaining candidate fix flagged in an earlier draft of this document:
the winning slot was still only updated on outright mistakes (`reward ==
0`), by a fixed step in the correcting direction — a perceptron rule, not
a proper loss gradient. But reward combined with the agent's own last
prediction implies the true label on *every* step, not just wrong ones (if
`reward == 1` the label was the prediction; if `reward == 0` it was the
opposite) — so a real supervised signal is available every step. Replaced
the perceptron update with an online logistic-regression gradient step
toward that implied label, applied every step to the winning slot (still
hard-assigned, as fixed above) — closer to how the multi-task RL baselines
this is modeled on (arXiv:2102.06177, arXiv:2207.02249) actually train
their context representations from reward, rather than a coarser
binary-correct/incorrect nudge.

**Result: `learned_embedding` improves further, to 0.837** (from 0.805 —
+0.137 over `flat`, 10/10 seeds), and **`rf_aware`'s margin over it tightens
to +0.080** (from +0.113, still 10/10 seeds, stdev 0.010). This is the
right direction for a meaningful test: strengthening condition B narrows
the gap rather than leaving it unchanged, and the hypothesis survives the
tighter comparison. `learned_embedding` (0.837) is now close to
`flat_oracle` (0.886) despite never being told which frame is active —
a genuinely capable opaque baseline at this point, not a strawman.

## `true_oracle`: a clean isolation of "cost of inference"

Added a second oracle-style control, `TrueOracleAgent`: given the true
active frame index *and* allowed to call the real `evaluate()` on the
correct known `ReferenceFrame` (unlike `flat_oracle`, which knows *which*
frame is active but still has to *learn* what its rule is from noisy
feedback). This isolates "cost of inference" as the entire gap between it
and `rf_aware`, rather than conflating it with "cost of relearning the
rule" the way the `flat_oracle` comparison does (see the original
`flat_oracle` caveat below, now resolved by this addition).

Result: `true_oracle` beats `rf_aware` by **+0.043, 10/10 seeds, very low
variance (stdev 0.004)** — small, consistent, and exactly what you'd
expect: `rf_aware` pays a real but modest cost for having to infer which
frame is active from feedback rather than being told directly.
`true_oracle`'s own accuracy (0.961/0.963 recovery/steady, both far above
everything else) is bounded only by observation noise near a true decision
boundary, confirming it behaves as the intended ceiling.

## `flat_oracle`'s residual gap, now explained rather than just flagged

The original `flat_oracle` vs. `rf_aware` gap (+0.032) previously carried
the caveat that it conflated "doesn't know which frame" with "doesn't know
the frame's rule." With `true_oracle` now available as the clean ceiling,
that decomposition is explicit: `true_oracle` (0.961) is the ceiling given
perfect information; `flat_oracle` (0.886) falls short of it by 0.075
purely from having to *learn* each frame's rule instead of being handed
it; `rf_aware` (0.918) falls short of the same ceiling by only 0.043,
despite not being told which frame is active at all. `rf_aware`'s
"knowing the exact rule via `evaluate()`" is worth more here than
`flat_oracle`'s "being told which frame is active" — a genuinely
interesting, non-obvious result, not just a marginal edge case.

## What this still does not establish

- **`learned_embedding` still isn't prior art's strongest opaque
  mechanism** — it's a hard-EM mixture of linear experts with an online
  logistic gradient update, not a full encoder-decoder trained on
  reward/dynamics prediction like arXiv:2102.06177 / arXiv:2207.02249.
  Each of the three fixes narrowed the gap to `rf_aware` further (from
  "loses to flat" → +0.113 → +0.080), which is the right trend, but
  extrapolating that trend to zero is speculation, not a result. Treat
  +0.080 as a real result against the strongest baseline actually built
  here, not as the final word against the strongest possible one.
- **No held-out-frame test** — all agents that use `FRAMES` know all four
  from the start; "performance on a frame not seen during training" from
  `research-agenda.md` #5's metrics list is still unmeasured.
- **No formal calibration metric** — `belief in true frame (steady)` =
  0.817 for `rf_aware` is a proxy (posterior mass on the actually-active
  frame), not a proper Brier/log score.
- ~~**Single environment configuration**~~ — swept below (noise and switch
  frequency), mirroring Experiment 2's noise sweep.

## What this changes going forward

- The core "structure beats flat/no-structure, under matched information"
  claim survives every test run so far, including against a fair condition
  B and two different oracle controls (`flat_oracle`, `true_oracle`) —
  genuinely encouraging, not just "not yet falsified."
- The clean ordering `flat` < `learned_embedding` < `flat_oracle` <
  `rf_aware` < `true_oracle` is itself a useful result: more capacity
  (single rule → multi-hypothesis opaque → multi-hypothesis + told which
  → multi-hypothesis + exact rule) monotonically helps, and knowing the
  *rule* (`rf_aware`'s `evaluate()`) matters more than knowing *which*
  frame is active (`flat_oracle`'s revealed id) — worth stating precisely
  in any future write-up rather than collapsing to "structure wins."
- `true_oracle` is now the right ceiling to cite for "cost of inference";
  `flat_oracle` is now best understood as isolating "cost of not knowing
  the rule" specifically, given the decomposition above.
- Done: `learned_embedding` was strengthened with a real gradient loss
  (see the third fix above), and the `rf_aware` margin over it did shrink
  (+0.113 → +0.080) while staying robust (10/10 seeds). The remaining
  lever — an actual small encoder-decoder over reward/dynamics, matching
  arXiv:2102.06177 more literally — is a larger build, not required to
  trust the current result, but would strengthen it further if pursued.
- The environment configuration is now swept (see below) rather than
  fixed at one noise level and one switch period.

## Sweep: the advantage is not noise-invariant, and shrinks toward zero

Ran: `PYTHONPATH=. python experiments/exp01_frame_conditioning/sweep.py`.
Two 1-D sweeps (10 seeds, 3000 steps each grid point), reusing the exact
same `run_agent_on_seed`/`recovery_curve`/`calibration` helpers as `run.py`
so the sweep and the headline single-configuration numbers above can't
silently drift apart in definition.

**Noise sweep** (switch period fixed at 40±10; same noise levels as
Experiment 2, for comparability):

| sigma | flat | learned_embedding | rf_aware | flat_oracle | true_oracle | rf_aware − flat | rf_aware − learned_embedding |
|---|---|---|---|---|---|---|---|
| 0.00 | 0.711 | 0.850 | 0.961 | 0.928 | 1.000 | +0.250 | +0.111 |
| 0.02 | 0.707 | 0.850 | 0.943 | 0.907 | 0.983 | +0.235 | +0.093 |
| 0.05 | 0.700 | 0.837 | 0.918 | 0.886 | 0.961 | +0.218 | +0.080 |
| 0.10 | 0.685 | 0.816 | 0.873 | 0.847 | 0.920 | +0.188 | +0.057 |
| 0.20 | 0.657 | 0.753 | 0.787 | 0.771 | 0.845 | +0.130 | +0.034 |
| 0.40 | 0.613 | 0.658 | 0.659 | 0.654 | 0.732 | +0.046 | **+0.001** |

**This is an important boundary condition, not just a robustness check:
`rf_aware`'s advantage over both `flat` and `learned_embedding` shrinks
monotonically as observation noise grows, and effectively vanishes by
`sigma=0.4`** (+0.001 over `learned_embedding` — indistinguishable from
noise in the estimate itself, given stdev on the order of 0.01-0.05 at
this noise level). At high enough noise, the raw observation carries so
little signal about the entity's true value that no amount of structural
knowledge about the candidate frames helps — everything converges toward
similarly mediocre performance (`flat` 0.613, `learned_embedding` 0.658,
`rf_aware` 0.659, `flat_oracle` 0.654, all within 0.05 of each other, vs.
a >0.35 spread at `sigma=0`). Report the hypothesis as holding *within the
noise regime tested* (σ ≤ ~0.2), not universally — this is exactly the
kind of scope qualifier `research-agenda.md`'s falsification discipline
calls for, and it wasn't visible from the single `sigma=0.05` configuration
alone.

**Switch-period sweep** (noise fixed at 0.05; jitter scaled to period/4):

| period | flat | learned_embedding | rf_aware | flat_oracle | true_oracle | rf_aware − flat | rf_aware − learned_embedding |
|---|---|---|---|---|---|---|---|
| 20 | 0.636 | 0.804 | 0.882 | 0.887 | 0.960 | +0.245 | +0.078 |
| 40 | 0.700 | 0.837 | 0.918 | 0.886 | 0.961 | +0.218 | +0.080 |
| 80 | 0.762 | 0.870 | 0.935 | 0.886 | 0.960 | +0.173 | +0.065 |

Two things worth stating precisely rather than averaging away:

- **`rf_aware`'s absolute margin over `flat` is largest under frequent
  switching** (+0.245 at period=20 vs. +0.173 at period=80) — faster
  regime changes punish a single slowly-readapting rule more than they
  punish an agent already tracking multiple hypotheses. Everyone's
  accuracy rises with slower switching (more time to settle before the
  next change), but `flat`'s rises fastest, closing part of the gap.
- **`flat_oracle` is essentially insensitive to switch frequency**
  (0.887 / 0.886 / 0.886) — because it maintains one independent sub-rule
  per frame id and each sub-rule accumulates training data continuously
  over the full 3000 steps regardless of how often the environment
  switches between them. This is a clean, sensible mechanical explanation,
  not a coincidence.
