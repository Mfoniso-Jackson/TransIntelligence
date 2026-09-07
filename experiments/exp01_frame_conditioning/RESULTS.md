# Experiment 1 results — explicit frame structure vs. flat, learned-embedding, and oracle baselines

Ran: `PYTHONPATH=. python experiments/exp01_frame_conditioning/run.py`.
4 candidate frames (mixed baselines and directions — see `FRAMES` in
`run.py`), 3000 steps, switch every 40±10 steps, observation noise
σ=0.05, 10 seeds. See `docs/research-agenda.md` #5 for the hypothesis and
`docs/related-work.md` #2 for why condition A alone doesn't earn the full
claim.

## Result — all five conditions

| agent | overall acc. | stdev (across seeds) | recovery acc. (steps 1-10 post-switch) | steady acc. (steps 30-40 post-switch) | belief in true frame (steady) |
|---|---|---|---|---|---|
| flat | 0.700 | 0.020 | 0.586 | 0.709 | n/a |
| learned_embedding | 0.632 | 0.050 | 0.576 | 0.625 | n/a |
| rf_aware | **0.918** | 0.005 | 0.780 | 0.906 | 0.817 |
| flat_oracle | 0.886 | 0.006 | 0.882 | 0.899 | n/a |
| true_oracle | **0.961** | 0.002 | **0.962** | **0.963** | n/a |

Paired, per-seed (same env/seed for both agents in each pair):

```
overall_acc(rf_aware) - overall_acc(flat):               mean=+0.218  stdev=0.016  wins=10/10 seeds
overall_acc(rf_aware) - overall_acc(learned_embedding):   mean=+0.286  stdev=0.051  wins=10/10 seeds
overall_acc(learned_embedding) - overall_acc(flat):       mean=-0.068  stdev=0.052  wins=1/10 seeds
overall_acc(rf_aware) - overall_acc(flat_oracle):         mean=+0.032  stdev=0.007  wins=10/10 seeds
overall_acc(true_oracle) - overall_acc(rf_aware):         mean=+0.043  stdev=0.004  wins=10/10 seeds
```

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

## Finding, post-fix: condition B does not yet establish a fair test — it underperforms flat

After the fix, `learned_embedding` (0.632) is **worse** than `flat`
(0.700), losing in 9 of 10 seeds, with over twice the variance (0.050 vs.
0.020). This is a real, honestly-reported negative result, not something
to quietly drop:

- It does **not** contradict the multi-task RL literature cited in
  `related-work.md` §2 (arXiv:2102.06177, arXiv:2207.02249) — those use
  encoder-decoder architectures trained on reward/dynamics prediction, a
  materially stronger mechanism than a soft-EM mixture of 4 linear experts
  updated by a crude perceptron rule. This implementation is a first,
  deliberately simple attempt, not a replication of that literature.
- The likely cause: with only ~40 steps between switches and 4
  near-randomly-initialized slots competing for a small, noisy reward
  signal, there isn't enough signal for slots to cleanly specialize before
  the environment moves on — the soft (belief-weighted) update spreads
  learning across all 4 slots at once rather than letting one slot commit,
  which is slower to converge than a single rule that commits fully to
  whatever it's currently seeing.
- **This means the `rf_aware` vs. `learned_embedding` comparison
  (+0.286, 10/10 seeds) is not yet a clean test of "explicit beats
  opaque-but-equally-capable."** It's currently closer to "explicit beats
  a weak/undertuned opaque baseline," which is a much less interesting
  claim. Do not cite this margin as evidence for the narrowed hypothesis
  from `related-work.md` §2 until `learned_embedding` at least reliably
  beats `flat` — right now it doesn't, so it isn't a strong enough
  baseline to lose to meaningfully.
- Candidate fixes, not yet tried: hard (argmax) responsibility assignment
  instead of soft/belief-weighted updates so one slot fully commits per
  step; a proper gradient loss instead of a perceptron mistake-rule; more
  steps between switches to give slots time to specialize; or fewer slots
  than known frames to reduce the competition-for-signal problem.

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

- **A fair test of condition B** — see above; `learned_embedding` needs to
  reliably beat `flat` before the `rf_aware` vs. `learned_embedding`
  margin means anything.
- **No held-out-frame test** — all agents that use `FRAMES` know all four
  from the start; "performance on a frame not seen during training" from
  `research-agenda.md` #5's metrics list is still unmeasured.
- **No formal calibration metric** — `belief in true frame (steady)` =
  0.817 for `rf_aware` is a proxy (posterior mass on the actually-active
  frame), not a proper Brier/log score.
- **Single environment configuration** — one noise level, one switch
  period, 10 seeds; not swept the way Experiment 2 was.

## What this changes going forward

- The core "structure beats flat/no-structure, under matched information"
  claim survives every test run so far, including against two different
  oracle controls (`flat_oracle`, `true_oracle`) — genuinely encouraging,
  not just "not yet falsified."
- Before improving `learned_embedding`, decide whether it's worth the
  engineering time: the point of condition B was to test against prior
  art's *strongest* opaque baseline, and a stronger implementation (hard
  responsibility assignment, real gradient loss) is a real but nontrivial
  build, not a quick fix.
- `true_oracle` is now the right ceiling to cite for "cost of inference";
  `flat_oracle` is now best understood as isolating "cost of not knowing
  the rule" specifically, given the decomposition above.
