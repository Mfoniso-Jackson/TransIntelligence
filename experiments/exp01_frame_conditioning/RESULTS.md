# Experiment 1 results — explicit frame structure vs. a flat baseline

Ran: `PYTHONPATH=. python experiments/exp01_frame_conditioning/run.py`.
4 candidate frames (mixed baselines and directions — see `FRAMES` in
`run.py`), 3000 steps, switch every 40±10 steps, observation noise
σ=0.05, 10 seeds. See `docs/research-agenda.md` #5 for the hypothesis and
`docs/related-work.md` #2 for why this alone doesn't yet earn the full
claim (condition B, below, is not built).

## Result

| agent | overall acc. | stdev (across seeds) | recovery acc. (steps 1-10 post-switch) | steady acc. (steps 30-40 post-switch) | belief in true frame (steady) |
|---|---|---|---|---|---|
| flat | 0.700 | 0.020 | 0.586 | 0.709 | n/a |
| rf_aware | **0.918** | 0.005 | 0.780 | **0.906** | 0.817 |
| flat_oracle | 0.886 | 0.006 | **0.882** | 0.899 | n/a |

Paired, per-seed (same env/seed for both agents in each pair):

```
overall_acc(rf_aware) - overall_acc(flat):        mean=+0.218  stdev=0.016  wins=10/10 seeds
overall_acc(rf_aware) - overall_acc(flat_oracle):  mean=+0.032  stdev=0.007  wins=10/10 seeds
```

## Finding: the hypothesis survives both tests run so far, with a real nuance on *why*

**RF-aware beats the flat baseline decisively and consistently** (+0.218,
10/10 seeds, low variance) when neither agent is told which frame is
active — both have to infer it purely from reward feedback. This is the
core comparison the hypothesis needs, and it's not close: the flat
baseline's single online-adapting linear rule struggles both to recover
after a switch (0.586 vs. 0.780) and to reach a good steady state (0.709
vs. 0.906).

**RF-aware also beats `flat_oracle`** — the deliberately unfair control
where the flat rule is *told* the true active frame index directly, per
the falsification criterion in `research-agenda.md` #5 ("if the RF-aware
agent's advantage disappears once the baseline is given the frame-id
feature ... the hypothesis is not supported"). The advantage doesn't
disappear (+0.032, 10/10 seeds) — but it's much smaller than the margin
over the uninformed flat baseline, and it's not uniform across the two
things being measured:

- **`flat_oracle` recovers faster after a switch** (0.882 vs. 0.780) —
  unsurprising, it doesn't have to infer anything, it just switches to
  the right one of its four independently-learned per-frame rules.
- **`rf_aware` reaches a better steady state** (0.906 vs. 0.899) — because
  `rf_aware` uses the *actual* `ReferenceFrame` baseline/direction via a
  real `evaluate()` call once its belief concentrates on the right frame,
  whereas `flat_oracle` still has to *learn* each frame's rule from noisy
  reward feedback via a perceptron-style update, which doesn't converge
  to the exact decision boundary as precisely.

**This is an important caveat, not a footnote:** `flat_oracle` as built
here (matching `research-agenda.md`'s original spec — "no `evaluate()`
call") is given the *categorical* frame id but not the frame's actual
numeric definition. So its small residual gap against `rf_aware` partly
reflects "doesn't know the exact rule," not purely "lacks compositional
structure." A cleaner isolation of *structure* alone would need an oracle
that also gets to call the real `evaluate()` once given the correct frame
id (making it converge to ~100% instantly, bounded only by observation
noise) — that would isolate "cost of inference" as the entire gap, rather
than conflating it with "cost of relearning the rule." Not built yet; flag
before citing this result as clean evidence for compositional structure
specifically, rather than for reference-frame conditioning in general.

## What this does *not* yet establish

- **Condition B (learned embedding) is not built.** Per
  `docs/related-work.md` #2, the real bar — given prior art already shows
  learned task embeddings beat flat task-ID conditioning in multi-task RL —
  is whether `rf_aware`'s *explicit, human-legible* representation beats an
  *opaque learned embedding* baseline with matched capacity, not whether it
  beats a flat linear rule. This run answers a necessary but not
  sufficient question. Do not present this as the full falsification test
  from `research-agenda.md` #5 — only the condition-A and oracle-control
  legs of it.
- **No held-out-frame test.** The original metrics list in
  `research-agenda.md` #5 includes "performance on a held-out frame not
  seen during training" — not measured here; all four frames are known to
  `rf_aware` from the start.
- **No formal calibration metric.** "belief in true frame (steady)" =
  0.817 is a reasonable proxy (average posterior mass on the actually-active
  frame during steady windows) but isn't a proper Brier/log-score against
  a full predictive distribution — worth tightening if calibration becomes
  load-bearing for a claim.
- **Single environment configuration.** One noise level (σ=0.05), one
  switch period (40±10), 10 seeds. Robust within that configuration
  (10/10 seed wins, low variance) but not swept across noise levels or
  switch frequencies the way Experiment 2 was.

## What this changes going forward

- The core "structure beats flat, under matched information" claim is not
  falsified by this run — proceed to building condition B (learned
  embedding) next, since that's the comparison that actually tests the
  narrowed claim from `docs/related-work.md` #2.
- Before citing the `rf_aware` vs. `flat_oracle` margin as evidence for
  *compositional structure* specifically, build the stronger oracle
  variant described above (given frame id, allowed to call `evaluate()`)
  to separate "doesn't know the rule" from "lacks structure."
