# Research Agenda

## 1. Purpose

This document formalizes the first falsifiable claims TransIntelligence is
willing to stand behind, and specifies the smallest experiments that could
support or kill each one. It intentionally covers less ground than the
project vision (see `docs/architecture.md`, `docs/intelligence-model.md`):
vision motivates the program, this document constrains near-term work to
what can actually be measured.

Distinguish four categories throughout:

- **Established theory** — already proven elsewhere; we reuse it.
- **Synthesis** — a recombination of established ideas, not itself new.
- **Hypothesis** — a claim we believe but have not tested.
- **Novel proposal** — a mechanism we believe is new; treat with the most
  skepticism.

## 2. Central hypothesis, formalized

> Representing the evaluation of an entity as `evaluate(x, reference_frame)`
> rather than `evaluate(x)`, and making `reference_frame` an explicit,
> inferable, human-legible object, improves an agent's decision quality,
> adaptation speed, and calibration under environments where the
> frame-conditional optimum changes over time — **relative to a baseline
> agent with access to the same underlying information but no compositional
> frame structure.**

The bolded clause is the whole ballgame. Without it, this collapses into
"giving an agent the context variable helps," which is not a claim, it's
contextual-bandit textbook material (established theory: contextual bandits,
POMDPs, meta-RL task embeddings, causal do-calculus conditioning sets). The
part that could be a genuine contribution is narrower:

> Does an explicit, compositional, inspectable frame representation
> outperform an equally-informed *implicit* representation (e.g. frame-id
> concatenated into a flat feature vector and left to a monolithic
> function approximator), particularly (a) under distribution shift when
> the frame changes, (b) in low-data regimes, and (c) on calibration —
> can the agent say *why* a conclusion changed, not just that it did.

That's testable. "Intelligence is fundamentally about navigating reference
frames" is not testable and should stay in the vision doc.

## 3. Relationship to existing work (compressed — full citations and per-area detail in [related-work.md](related-work.md))

| TransIntelligence concept | Closest established analogue | Status |
|---|---|---|
| Reference frame | Context variable in contextual bandits (Li et al. 2010) / POMDP belief states (Kaelbling et al. 1998); goal-conditioned value functions (Schaul et al. 2015); task embeddings in meta-RL (Duan et al. 2016 RL²; Finn et al. 2017 MAML) and multi-task RL (arXiv:2102.06177, arXiv:2207.02249) | established theory, we're proposing a specific (explicit, human-legible) representation of it |
| `evaluate(x, R)` vs `evaluate(x)` | Conditional expectation / conditioning set in causal inference (Pearl 2009) | established theory |
| Frame-dependent vs frame-invariant detection | Invariant risk minimization (Arjovsky et al. 2019, with reliability caveats in arXiv:2010.05761); domain generalization surveys (Zhou et al. 2023; Wang et al. 2022) | established theory, applied to a new object |
| Cross-domain transfer of reasoning strategy | Structure-mapping theory (Gentner 1983); compositional generalization (Lake & Baroni 2018/2023) | established distinctions for what counts as real transfer vs. relabeling; our specific transfer claim is a hypothesis and the least supported of the three |
| Self-model / strange loops | Active inference / free-energy principle (Friston 2010); Machine Theory of Mind (Rabinowitz et al. 2018, models *other* agents, not self) | motivation only, not yet a hypothesis — no operationalization exists, do not build Phase 8-9 code yet |

Do not claim novelty on "reference frames matter." **[related-work.md §2](related-work.md#2-reference-frames-as-conditioning-context)
narrows the claim further than originally scoped**: arXiv:2102.06177 and
arXiv:2207.02249 already compare flat task-ID conditioning against richer
*learned* task-embedding representations in multi-task RL. If that
literature already shows structured representations beat flat ones under
shift, the novel part of experiment 1 is not "structured beats flat" — it's
narrower still: whether an *explicit, human-legible* `ReferenceFrame`
object beats an *opaque learned embedding* with equivalent expressive
power, and whether any gap is in reward or in interpretability/
calibration. State it at that resolution, not the broader one. A broad
claim will not survive contact with a reviewer who works in meta-RL.

## 4. Current state vs. what each experiment needs

| Needed | Exists today | Gap |
|---|---|---|
| `ReferenceFrame`, `Entity`, `Observation`, `Context` | Yes ([transintelligence/core](../transintelligence/core), [representation/reference_frames](../transintelligence/representation/reference_frames)) | none |
| `evaluate/compare/rank/sensitivity` | Yes, but purely deterministic arithmetic on hand-supplied frames ([reasoning/relative/model.py](../transintelligence/reasoning/relative/model.py)) | no inference, no learning |
| An environment with a hidden/changing active frame | No | build |
| An agent that selects actions and receives reward | No — `KernelAgent` only records steps ([agents/base.py](../transintelligence/agents/base.py)) | build |
| A baseline agent with equivalent info, no frame structure | No | build |
| Metrics/logging harness, `experiments/` dir | No | build |

Experiments 1 and 2 are achievable with modest new code on top of the
existing core. Experiment 3 additionally requires two structurally-related
synthetic domains and is the least mature — sequence it last, and only if
1 and 2 hold up.

## 5. Experiment 1 — Does explicit frame structure beat implicit context-conditioning?

**This is the load-bearing experiment. If this fails, the "signature
component" in §8 of the master context does not have a technical
contribution yet, whatever the vision document says.**

**Status: run, all five conditions.** Environment, condition A, condition
A-oracle, condition B, and a second cleaner oracle all built and run —
[environments/transworld/](../environments/transworld/) and
[experiments/exp01_frame_conditioning/](../experiments/exp01_frame_conditioning/RESULTS.md).
Summary:

- **Condition A (flat, matched zero-information): decisive.** RF-aware
  beats flat by +0.218 accuracy, 10/10 seeds, low variance.
- **`flat_oracle` control: survives, small margin (+0.032, 10/10 seeds).**
  Now cleanly decomposed by a second oracle (`true_oracle`, given the true
  frame id *and* allowed to call the real `evaluate()`, isolating "cost of
  inference" alone): `true_oracle` beats `rf_aware` by +0.043 (10/10 seeds,
  very low variance) — a real, modest cost for having to infer rather than
  be told. `flat_oracle` falls short of that same ceiling by 0.075, purely
  from having to *learn* each frame's rule instead of being handed it.
  `rf_aware`'s exact-rule-via-`evaluate()` is worth more than
  `flat_oracle`'s "told which frame" — a genuinely interesting result.
- **Condition B (learned embedding): built, fixed twice, now a fair test.**
  First implementation had a real bug (zero-initialized slots are
  provably degenerate — collapse into a scaled copy of a single flat rule).
  Fixed via random initialization, but that alone still underperformed
  flat (0.632 vs. 0.700, losing 9/10 seeds) — soft belief-weighted updates
  diluted learning across all 4 competing slots. Fixed again via hard
  (argmax) responsibility assignment for weight updates (belief itself
  stays soft, only the update target is hardened): `learned_embedding` now
  decisively beats `flat` (0.805, +0.105, 10/10 seeds) — a legitimate,
  non-degenerate, capacity-matched opaque baseline. `rf_aware` still beats
  it, **+0.113, 10/10 seeds** — this is now citable as evidence for the
  narrowed claim from `related-work.md` §2: explicit structure beats a
  baseline that itself clearly beats flat, not just "explicit beats
  nothing." Full ordering: `flat` (0.700) < `learned_embedding` (0.805) <
  `flat_oracle` (0.886) < `rf_aware` (0.918) < `true_oracle` (0.961) —
  clean and monotonic.

Read the linked results in full before citing any of this externally —
several of these numbers only make sense with the decomposition explained
there, and `learned_embedding` is still not prior art's *strongest*
possible opaque mechanism (a hard-EM mixture of linear experts, not an
encoder-decoder), so the +0.113 margin is a real result against the
baseline actually built, not the final word against the strongest one.

- **Hypothesis:** In a synthetic environment where the reward-optimal
  action depends on a hidden "active reference frame" that changes at
  unannounced change-points, an agent using explicit `evaluate(x, R)` over
  a small candidate set of frames, with frame-belief updated from reward
  feedback, achieves higher cumulative reward, faster post-shift recovery,
  and better-calibrated frame-belief than a baseline of matched capacity
  that receives the same raw features (including a frame-id feature) but
  has no compositional evaluate/frame structure.
- **Environment:** A minimal synthetic world — a handful of entities with
  2-3 properties, 3-4 known candidate reference frames (differing
  `baseline`/`direction`/weighting, as already modeled in `ReferenceFrame`),
  reward = correctness of the action under the *currently active* frame,
  frame switches every N steps without warning. Small enough to fit in
  `environments/transworld/` as a few hundred lines.
- **Baseline agent (condition A — flat):** contextual bandit / small
  function approximator that takes raw features + frame-id as input, no
  `ReferenceFrame` object, no `evaluate()` call — same information, flat
  representation.
- **Baseline agent (condition B — learned embedding):** per
  [related-work.md §2](related-work.md#2-reference-frames-as-conditioning-context),
  condition A alone is not a strong enough baseline — arXiv:2102.06177 and
  arXiv:2207.02249 show learned task embeddings already beat flat
  conditioning in multi-task RL. Add a second baseline that learns an
  opaque embedding of the active frame from reward/dynamics differences
  (same spirit as those papers), with capacity matched to the RF-aware
  agent. This is the real bar to clear.
- **RF-aware agent:** maintains a belief distribution over the known frame
  set, calls `evaluate(x, R)` per candidate frame, updates belief via
  reward feedback (this requires adding a frame-inference/belief-update
  step that does not exist yet — the current `BaselineRelativeReasoner`
  assumes the frame is given).
- **Metrics:** cumulative regret, steps-to-recover after a frame switch,
  Brier score / calibration of frame-belief vs. ground-truth active frame,
  performance on a held-out frame not seen during training, plus (for the
  RF-aware agent vs. condition B specifically) whether the RF-aware agent
  can *explain* a conclusion change in terms of `ReferenceFrame` fields —
  something an opaque embedding structurally cannot do, regardless of the
  reward comparison.
- **Falsification:** if the RF-aware agent's advantage over condition A
  disappears once condition A gets the frame-id feature directly, the
  broad hypothesis is dead — the benefit was information access, not
  structure. If the RF-aware agent's reward advantage over condition B
  (learned embedding) also disappears, the remaining defensible claim
  shrinks to interpretability/calibration only, not decision quality — say
  so plainly rather than quietly dropping condition B from the writeup.

## 6. Experiment 2 — Frame-dependence detection and calibration — RUN, FALSIFIED, FIXED, RE-VERIFIED

**Status: done, including the fix.** Code, data generator, and full
write-up (both the original falsifying result and the post-fix numbers) in
[experiments/exp02_frame_dependence/](../experiments/exp02_frame_dependence/RESULTS.md);
regression-locked in `tests/test_exp02_frame_dependence.py`. Summary below —
read the linked results for the algebra and the full noise-sweep tables.

- **Hypothesis (as originally stated):** `sensitivity(x, R1, R2)` (already
  implemented, `reasoning/relative/model.py:44`) can distinguish, with
  reasonable precision/recall, properties that are frame-dependent by
  construction from properties that are frame-invariant by construction,
  across varying observation noise.
- **Design (as run):** synthetic entities with a known noiseless true value
  `mu`; ground-truth label = whether `sign(evaluate(x,R))` flips between two
  candidate frames at the true `mu`. Compared `sensitivity()` against that
  label via ROC-AUC, across a noise sweep, in two regimes — frame pairs that
  share a `direction` and frame pairs that don't.
- **Result: the hypothesis is false whenever the compared frames share a
  `direction` (the common case — e.g. all three volatility frames in
  `examples/finance_demo.py` share `direction`).** `sensitivity()` reduces
  algebraically to `abs(baseline2 - baseline1)` in that regime — the
  entity's raw value cancels out of the subtraction entirely, so the
  statistic is a constant function of the frame pair, not a per-entity
  signal. AUC sits at chance (0.49-0.59) regardless of noise, because
  there was never a per-entity signal to degrade. This is proven
  algebraically, not just observed at one seed — see the regression test.
  It's genuinely informative (AUC 1.000 → 0.530 across the noise sweep,
  the degradation curve originally asked for) only when the two frames
  disagree on `direction`.
- **Interpretation:** this is a real finding about
  `reasoning/relative/model.py`, not a dead end. `_score()` is a bare
  location-shift (`raw - baseline`) with no mechanism for entity-level data
  to survive a same-direction comparison. A fix needs a statistic relative
  to the entity's position *between* the two (direction-adjusted)
  baselines — e.g. a boundary-crossing probability under the observation's
  confidence — not a raw point-difference. Do this before leaning on
  `evaluate`/`compare`/`sensitivity` for anything in Experiment 1, which
  reuses the same reasoner and inherits the same blind spot.
- **Cost:** low, as predicted — pure stdlib, no new dependencies, ran in
  under a second. Confirms this was the right one to ship first: it caught
  a real defect before any engineering time went into Experiment 1's
  environment/agent build-out.
- **Fix, and re-verification:** `sensitivity()` now compares the signs of
  the two `evaluate()` results instead of their difference (negative means
  the conclusion flips / frame-dependent, positive means it agrees /
  frame-invariant, magnitude is the margin to the decision boundary).
  Re-running the same `run.py` harness with no changes other than score
  polarity shows `same_dir` AUC now matches `diff_dir` exactly: 1.000 at
  zero noise, degrading smoothly to chance by `sigma=0.4`, in both regimes.
  The previously-failing regime is fixed, not worked around.

## 7. Experiment 3 — Minimal cross-domain structural transfer

- **Hypothesis:** a frame-inference strategy trained in one synthetic
  domain (e.g. finance-shaped: assets/volatility/regime) transfers to a
  structurally isomorphic but superficially different domain (e.g.
  knowledge-shaped: ideas/novelty/citation-regime) with better sample
  efficiency than a domain-specific baseline trained from scratch.
- **Status: sequence last, treat as the least mature.** It depends on
  experiment 1's infrastructure existing and working, and on being able to
  construct two domains that are genuinely structurally isomorphic (same
  frame-switch dynamics, same reward structure) and not just superficially
  relabeled — if they're relabeled, "transfer" is trivial and proves
  nothing. Use Gentner's structure-mapping criterion (1983, see
  [related-work.md §6](related-work.md#6-analogical--structural-cross-domain-transfer-experiment-3-and-the-trans-claim-generally))
  as the actual test: the two domains must share a system of
  interconnected *relations* (same frame-switch dynamics, same reward
  structure) while deliberately differing in surface *attributes*
  (asset/volatility vs. idea/novelty). If they instead share attributes
  and differ only in label strings, that's similarity or relabeling, not
  analogy, and doesn't satisfy the experiment.
- **Falsification:** no sample-efficiency advantage over from-scratch
  training, or the advantage disappears once the domains are made
  non-isomorphic (a critical control condition — run this control, not
  just the positive case).

## 8. Sequencing

1. ~~Experiment 2 first~~ — **done**, see §6. Result: `sensitivity()` failed
   in the same-direction regime, provably.
2. ~~Fix `sensitivity()`~~ — **done**. Rewritten to compare the signs of
   the two `evaluate()` results (each individually raw-value-dependent)
   instead of their point-difference; negative result now means the
   conclusion flips (frame-dependent), positive means it agrees
   (frame-invariant), magnitude is the margin. Re-running
   `experiments/exp02_frame_dependence/run.py` confirms the gap is closed:
   `same_dir` AUC now matches `diff_dir` — 1.000 at zero noise, degrading
   smoothly to chance by `sigma=0.4` in both regimes. Full numbers in
   [experiments/exp02_frame_dependence/RESULTS.md](../experiments/exp02_frame_dependence/RESULTS.md).
   Regression-locked in `tests/test_exp02_frame_dependence.py`.
3. Experiment 1 next — requires building `environments/transworld/` and a
   real agent with belief-update, the actual infrastructure investment.
   `evaluate`/`compare`/`sensitivity` are now safe to build on for both
   same- and opposite-direction frame pairs.
4. Experiment 3 only if 1 and 2 hold up, plus the isomorphism control.

## 9. What would make this publishable, and what would make a reviewer skeptical

- **Publishable:** a positive experiment 1 result that survives the
  "give the baseline the frame-id feature" control, plus a calibration
  result from experiment 2, framed narrowly as "compositional
  frame-conditioning as an inductive bias" rather than as a claim about
  general intelligence.
- **Skeptical reviewer's first question:** "isn't this just a context
  variable / task embedding with extra steps?" Experiment 1's control
  condition exists specifically to answer that question before a reviewer
  asks it.
- **Do not** lead with cross-domain transfer or strange loops in any
  external write-up until experiments 1-2 produce evidence; the master
  context's own §21 and §33 already say this.
