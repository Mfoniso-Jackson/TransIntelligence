# Research Agenda

## 1. Purpose

This document formalizes the first falsifiable claims TransIntelligence is
willing to stand behind, and specifies the smallest experiments that could
support or kill each one. It intentionally covers less ground than the
project vision (see `docs/architecture.md`, `docs/intelligence-model.md`):
vision motivates the program, this document constrains near-term work to
what can actually be measured.

All twenty experiments below have now run. For a standalone summary of
what they actually established — without reading this document's
incremental updates or twenty separate `RESULTS.md` files — see
[docs/findings.md](findings.md).

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
- **Condition B (learned embedding): built, fixed three times, now a fair
  test.** First implementation had a real bug (zero-initialized slots are
  provably degenerate — collapse into a scaled copy of a single flat rule).
  Fixed via random initialization, but that alone still underperformed
  flat (0.632 vs. 0.700, losing 9/10 seeds) — soft belief-weighted updates
  diluted learning across 4 competing slots. Fixed again via hard (argmax)
  responsibility assignment for weight updates: `learned_embedding` then
  decisively beat `flat` (0.805, +0.105) but still trailed `rf_aware` by
  +0.113. A third fix — a real online logistic-regression gradient step
  (using every step's reward-implied label, not just mistakes) in place of
  the perceptron rule — pushed `learned_embedding` to 0.837 and narrowed
  the `rf_aware` margin further to **+0.080, 10/10 seeds**. Full ordering:
  `flat` (0.700) < `learned_embedding` (0.837) < `flat_oracle` (0.886) <
  `rf_aware` (0.918) < `true_oracle` (0.961) — clean and monotonic, and the
  gap narrows in the right direction each time the baseline is
  strengthened, which is the trend that actually earns trust in the
  result (rather than a single lucky number).
  A fourth attempt (a shared reward-EMA temporal feature per slot) made no
  measurable difference and was reverted rather than kept as unjustified
  complexity — a genuine negative result, and evidence the remaining gap
  isn't caused by "no temporal context" (the belief vector already
  supplies that), narrowing where a future, more ambitious rebuild would
  need to look.
- **The more literal encoder-decoder rebuild was attempted and failed for
  a well-understood, unrelated reason.** Built `RNNEmbeddingAgent`: a
  genuine recurrent hidden state, no discrete slots, trained via truncated
  BPTT (Williams & Peng, 1990) — the architecture actually closest to
  arXiv:2102.06177's "infer context from trajectory." It scored **0.630,
  worse than `flat` (0.700)**, and increasing the BPTT truncation length
  from 1 to 20 steps changed nothing — a textbook signature of the
  vanishing-gradient problem in vanilla RNNs (Bengio, Simard, Frasconi,
  1994), the exact problem LSTM/GRU gating (Hochreiter & Schmidhuber,
  1997) was invented to fix. **This strengthens rather than weakens the
  overall case**: a plausible, more literal opaque baseline failed for a
  reason unrelated to `rf_aware`'s specific advantage, while
  `learned_embedding` (the simpler discrete-mixture baseline) remains the
  strongest opaque baseline that actually works, and clears +0.080 below
  `rf_aware`. An LSTM/GRU-gated version is the concrete next lever, not
  attempted.
- **Swept across noise and switch frequency** (mirroring Experiment 2's
  noise sweep) — **and found a real boundary condition**: `rf_aware`'s
  advantage over both `flat` and `learned_embedding` shrinks monotonically
  as observation noise grows and **effectively vanishes by σ=0.4**
  (+0.001 over `learned_embedding`, indistinguishable from estimation
  noise). Report the hypothesis as holding within the tested noise regime
  (σ ≲ 0.2), not universally. Separately, `rf_aware`'s margin over `flat`
  is largest under *frequent* switching (+0.245 at period=20 vs. +0.173 at
  period=80) — faster regime changes punish a single slowly-readapting
  rule more than an agent already tracking multiple hypotheses.
- **Formal calibration added**: `brier_score()` (proper scoring rule over
  the full belief distribution, not just mass on the true frame) gives
  0.191 for `rf_aware` — well below the 0.75 a uniform guess over 4 frames
  would score.
- **Held-out-frame test run — and it surfaced a real, substantial
  limitation, not a clean pass.** `rf_aware`, constructed knowing only 3
  of the environment's 4 frames, degrades sharply when the 4th (genuinely
  distinct, not a near-duplicate) frame is active: **−0.181 accuracy**,
  vs. essentially no gap for `flat` (−0.015) or `learned_embedding`
  (+0.003), neither of which was ever conditioned on a fixed candidate
  set. **The same design property that makes `rf_aware` strong when the
  world matches its known frame set makes it brittle exactly when it
  doesn't** — a real caveat for any external framing of this experiment,
  not a footnote. Points toward frame *discovery*, not just frame
  *selection* among a fixed list, as necessary if this approach is
  extended toward less controlled environments.

Read the linked results in full before citing any of this externally —
several of these numbers only make sense with the decomposition explained
there, and `learned_embedding` is still not prior art's *strongest*
possible opaque mechanism (a hard-EM mixture of linear experts with a
logistic update, not a full encoder-decoder), so +0.080 is a real result
against the strongest baseline actually built, not the final word against
the strongest possible one. **The overall picture is positive-with-real-
caveats, not simply positive** — lead with the noise and held-out-frame
limits in any future pitch, don't bury them.

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
- **Design, as run:** used `LearnedEmbeddingAgent` (docs/research-agenda.md
  #5), not `RFAwareAgent` — `RFAwareAgent`'s "knowledge" is just its
  `ReferenceFrame` list, trivial to hand over verbatim, nothing learned to
  test transfer of. Trained `LearnedEmbeddingAgent`'s `(w, b)` slots on a
  finance-shaped domain, warm-started a fresh agent on a knowledge-shaped
  domain with those values, and compared against training from scratch —
  see [experiments/exp03_cross_domain_transfer/](../experiments/exp03_cross_domain_transfer/RESULTS.md)
  for the domain definitions (built to satisfy Gentner's structure-mapping
  criterion, 1983: same relational structure, i.e. same `(baseline,
  direction)` numeric rules, different surface attributes) and the
  non-isomorphic control (same labels, deliberately different numeric
  structure).
- **Status: run, with a critical third control added beyond the original
  design.** The naive transfer-vs-scratch comparison looked clean
  (+0.218 accuracy in an early sample-efficiency window, 10/10 trials,
  isomorphic target) — but a `random_differentiated` control (slots
  initialized at the same magnitude as trained ones, but never trained on
  anything) revealed that **most of that gap is a confound**: any
  non-near-zero starting point beats `scratch`'s clustered-near-zero init
  by +0.088 regardless of domain match, because `scratch`'s slots start
  predicting almost identically and the belief filter has nothing to
  differentiate on. Subtracting the confound leaves a real,
  structure-specific effect that *is* larger for the isomorphic target
  than the non-isomorphic control (+0.131 vs. +0.071 early-window, 10/10
  vs. 8/10 trial wins) but does not fully vanish on the control, and
  washes out to small margins (+0.016 / +0.010) by the end of a full run.
  **Verdict: weak, partial support — a real sample-efficiency effect that
  is substantially smaller than a naive comparison would suggest, not a
  clean "transfer works" result.** Full decomposition and the reasoning
  behind the confound in the linked RESULTS.md.
- **Falsification:** no sample-efficiency advantage over from-scratch
  training, or the advantage disappears once the domains are made
  non-isomorphic (a critical control condition — run this control, not
  just the positive case). **Not strictly met** — a structure-specific
  advantage survives the non-isomorphic control, smaller and less
  consistent (roughly halved, win rate down from 10/10 to 8/10) but
  nonzero — so report this as partial support with a large, quantified
  confound, not as confirmation or as falsification.

## 7a. Experiment 4 — Frame discovery, not just frame selection

**Status: run — the cleanest positive result of the four experiments.**
`discovering_rf` recovers held-out accuracy from 0.701 (before discovery)
to **0.914** (after), beating the `random_discovery` confound control
(0.786) by +0.128 with zero false discoveries across both agents (10 and
22 total triggers respectively). Unlike experiments 1 and 3, the control
here *confirmed* a real, structure-specific effect rather than mostly
explaining it away — spot-checking the actually-fitted frames confirmed
they land close to the true held-out `(baseline=0.7, direction=
lower_is_better)`. Hyperparameters (window=40, `min_accuracy=0.85` — a
margin below experiment 1's observed steady-state, not the internal
`error_rate` naively — `alpha=0.01`) were derived analytically via exact
binomial tail probabilities before running anything, predicting ~4.6e-5
false-positive rate/window and ~21% per-window detection power; the
observed result matches that prediction closely. Full numbers and the
actual fitted baselines in
[experiments/exp04_frame_discovery/RESULTS.md](../experiments/exp04_frame_discovery/RESULTS.md).
**This means the held-out-frame limitation found in experiment 1 is not
fundamental — a crude, hand-designed discovery mechanism recovers most of
the lost accuracy** — but two follow-up tests, run in the same session,
sharpened the honest scope of that claim considerably:

- **Noise sweep**: the fixed `min_accuracy=0.85` threshold does not
  degrade gracefully outside the noise level it was calibrated for. Below
  and at σ=0.05 the result holds (false-discovery rate 0.000 throughout).
  Above it, the dominant failure mode is not reduced detection power (the
  original prediction) but a **specificity collapse**: false-discovery
  rate rises to 0.25 at σ=0.10, 0.645 at σ=0.20, 0.763 at σ=0.40, and by
  σ=0.20 the mechanism has stopped helping at all (post-discovery accuracy
  ≈ pre-discovery accuracy) because the candidate list fills with
  noise-driven junk before a useful frame can be fit. This tracks
  experiment 1's own noise boundary (σ≲0.1-0.2) closely.
- **Two simultaneously-missing regimes**: revealed a more precise
  characterization of the mechanism than "detects novel regimes." One
  held-out frame was discovered in 9/10 seeds (replicating the original
  result); a second, chosen to be just as parametrically distinct, was
  discovered in only 2/10 — because its true rule happened to agree with
  the known (wrong) frames' predictions ~90% of the time by structural
  coincidence (checked directly), versus 40% for the reliably-discovered
  one. **The mechanism detects regimes that are behaviorally
  distinguishable under the available reward signal, not regimes that are
  merely parametrically different** — a real, structural limitation shared
  with the established changepoint-detection and open-set-recognition
  literature this borrows from (`docs/related-work.md` §8a), not a defect
  specific to this implementation.

Both follow-ups run in
[experiments/exp04_frame_discovery/RESULTS.md](../experiments/exp04_frame_discovery/RESULTS.md).

Original motivation, unchanged: experiment 1's held-out-frame test
([RESULTS.md](../experiments/exp01_frame_conditioning/RESULTS.md)) showed
`rf_aware` loses 0.181 accuracy the instant the active regime isn't in its
fixed candidate list, while baselines with no such list barely notice.
Every experiment on `rf_aware` so far has assumed the candidate frame set
is given and complete. This experiment asks whether that assumption can be
relaxed at all, in the narrowest possible way.

- **Hypothesis:** an agent that (a) monitors its own recent reward rate as
  a fit-quality signal, (b) triggers a "discovery" step when that signal
  drops persistently below what its known frames should produce, and
  (c) fits a new candidate frame (a `(baseline, direction)` pair) from a
  window of recent `(raw, reward, own-prediction)` triples — using the
  same reward-implies-label trick `LearnedEmbeddingAgent` already uses —
  recovers materially more of the held-out-frame accuracy loss than a
  static-candidate-list `rf_aware`, without spuriously growing its
  candidate list when no novel regime is actually present.
- **Established theory this borrows from, and how it differs:**
  - Change-point detection: Adams & MacKay, *Bayesian Online Changepoint
    Detection*, arXiv:0710.3742, 2007. Their message-passing posterior
    over "time since the last changepoint" is the principled version of
    step (b)'s "recent fit quality dropped" trigger; the mechanism
    proposed here is a much cruder rolling-window heuristic, not their
    exact-inference algorithm. If the crude version doesn't work
    reliably, this is the natural upgrade path.
  - Growing a hypothesis space online: Dirichlet process / Chinese
    restaurant process mixture models (Ferguson 1973; Neal, *Markov Chain
    Sampling Methods for Dirichlet Process Mixture Models*, J. Comp.
    Graph. Stat., 2000) let the number of mixture components grow
    nonparametrically as data demands it. Step (c) is a hand-rolled,
    single-shot version of this idea (fit one new component when
    triggered, not a full nonparametric posterior over how many
    components should exist) — accept that as a real limitation, not an
    oversight, given this repo's "lightweight dependencies" constraint.
  - Recognizing when input doesn't belong to any known class: open-set
    recognition / novelty detection (see the survey landscape in
    arXiv:2312.08785 and arXiv:2110.14051). Step (b) is this problem in
    its simplest possible form (one scalar fit-quality signal, not a
    learned rejection boundary).
  - **None of this theory is being implemented in its full form here.**
    The point of citing it is to be honest that a crude heuristic trigger
    and a crude grid-search fit are being tested first, precisely because
    `research-agenda.md` §21/§30 call for the smallest experiment before
    the more ambitious mechanism, not to claim the established machinery
    was actually built.
- **Design:** reuse `FrameSwitchEnv` and the exact held-out-frame setup
  from experiment 1 (`held_out_frame.py`) — `rf_aware` still starts
  knowing only 3 of 4 frames, the environment still switches among all 4.
  Add a `DiscoveringRFAgent`: same Bayesian belief filter as `rf_aware`,
  plus a rolling reward-rate window and a threshold trigger; on trigger,
  fit a new `(baseline, direction)` pair via a small grid search over
  recent implied labels, append it to the candidate list with a modest
  initial belief share, and continue as normal.
- **Critical control, planned from the start (the lesson from experiment
  3):** a `RandomDiscoveryAgent` that triggers on the identical condition
  but appends a *randomly generated* frame instead of a fitted one. If
  this control recovers nearly as much accuracy as the fitted version,
  the result is really "having a growable candidate list helps" — the
  same "any extra capacity helps regardless of whether it's informed"
  confound experiment 3 found — not evidence that discovery is finding
  anything real. Do not run the positive case without this control; build
  both from the first commit, not as an afterthought once a clean number
  shows up.
- **Metrics:** (1) accuracy during held-out-frame-active periods, split
  into before-first-discovery vs. after — the recovery this experiment is
  actually about; (2) false-discovery rate — how often the trigger fires
  while the active frame *is* one of the 3 known ones (just noisy), since
  a mechanism that isn't specific enough will pollute the candidate list
  and dilute belief across near-duplicate frames; (3) candidate-list size
  over time — does it stabilize once the true regime set is covered, or
  keep growing.
- **Falsification:** no accuracy recovery relative to static `rf_aware`
  on the held-out regime, or the fitted-discovery agent's recovery is not
  meaningfully larger than the random-discovery control's — either result
  means this narrow heuristic doesn't do what it claims, and the next
  step would be the Adams & MacKay-style or Dirichlet-process-style
  upgrade instead of tuning this version further. **Neither happened —
  fitted discovery beat the random control by +0.128 (0.914 vs. 0.786),
  and both beat the pre-discovery baseline (0.701).** This narrow
  heuristic does what it claims within the tested regime; the upgrade
  path is not currently needed, though it remains the honest next step
  once a richer frame family (more than one `(baseline, direction)` pair
  missing at once, or a continuous drift) is tested.

## 7b. Experiment 5 — Regime-change detection as a core kernel capability (Phase 4)

**Status: run.** The first work on the master context's Phase 4
("Temporal Intelligence": state histories, trajectories, change
detection, regime detection, temporal comparison). Unlike experiments 1-4
(RL research scripts under `experiments/`), this phase's deliverable is a
genuine, reusable **kernel** primitive:
`transintelligence/reasoning/temporal/` was, before this, a single-line
docstring stub (`transintelligence/reasoning/temporal/__init__.py`) — the
`TemporalReasoner` protocol in `reasoning/interfaces.py` declared no
methods at all. `State`/`StateHistory` (Phase 1) already provided
`state_at`/`trajectory` (point-in-time lookup, range query) but nothing
*derived* — no way to ask "when did this change" or "what regimes did
this go through."

- **Hypothesis:** a self-calibrating CUSUM change detector (Page 1954,
  see `docs/related-work.md` §9a), operating on a `StateHistory`'s numeric
  `values[key]` stream, can recover known regime-change points from a
  synthetic ground-truth generative process (Hamilton-style discrete
  regime shifts, Hamilton 1989) with useful recall/precision, and degrade
  in an understood, ideally graceful way outside its calibrated
  conditions — mirroring the noise-sweep discipline from experiments 1/4
  rather than reporting one fixed configuration.
- **Design:** `CUSUMTemporalReasoner` (`transintelligence/reasoning/temporal/model.py`)
  implements `change_points`, `regime_segments`, `compare` against the
  now-filled-in `TemporalReasoner` protocol. Benchmarked in
  `experiments/exp05_regime_change_detection/` against synthetic
  `StateHistory` sequences with known regime boundaries, across a noise
  sweep and a regime-length sweep, with hyperparameters held fixed across
  each sweep (not recalibrated per condition) — the same discipline
  experiments 1 and 4 used.
- **A real calibration finding, caught before the main result, not
  after:** the "conventional" statistical-process-control starting point
  (small burn-in, `h_sigma=5`) produced a **47% false-positive rate on
  genuinely stationary data** (measured directly, 200 trials) — because a
  short self-calibrated burn-in window gives an unreliable σ estimate,
  and CUSUM is tested at every subsequent step, not once. Recalibrated to
  `burn_in=30`, `h_sigma=8.0` (measured false-positive rate ~6%) as the
  new class defaults, documented in the class docstring itself so the
  calibration travels with the code, not just this document.
- **Result — noise sweep: degrades gracefully, unlike experiment 4's
  trigger.** Recall/precision stay at ~0.97-0.99 from σ=0.01 to σ=0.10,
  only dropping to 0.887 at σ=0.20 and below 0.5 past σ≈0.4 — a smooth
  decline, not experiment 4's sharp specificity collapse. **This is
  direct evidence for the fix experiment 4's own results named but didn't
  build**: a threshold that recalibrates from locally observed noise,
  rather than one fixed at design time, degrades far more gracefully.
- **Result — regime-length sweep: a sharp, structural, and correctly
  mechanistic threshold at `burn_in`.** Recall is exactly 0.000 at
  `regime_length=10` (a third of `burn_in=30`), 0.312 at 20, then jumps to
  a clean 1.000 the moment regimes reach or exceed 30. This is not a bug
  — a regime shorter than the calibration window can never be calibrated
  on before the next change happens, and the sharp transition exactly at
  the parameter boundary confirms the mechanism behaves exactly as
  designed. States a real, structural precondition plainly: this detector
  is only meaningful for regimes expected to persist for at least
  `burn_in` observations.
- **Falsification:** would have been "no useful recall/precision at any
  noise level" or "no coherent relationship between regime length and
  detection" — neither happened; both sweeps produced clean, mechanistically
  explicable results.
- **Follow-up 1 — `regime_segments()`'s per-segment accuracy (not just
  `change_points()`'s recall/precision):** measured mean absolute error
  between each state's assigned segment mean and its true regime's mean,
  against a no-segmentation floor and a ground-truth-segmentation ceiling.
  Result: detected segmentation stays consistently ~2-2.6x worse than the
  ceiling across every noise level tested (not diverging as noise grows),
  and 60x+ better than not segmenting at low noise, still ~2.75x better
  at the noisiest level tested. Detection quality degrading under noise
  does not translate into disproportionately worse segment estimates.
- **Follow-up 2 — temporal comparison via DTW** (Sakoe & Chiba 1978, see
  `docs/related-work.md` §9a): implemented `dynamic_time_warp()` and
  `trajectory_distance()`, then constructed the literature's own
  motivating case directly rather than citing it on faith — a template
  trajectory, a time-shifted-but-identically-shaped version, and a
  same-position-but-different-shaped version. Naive same-index comparison
  ranks these **backwards** starting at a 2-step shift (through a 9-step
  shift, an 8-value window), while DTW ranks correctly throughout,
  recognizing the shifted version as a perfect match (distance 0)
  regardless of delay. DTW's own apparent breakdown past shift=10 is a
  construction artifact (the fixed-length test harness runs out of room
  to represent a pure shift at that exact point, so the trajectory
  genuinely changes shape there) — stated precisely rather than left
  implying a real DTW limitation.
- **Follow-up 3 — multi-key tracking** (Crosier, *Multivariate
  Generalizations of Cumulative Sum Quality-Control Schemes*, Technometrics
  30(3), 1988): `joint_change_points()` combines per-key z-scored
  deviations into a single scalar (Crosier's "reduce to scalar first"
  variant) before running CUSUM on it. Calibration held up unchanged
  across 1-5 keys (5.5-8% false-positive rate, no retuning needed). With
  the union-of-independent-detectors control experiments 3/4 taught this
  program to always build: joint detection **genuinely beats the union
  control** in a moderate-signal band (paired wins 9:2, 6:1, 3:0 at
  shift/noise 0.6-1.0) — a real gain from combining evidence, not the
  "more chances helps" confound — but the advantage **disappears at the
  weakest signal tested** (2:3, a wash). State the claim at its actual
  width, not universally.
- **Follow-up 4 — non-Gaussian noise:** a standard contaminated-Gaussian
  mixture (5% wide-variance outliers) more than **quadruples the
  false-positive rate** (0.08 → 0.34) while barely touching recall
  (1.00 → 0.967) — the same specificity-over-power fragility pattern
  found in experiment 4's trigger, now confirmed in a second, unrelated
  mechanism. No single citation adopted for "CUSUM's non-normal
  robustness" (unlike every other claim in this document) — the SPC
  literature here is a family of results, not one seminal paper; tested
  empirically instead of asserted from authority.
- **Follow-up 5 — gradual drift, a hypothesis tested and corrected, not
  confirmed:** the working hypothesis going in was that a slow enough
  drift might never be detected, since calibration is fixed once per
  detection cycle rather than continuously updated. **That hypothesis was
  wrong** — recall stayed at 1.00 across every ramp length tested,
  including a 1000-step ramp that never completes within an 1100-step
  series, because a fixed calibration reference guarantees any persistent
  drift eventually crosses threshold. Detection delay grows sub-linearly
  with ramp length (a real, expected cost, not a breakdown). Report this
  as "the hypothesis was wrong" plainly, not as "confirmed robust from the
  start."
- Full numbers, the calibration checks, and what still isn't tested
  (non-i.i.d. noise beyond the contaminated-Gaussian case tested;
  more than 5 keys; correlated cross-key covariance in the joint
  statistic, which the current independence assumption ignores) in
  [experiments/exp05_regime_change_detection/RESULTS.md](../experiments/exp05_regime_change_detection/RESULTS.md).

## 7c. Experiment 6 — Confounding bias and the backdoor criterion (Phase 5)

**Status: run.** The first work on the master context's Phase 5 ("Causal
+ Counterfactual Intelligence": causal graphs, hypotheses, interventions,
counterfactual simulations). Like Phase 4, this fills a genuine, previously-
empty kernel stub: `transintelligence/reasoning/causal/` was a one-line
docstring, `CausalReasoner` in `reasoning/interfaces.py` declared zero
methods, and grepping the whole repo for "intervention" returned zero
hits anywhere.

- **Hypothesis:** Pearl's backdoor criterion (1995), computed purely from
  graph structure with no data, correctly predicts which covariate
  adjustment sets remove confounding bias and which don't — including a
  *specific negative case* (adjusting for a collider) that the graph
  should flag as invalid and that should empirically show real bias, not
  just "less good than the valid case."
- **Design:** `CausalGraph` (`transintelligence/reasoning/causal/model.py`)
  implements d-separation via explicit path enumeration (Verma & Pearl,
  *Causal Networks: Semantics and Expressiveness*, UAI 1988) and the
  backdoor criterion (Pearl, *Causal Diagrams for Empirical Research*,
  Biometrika 82(4), 1995) on top of it, plus a small `ordinary_least_squares`
  utility for linear effect estimation. **Both were verified for
  correctness independently before being trusted for anything**: d-separation
  against the three canonical structures (chain, fork, collider, plus a
  collider-with-conditioned-descendant case) in
  `tests/test_causal_reasoning.py`, OLS against closed-form linear
  relationships recovered to floating-point precision — the same
  discipline Phase 4 used for DTW (verify the mechanism on textbook cases
  *before* building an experiment on top of it, not after).
- **The experiment, constructed directly rather than cited from
  authority** (the same approach as experiment 5's DTW motivating case):
  a linear SCM with confounder `Z→X, Z→Y`, a **true causal effect of X on
  Y fixed at exactly 0.0**, and a collider `W` (a common *effect* of X and
  Y, `X→W, Y→W`) — not a cause of either, and therefore graph-invalid as
  an adjustment set. Four regression conditions: naive (no adjustment),
  adjusted (condition on `Z`), collider (condition on `W`), and both.
- **Result: as clean as a falsifiable result gets — the graph's
  prediction matches the empirical bias exactly, with a genuinely
  important extra nuance.** Naive regression finds a large, entirely
  spurious effect (0.881, true value 0.0) from confounding alone.
  Adjusting for `Z` (the only backdoor-valid set) recovers ~0 (0.0065).
  Adjusting for the collider `W` instead — correctly flagged invalid by
  the graph, since `W` is a descendant of `X` — produces its own
  substantial bias (0.281), a genuinely different failure mode from
  naive, not just "less effective." **Most importantly: adding `W` on
  top of the already-correct `{Z}` adjustment makes the estimate *worse*
  (bias rises from 0.031 to 0.191, even flipping sign) — the same lesson
  experiments 3-5 already established with different mechanisms (a
  confound control, a random-discovery control, a union-of-detectors
  control): "more/any extra information" is not a safe default, and the
  graph-theoretic validity check is doing real work, not formal
  decoration.
- **Falsification:** would have been the graph's valid/invalid
  predictions failing to line up with which conditions were empirically
  unbiased, or the collider adjustment showing no worse bias than the
  valid one — neither happened. Full numbers, what isn't tested (only
  linear SCMs; only the backdoor criterion, not front-door adjustment or
  instrumental variables; no causal discovery; no per-unit counterfactual
  queries, which is `CounterfactualReasoner`, still an empty stub and the
  natural next experiment) in
  [experiments/exp06_confounding_bias/RESULTS.md](../experiments/exp06_confounding_bias/RESULTS.md).

## 7d. Experiment 7 — Per-unit counterfactual queries (Phase 5, continued)

**Status: run.** Fills `CounterfactualReasoner`
(`transintelligence/reasoning/counterfactual/`), the last remaining empty
stub among the reasoning protocols that had one before Phase 4 started
(`TemporalReasoner`, `CausalReasoner`, now `CounterfactualReasoner` all
have real implementations; `Predictor`, `Simulator`, `Planner`, `Verifier`
remain unbuilt). Experiment 6 answered a *population* question ("does X
affect Y on average, adjusting for confounders"); this answers a
genuinely different *per-unit* one ("what would THIS unit's Y have been,
had its X been different") — Pearl's third rung of the causal hierarchy.

- **Hypothesis:** Pearl's abduction-action-prediction procedure (Pearl,
  Glymour, Jewell, *Causal Inference in Statistics: A Primer*, Wiley,
  2016; computational treatment in Balke, Pearl, *Counterfactual
  Probabilities: Computational Methods, Bounds and Applications*, UAI
  1994), specialized to a linear+additive-noise SCM, correctly recovers a
  specific unit's counterfactual outcome — and does so by genuinely
  preserving that unit's own idiosyncratic noise, not just reproducing
  the population-average prediction at the new treatment value.
- **The confound this needed to control for, stated up front rather than
  discovered after a misleadingly clean number**: the "naive" alternative
  (plug the new treatment value into the fitted population regression,
  skip abduction entirely) is an *unbiased estimator of the average
  effect* — so a comparison that only checks average accuracy across many
  units risks naive looking "close enough." The real test is per-unit
  accuracy for units with nonzero residuals specifically.
- **Design:** `StructuralCausalModel` (`transintelligence/reasoning/counterfactual/model.py`)
  implements `abduct()` (closed-form residual per node) and
  `counterfactual()` (fix intervened nodes, recompute everything else
  using each node's own inferred noise). Verified against hand-computed
  cases in `tests/test_counterfactual_reasoning.py` *before* the
  experiment was built (a simple two-node chain with an exact expected
  residual and counterfactual value; two units with identical observed
  treatment but different outcomes correctly getting different
  counterfactual predictions) — the same discipline as verifying
  d-separation on canonical structures before experiment 6.
- **Result: as clean as experiment 6's, for the same reason — the
  mechanism is exact by construction, not approximately right.** With the
  *true* structural coefficients, abduction's mean absolute error is
  **0.0000** (exact recovery to floating-point precision) while the naive
  plug-in's error is 0.2413 — which matches the theoretical mean absolute
  value of the simulated exogenous noise (`0.3·√(2/π) ≈ 0.239`) almost
  exactly, confirming *why* naive is wrong: its error literally equals
  the unit-specific residual it silently discards. With *estimated*
  (OLS-fit) coefficients — the realistic case — abduction still wins
  roughly 9x (0.0284 vs. 0.2440), and **naive's error barely moves at
  all between the two conditions**, because its error source (discarding
  a unit's own residual) is structurally independent of how well the
  population parameters are estimated — no amount of additional
  observational data would ever close that gap, whereas abduction's small
  remaining error is pure finite-sample noise that does shrink with more
  data (checked directly).
- **Falsification:** would have been naive performing comparably to
  abducted (showing the "unbiased on average" property was enough in
  practice), or abducted's error with true coefficients being
  meaningfully nonzero (a bug in the closed-form residual logic) — neither
  happened. Full numbers and what isn't tested (nonlinear/non-additive-
  noise SCMs; structure/coefficients assumed known or estimated, not
  discovered; single intervention point only; no computational comparison
  against Rubin's potential-outcomes framework) in
  [experiments/exp07_counterfactual_queries/RESULTS.md](../experiments/exp07_counterfactual_queries/RESULTS.md).

## 7e. Experiment 8 — Causal discovery: recovering structure instead of assuming it (Phase 5, continued)

**Status: run.** Experiments 6 and 7 both assumed the causal graph and
structural equations were given. This experiment asks the prior question:
can that structure be recovered from data at all, using the smallest
constraint-based mechanism (Spirtes & Glymour's PC algorithm, 1991) that
could produce a falsifiable claim about it?

- **Hypothesis:** PC-style skeleton recovery (via Fisher-z partial-
  correlation independence tests) plus collider/v-structure orientation
  for unshielded triples recovers the correct undirected skeleton as
  sample size grows, correctly orients genuine unshielded colliders, and
  correctly declines to orient shielded ones — all without ever treating
  pure noise as structure.
- **The confound this needed to control for, stated up front:** a
  discovery procedure that just proposes more edges as sample size (and
  therefore statistical power) grows would produce clean-looking positive
  results for the wrong reason. The negative control — four mutually
  independent variables, sample size swept the same way as the positive
  cases — has to show a false-edge rate that stays flat or shrinks, not
  one that grows alongside the true-positive recall in the positive
  cases.
- **A structural subtlety identified before writing any experiment code**:
  experiment 6's confounding graph (`Z→X, Z→Y, X→Y, X→W, Y→W`) has a
  collider `W` that is **shielded** — `X→Y` is also a direct edge — so
  standard v-structure orientation cannot and should not fire there. That
  graph is still valid for testing *skeleton* recovery (with `TRUE_EFFECT`
  changed from experiment 6's deliberate 0.0 to a nonzero 0.5, so the
  `X-Y` edge is real; see the module docstring in
  `experiments/exp08_causal_discovery/run.py` for why), but a *separate*,
  dedicated unshielded-collider graph (`A→B←C`, `A` and `C` independent)
  was built specifically to exercise collider orientation at all.
- **Design:** `discover_skeleton`/`orient_colliders`
  (`transintelligence/reasoning/causal/model.py`) implement PC's skeleton
  phase (remove an edge x-y once some conditioning set from x's/y's
  neighbors makes them independent, via `partial_correlation` +
  `fisher_z_independence_test`) and its collider-orientation phase.
  Three of Meek's four orientation-propagation rules (UAI 1995) are
  implemented as a follow-up, below; see
  [related-work.md §3c](related-work.md#3c-causal-discovery-constraint-based-structure-recovery-experiment-8-phase-5).
  **Verified against three noiseless-mechanism hand-checks before being
  trusted for anything**: a chain (correct skeleton, correctly left
  unoriented), an unshielded collider (correct skeleton, correctly
  oriented), and a shielded triple mirroring experiment 6's exact shape
  (correctly left unoriented despite `W` genuinely being a collider) —
  all in `tests/test_causal_reasoning.py`.
- **Result: all three qualitative predictions held at every sample size
  tested (100 to 3000), across 20 seeds each.** On the confounding graph,
  skeleton precision is ≈1.000 throughout and recall rises from 0.750
  (n=100) to 1.000 (n≥1000); `W` is never falsely oriented as a collider,
  0/80 trials across all four sample sizes. On the dedicated unshielded
  graph, skeleton recovery and collider orientation are both exact
  (20/20) at every sample size including the smallest. On the
  independent-variables negative control, the false-edge rate does not
  grow with N — it shrinks (0.100 mean false edges/trial at n=100,
  0.000 at n=3000), ruling out the "just finds more structure with more
  power" failure mode this experiment was specifically built to catch.
- **Falsification:** would have been the false-edge rate climbing with N
  in the negative control (statistical-power artifact, not real
  discovery), `W` ever being falsely oriented (a bug in the shielded-
  triple exclusion), or the dedicated collider graph failing to orient
  correctly even at large N (a bug in the orientation rule itself) — none
  happened. Full numbers and what isn't tested (nonlinear dependencies
  with zero linear partial correlation; graphs larger than 4-5 nodes; no
  comparison against score-based discovery methods) in
  [experiments/exp08_causal_discovery/RESULTS.md](../experiments/exp08_causal_discovery/RESULTS.md).

**Follow-up: Meek's orientation-propagation rules (R1-R3).** Collider
orientation alone can only ever find direct v-structures — some graphs
are fully identifiable from their CPDAG but need edge-orientation
*propagation* to recover completely. `apply_meek_rules`
(`transintelligence/reasoning/causal/model.py`) implements three of
Meek's four rules; R4 is not implemented, and provably could not fire in
this pipeline at all — it only propagates externally-supplied background
knowledge, which this implementation has no mechanism to provide (R1-R3
alone are established to be complete for the no-background-knowledge
case: Perkovic, Textor, Kalisch, Maathuis, UAI 2017). Verified against
three hand-computed cases (R1, R2, R3 each in isolation) plus a
two-hop-propagation case confirmed via real simulated data through the
full `discover_skeleton`→`orient_colliders`→`apply_meek_rules` pipeline,
all in `tests/test_causal_reasoning.py`, before the follow-up experiment
(`experiments/exp08_causal_discovery/meek_rules.py`) measured it under
sampling noise. On a fully-identifiable collider-then-chain graph
(`A→C←B, C→D→F`), collider orientation alone is capped at exactly 0.500
recall (2 of 4 true edges, the hard ceiling from only ever finding
v-structures); with Meek's rules, recall reaches 1.000 once the skeleton
is reliably correct (n≥1000). Investigating an apparent "wrong
orientation" at one n=300 seed (rather than reporting the clean mean and
moving on) found the actual cause was a skeleton-recovery error at that
seed propagating downstream, not a bug in the orientation logic — given a
correct skeleton, `apply_meek_rules` never produced a wrong orientation
across all 80 trials tested. A negative control (a plain collider-free
chain, Markov-equivalent to a fork and a reverse chain, genuinely
undetermined by any amount of data) confirmed zero edges get oriented at
any sample size — Meek's rules propagate from real evidence, they don't
invent orientations. Full numbers in
[experiments/exp08_causal_discovery/RESULTS.md](../experiments/exp08_causal_discovery/RESULTS.md)
§4.

## 7f. Experiment 9 — Instrumental variables and front-door adjustment: identification under an unobserved confounder (Phase 5, continued)

**Status: run.** Experiments 6 and 8 both assumed (or discovered) an
observed causal graph — every variable the backdoor criterion needed to
adjust for was actually in the dataset. This experiment asks what
happens when the confounder itself is never observed at all, the case
neither the backdoor criterion nor discovery can handle, using the two
classic alternative identification strategies.

- **Hypothesis:** two-stage least squares (via a valid instrument) and
  front-door adjustment (via a fully-mediating observed variable) both
  recover the true effect despite a genuinely unobserved confounder — and
  both fail in their own specific, theory-predicted ways when their
  respective assumptions don't hold (weak instrument strength; the
  confounder also reaching the mediator).
- **The confound each part needed to control for, stated up front:**
  showing 2SLS work at one comfortable instrument strength, or front-door
  work under one assumption-satisfying setup, would each invite the
  obvious follow-up question a skeptical reviewer would ask — what
  happens when that isn't true? Bound, Jaeger, Baker (JASA 90(430), 1995)
  make a specific prediction about weak instruments; Pearl's front-door
  criterion (1995) states a specific structural precondition. Both parts
  test the failure directly rather than only asserting it as a caveat.
- **Design:** `two_stage_least_squares` and `front_door_adjustment`
  (`transintelligence/reasoning/causal/model.py`) reuse
  `ordinary_least_squares` for both stages/steps.
  **Verified against a hand-checked simulation before being trusted for
  anything**: both mechanisms recover close to their true effects against
  heavily biased naive OLS baselines on data with a genuinely unobserved
  confounder never passed to either function, in
  `tests/test_causal_reasoning.py`.
- **Result, part 1 (2SLS):** with a strong-to-moderate instrument
  (strength 0.9 down to 0.2), 2SLS's bias stays under 0.10 against naive
  OLS's 0.41-0.83 — roughly a 30x reduction at the strongest setting.
  **Below that, 2SLS doesn't degrade gracefully — it becomes wildly
  unstable and worse than naive**: at instrument strength 0.05,
  individual estimates ranged from -96.3 to +29.4 (stdev 18.7, a ~390x
  increase over the strength-0.9 case), consistent with — and a more
  dramatic version of — the weak-instrument pathology Bound, Jaeger, and
  Baker documented.
- **Result, part 2 (front-door adjustment):** when its structural
  assumption holds (the confounder doesn't reach the mediator), front-door
  adjustment recovers the true effect almost exactly (bias 0.0094 against
  naive's 0.8807). **When that assumption is violated — the confounder
  given a direct effect on the mediator, using the identical adjustment
  call — the estimate is nearly as biased as naive** (0.9304, a ~99x jump
  from the valid condition), the same "adjusting incorrectly is worse,
  not neutral" lesson experiment 6's collider condition established for
  the backdoor criterion.
- **Falsification:** would have been 2SLS remaining reliable regardless
  of instrument strength (no weak-instrument pathology reproduced), or
  front-door adjustment remaining approximately unbiased even when its
  assumption was violated (meaning the "fully mediating, unconfounded
  mediator" precondition doesn't actually matter in practice) — neither
  happened. Full numbers and what isn't tested (only just-identified
  2SLS; no automated weak-instrument diagnostic; only one of front-door's
  three structural preconditions was stress-tested; linear structural
  equations only; no comparison to sensitivity-analysis approaches like
  Rosenbaum bounds) in
  [experiments/exp09_iv_and_frontdoor/RESULTS.md](../experiments/exp09_iv_and_frontdoor/RESULTS.md).

## 7g. Experiment 10 — Nonlinear structural equations: does abduction stay exact, and does linear effect estimation break? (Phase 5, continued)

**Status: run.** The last of the three stated gaps from `docs/findings.md`'s
"what isn't tested yet" list. Experiments 6, 7, and 9 all assumed linear
structural equations, the same simplification `reasoning/causal/` and
`reasoning/counterfactual/` made from the start. This asks the two
questions that assumption was hiding: does linear-adjusted OLS effect
estimation actually break under true nonlinearity, and does
counterfactual abduction — argued from the start to need only additive
noise, not linearity — actually stay exact when the structural equation
genuinely is nonlinear?

- **Hypothesis:** `StructuralEquation` can be generalized to an arbitrary
  `nonlinear_fn` with zero changes to `abduct()`/`counterfactual()`
  themselves (only to `predict()` and parent-name lookup), because
  Pearl's abduction step is a residual against *any* additive-noise
  function, not specifically a linear one. Separately: `reasoning/causal/`'s
  OLS-based effect estimation, which does assume linearity, will be
  substantially biased on a genuinely nonlinear relationship — and,
  because a nonlinear relationship has no single well-defined scalar
  "effect" at all, a single linear coefficient cannot represent a
  heterogeneous (unit-dependent) true effect regardless of how it's
  estimated.
- **The confound this needed to control for:** showing linear-adjusted
  OLS fail on a nonlinear model alone could just mean the experimental
  setup is broken, not that nonlinearity specifically causes it. A
  `GAMMA2=0.0` control (the truth really is linear) has to show
  linear-adjusted OLS closely matching the true effect there.
- **Design:** `StructuralEquation.nonlinear_fn`
  (`transintelligence/reasoning/counterfactual/model.py`), a backward-
  compatible generalization (existing linear call sites unchanged).
  **Verified against a hand-computed quadratic case before being trusted
  for anything**: `A -> B, B = A² + 3A + noise`, exact residual and
  exact counterfactual recovery, in `tests/test_counterfactual_reasoning.py`.
  The experiment reuses experiments 6/7/9's exact confounding-graph shape
  (`Z→X, Z→Y, X→Y`) with `Y = 0.8Z + 0.3X + GAMMA2·X² + noise`, and
  computes the **true average shift effect exactly** (not estimated) via
  `counterfactual(observed, {"X": x+1})["Y"] - observed["Y"]` per unit —
  each unit's own noise is identical in both terms and cancels
  algebraically, the same "exact by construction" trick experiment 7
  used for the linear case.
- **Result: both predictions held, and the nonlinear case's mismatch was
  more dramatic than a simple "biased estimate" — the linear model's
  output was nearly indistinguishable from its own linear-truth control
  run despite the true effect nearly tripling.** In the linear control
  (`GAMMA2=0.0`), linear-adjusted OLS matched the true effect closely
  (gap 0.0121). In the nonlinear condition (`GAMMA2=0.6`), the gap grew
  to **0.5799** — but the linear-adjusted estimate itself (0.3255) barely
  moved from the control's (0.3121), because `X`'s near-zero skew makes
  `Cov(X, X²) ≈ 0`: OLS's linear coefficient is nearly blind to the
  quadratic contribution rather than reporting a scaled-down version of
  it. At five fixed reference points, the true shift effect ranged from
  **-1.5 to +3.3** (even flipping sign) while the linear model predicted
  the same +0.3255 at every one of them — not a quantitative miss, a
  category error: a linear coefficient cannot represent an effect that
  depends on where a unit starts.
- **Falsification:** would have been the linear control condition itself
  showing a large gap (meaning the experimental setup, not nonlinearity,
  was the problem), or the nonlinear hand-computed abduction/
  counterfactual test failing to match hand arithmetic exactly (a bug in
  the generalization, since it's supposed to require zero logic changes)
  — neither happened. Full numbers and what isn't tested (only one
  functional form tested; non-additive noise still unattempted; no
  nonlinear effect-estimation alternative added; functional form is
  given, not discovered) in
  [experiments/exp10_nonlinear_scm/RESULTS.md](../experiments/exp10_nonlinear_scm/RESULTS.md).

## 7h. Experiment 11 — World models: does explicit dynamics modeling beat model-free value estimation? (Phase 6)

**Status: run.** The first Phase 6 work
(`docs/master-context.md` §13/§19: `M(S_t, A_t) → S_{t+1}`, a *transition*
model, useful because predicted outcomes of different actions from the
same state can be compared). `transintelligence/world_models/` was, before
this, a directory that didn't exist at all, and `Predictor` in
`reasoning/interfaces.py` declared no methods — the same starting point
every other empty-stub reasoning protocol had before its phase started.

- **Hypothesis:** an agent that learns a forward dynamics model online
  and plans by simulating candidate actions' predicted next-states beats
  a model-free baseline with the same state access but no explicit
  dynamics model — specifically because decomposing "learn the (easy,
  linear) transition dynamics, then apply the (known, exact) reward
  formula" differs from directly fitting a (harder, nonlinear) value
  function with the same linear tool.
- **The confound this needed to control for, stated up front:** showing
  a state-aware agent beat a state-blind one would only prove using
  context helps at all — trivial, and not Phase 6's actual claim. A
  second, state-aware-but-model-free baseline (`model_free_linear_q`),
  given identical state access and the identical `ordinary_least_squares`
  tool, isolates the real claim: does modeling *dynamics* specifically
  (not just *using* state) matter?
- **Design:** `environments/transworld/resource_control_env.py`'s
  `ResourceControlEnv` — each trial draws a fresh state, the agent picks
  a discrete "nudge" action, `reward = -(next_state - target)²`. True
  dynamics are linear in state given the action; true reward is
  quadratic in the resulting state — the source of the structural
  asymmetry between the two state-aware conditions. `LinearDynamicsModel`
  (`transintelligence/world_models/model.py`) fits a per-action linear
  transition model via `ordinary_least_squares`, reused from Phase 5.
  **Verified against a hand-computed exact-linear-dynamics case (two
  actions, different known intercepts/slopes, recovered to
  floating-point precision) before being trusted for anything**, in
  `tests/test_world_models.py`. Four conditions: `state_blind` (floor),
  `model_free_linear_q` (fair, identically-tooled baseline),
  `world_model` (the treatment), `oracle_dynamics` (given the true
  dynamics exactly, isolating cost-of-learning the way experiment 1's
  `true_oracle` did).
- **Result: `world_model` matched the oracle ceiling almost exactly
  (regret -0.0002), while the identically-tooled `model_free_linear_q`
  fell far short (regret 3.5421) — roughly 3.5x further from optimal
  despite seeing the same states and using the same regression tool.**
  `state_blind` had the worst regret (6.8280), as expected. Investigated
  *why* the fair baseline still underperformed substantially, rather than
  reporting the gap and moving on: each action's true reward is a
  downward parabola in state, peaking where that action lands the state
  exactly on target — a linear fit is forced to be monotonic, so it can
  track the parabola's rising side but has no way to represent the
  peak-then-decline, systematically misranking actions past their
  optimal zone. **The same lesson experiment 10 established for effect
  estimation — a linear coefficient cannot represent a relationship that
  depends on where a unit starts — reappears here in a planning setting**:
  the value surface isn't just nonlinear, it's non-monotonic, which no
  amount of data lets a linear fit represent, while `world_model` never
  has to fit that curvature at all — it fits the (truly linear)
  transition and applies the exact known reward formula to the result.
- **Falsification:** would have been `model_free_linear_q` matching
  `world_model`'s performance (meaning the dynamics/reward decomposition
  doesn't actually matter, just state access does), or `world_model`
  failing to reach the oracle ceiling despite ample data (a bug in the
  learned-dynamics mechanism) — neither happened. Full numbers and what
  isn't tested (single-step lookahead only, not multi-step planning;
  genuinely linear dynamics; discrete small action set; stationary
  environment) in
  [experiments/exp11_world_model_planning/RESULTS.md](../experiments/exp11_world_model_planning/RESULTS.md).

**Follow-up: does a nonlinear model-free baseline close the gap?** The
main result's explanation makes a specific, checkable prediction — a
function class that CAN represent the true reward's shape should close
the gap. `model_free_quadratic_q`
(`experiments/exp11_world_model_planning/quadratic_baseline.py`, same
`ordinary_least_squares` tool with an added quadratic feature) confirmed
it: regret dropped from `model_free_linear_q`'s 3.5421 to **0.0144**,
landing at essentially the same level as `world_model` (-0.0002). This
confirms the original mechanism (linear cannot represent a peak) was the
actual cause, not an unaccounted-for difference between conditions — and
sharpens what `world_model`'s real advantage in this environment is: not
unbeatable accuracy, but getting that accuracy "for free" from correctly
specifying the *easier* (linear) part of the problem, rather than needing
to discover the *harder* (quadratic) part empirically with the exactly
right feature set. Full numbers in
[experiments/exp11_world_model_planning/RESULTS.md](../experiments/exp11_world_model_planning/RESULTS.md)
("Follow-up" section).

## 7i. Experiment 12 — Multi-step planning: does receding-horizon control beat greedy lookahead? (Phase 6, continued)

**Status: run.** Experiment 11's environment (`ResourceControlEnv`) has
pure translation dynamics with no delayed effects, so greedy 1-step
lookahead is already globally optimal there — nothing for multi-step
planning to improve on. This experiment builds the smallest environment
where that stops being true.

- **Hypothesis:** a receding-horizon planner (simulate several steps
  ahead using the learned dynamics model, execute only the first action,
  replan every step — Richalet et al.'s Model Predictive Heuristic
  Control, 1978) beats greedy 1-step lookahead specifically when actions
  have a delayed effect a 1-step model is structurally blind to.
- **The confound this needed to control for, stated up front:** a
  planning-horizon comparison could get conflated with a model-quality
  difference if the two conditions happen to learn dynamics of different
  quality. A 2×2 factorial design (`greedy`/`mpc` × `learned`/`oracle`)
  isolates the two questions: does horizon matter, and does model
  quality matter, checked independently rather than bundled into one
  number.
- **Design:** `environments/transworld/delayed_control_env.py`'s
  `DelayedControlEnv` — each nudge splits between an immediate effect
  (`lag_weight=0.5`) and a delayed one landing on the position after
  next; episodes persist for `HORIZON=5` steps with reward only at the
  final step. `choose_action`
  (`experiments/exp12_multistep_planning/run.py`) enumerates action
  sequences of depth `min(lookahead, remaining_steps)` and scores each
  by simulated final-position distance. **Verified by hand before being
  trusted for anything**: given known dynamics and a specific starting
  point, both `lookahead=1` and `lookahead=2` were checked against a
  hand-computed optimal first action, and capping lookahead at the
  episode's remaining steps was confirmed to make `mpc` reduce to
  exactly `greedy`'s choice at the final step, in
  `tests/test_exp12_multistep_planning.py`.
- **Result: `mpc` beat `greedy` by almost exactly the same margin in
  both the learned pair (0.0246→0.0204, gap 0.0042) and the oracle pair
  (0.0241→0.0200, gap 0.0041) — confirming the advantage is about
  planning horizon, not an accident of model quality differing between
  conditions.** `mpc_learned` beat `greedy_learned` in 14 of 15 seeds,
  a consistent effect. Learned and oracle dynamics performed almost
  identically within each horizon pair, consistent with experiment 11's
  finding that `LinearDynamicsModel`-style linear dynamics are recovered
  almost exactly given enough data. **The effect size is real but
  modest (~17% relative reduction in mean squared final-distance), and
  reported at that size rather than inflated**: because every step gives
  full state feedback and lets the agent replan, a myopic policy already
  self-corrects reasonably well over several free steps — multi-step
  planning's advantage here is in efficiency, not in avoiding
  catastrophic, uncorrectable mistakes.
- **Falsification:** would have been `mpc` failing to beat `greedy`
  despite the delayed-effect structure (meaning the mechanism doesn't
  actually help, or the environment doesn't actually require lookahead),
  or the learned/oracle gap differing substantially between the greedy
  and mpc pairs (meaning the comparison was contaminated by a
  model-quality confound rather than isolating planning horizon) —
  neither happened. Full numbers and what isn't tested (only a 2-step
  lookahead; a single fixed lag structure; exhaustive search over action
  sequences, not scalable to larger action sets; no nonlinear/momentum-
  based delay structure; stationary environment) in
  [experiments/exp12_multistep_planning/RESULTS.md](../experiments/exp12_multistep_planning/RESULTS.md).

## 7j. Experiment 13 — Combining world models with regime-change detection (Phase 6, continued)

**Status: run.** A synthesis experiment, not a new-mechanism one: does
`CUSUMTemporalReasoner` (Phase 4, experiment 5) composed with
`LinearDynamicsModel` (Phase 6, experiment 11) actually work as expected
when wired together — an agent that detects a mid-experiment shift in
the true dynamics and discards its stale data should recover performance
an agent that never adapts loses.

- **Hypothesis:** detection-triggered adaptation beats both a
  never-adapts baseline and a naive always-use-recent-data heuristic
  (`sliding_window_baseline`), because it discards stale data only when
  there's genuine evidence the world changed, not on a fixed schedule.
- **The confound this needed to control for, stated up front:** showing
  CUSUM-triggered adaptation beat never-adapting alone would only prove
  *some* adaptation helps — the same trivial trap experiments 3, 8, and
  11 each had to control for. `sliding_window_baseline` isolates whether
  explicit detection specifically is doing real work.
- **What actually happened, honestly reported rather than re-run until
  it looked clean:** the first version tested one fixed, mild regime
  shift (`post_shift_scale=0.4`) and found `oracle_adapts` (told the true
  shift trial exactly) performed statistically indistinguishably from
  `never_adapts` — the opposite of the hypothesis. Investigating why
  (rather than reporting a null result and stopping) found the actual
  mechanism: this environment's state range is wide relative to its
  nudge magnitudes, so most trials start far enough from target that a
  stale and a correctly-calibrated model pick the *same* largest-available
  action regardless of the exact scale factor — mild miscalibration
  rarely changes which action is locally best, while discarding a large
  body of converged prior data for a small, noisy post-shift refit has a
  real, visible cost (confirmed directly: `oracle_adapts` starts *worse*
  than `never_adapts` in the first 100-trial chunk after the shift).
  This turned the experiment into a deliberate two-severity sweep, not a
  single condition.
- **Design:** `RegimeShiftControlEnv`
  (`environments/transworld/regime_shift_control_env.py`) — like
  experiment 11's environment, but nudge magnitudes silently rescale at
  an unknown trial. Four conditions sharing the identical greedy 1-step
  policy and `LinearDynamicsModel` fitting tool, differing only in which
  transitions each trains on: `never_adapts`, `oracle_adapts` (given the
  true shift trial exactly), `sliding_window_baseline` (always the most
  recent transitions), `cusum_detects_and_adapts` (tracks its own
  model's prediction residuals as a `StateHistory`, calls
  `CUSUMTemporalReasoner.change_points()` on that series, discards
  pre-detection data once triggered).
- **Result: severity-dependent, exactly as the mechanism predicts once
  investigated.** At mild attenuation (`0.4`), all four conditions
  perform similarly post-shift (-5.08 to -5.41) — the honest null case,
  reported as such. At a severe sign-flip (`-1.0`, the actuator's effect
  reverses direction entirely), `never_adapts` collapses (post-shift
  reward -15.49, ~8x worse than pre-shift), `oracle_adapts` recovers
  almost completely (-2.36), and **`cusum_detects_and_adapts` (-2.62)
  beats `sliding_window_baseline` (-3.00)** — the confound-controlled
  result the experiment needed, landing closer to the oracle ceiling
  than the simpler heuristic does. Detection: 15/15 seeds detected the
  severe shift (mean latency 23.0 trials) vs. 13/15 for the mild one
  (mean latency 36.5 trials) — a larger shift produces an
  easier-to-detect signal, as expected. **3/15 seeds (20%) showed a
  false-positive detection before the true shift, in both severities —
  notably higher than experiment 5's own ~6% baseline**, traced to a
  real, reported cause: this experiment's residual stream comes from a
  periodically-refit model, whose own re-fits introduce small genuine
  jumps a stationary raw signal (like experiment 5's) wouldn't have.
- **Falsification:** would have been detection-triggered adaptation
  failing to beat the sliding-window baseline even at severe shift
  severity (meaning explicit detection isn't earning its complexity at
  all), or the mild-severity null finding turning out to be a bug rather
  than a real, investigated mechanism — neither happened once both
  severities were tested and the discrepancy was traced to its actual
  cause. Full numbers and what isn't tested (only two severities; a
  single fixed shift trial and post-shift window length; only tested
  with the linear dynamics model, not experiment 10's nonlinear
  generalization or experiment 12's multi-step planner; a single
  sliding-window size) in
  [experiments/exp13_regime_shift_world_model/RESULTS.md](../experiments/exp13_regime_shift_world_model/RESULTS.md).

**Follow-up: does refitting more often reduce the false-positive rate?**
The diagnosis above makes a specific, checkable prediction — refitting
the dynamics model more frequently should keep its residuals closer to
the true dynamics continuously, reducing the spurious jumps a stale,
infrequently-updated model produces. Checked directly
(`experiments/exp13_regime_shift_world_model/refit_interval_calibration.py`),
measuring both false-positive rate AND true-detection performance, since
a fix that also cripples detection wouldn't be a fix. The prediction
held (`refit_interval=10` roughly halves the false-positive rate, 4/15 →
2/15) but it's a genuine tradeoff, not a free improvement: true detection
under the severe shift also drops (15/15 → 13/15), though successful
detections become far faster (23.0 → 2.5 trials latency) — a
precision/recall-style tradeoff reported as the honest answer, not
resolved to a single "correct" value, since which side matters more
depends on the application. Full numbers in
[experiments/exp13_regime_shift_world_model/RESULTS.md](../experiments/exp13_regime_shift_world_model/RESULTS.md)
("Follow-up" section).

## 7k. Experiment 14 — Beam search for scalable receding-horizon planning (Phase 6, continued)

**Status: run.** `RecedingHorizonPlanner`'s exhaustive search
(experiments 12-13) was flagged from the start as `O(len(actions)^depth)`
— not scalable beyond the 6 actions and depth 2 those experiments used.
This tests the expected fix directly, and ends up finding something
sharper than expected by investigating a result that looked wrong rather
than reporting it.

- **Hypothesis (the expected result):** beam search (Lowerre, 1976)
  recovers near-exhaustive decision quality using dramatically fewer
  `transition_fn` calls, letting planning scale to depths exhaustive
  search can't reach.
- **The confound this needed to control for:** showing beam search uses
  fewer calls proves nothing about whether it's still a good planner —
  every beam-search condition is scored against the same exhaustive-
  search ground truth at the same starting states.
- **Two real bugs caught and fixed before the actual finding could be
  trusted:** (1) the first evaluation scored only one executed step's
  immediate outcome, not the multi-step rollout the search was actually
  optimizing for — showing beam search "beating" exhaustive on an
  objective a single-step evaluation doesn't measure at all; fixed by
  scoring a full 5-step receding-horizon rollout with replanning every
  step. (2) A first attempt used a finer, 21-action vocabulary; checked
  directly, one state at depth 3 had 19 different 3-action sequences
  tied at score 0.0, and exhaustive search's tie-breaking always favored
  the most extreme first action among ties — fixed by reusing experiment
  12's original, widely-spaced 6-action vocabulary (checked to have far
  fewer ties: 1 and 2 at depths 2 and 3, respectively).
- **The actual finding, once both bugs were fixed: beam search isn't
  just cheaper, it's measurably more robust to a genuine receding-
  horizon-control pathology.** At depth 3, exhaustive search converges
  to (and stays at) the target in only 2 of 15 starting states — while
  every beam width tested converges in 13-15 of 15, using 12-22x fewer
  `transition_fn` calls. Traced one misbehaving state step by step:
  exhaustive search's chosen policy enters a persistent oscillation
  (position stuck away from target, `pending` alternating sign forever),
  while beam search from the identical starting state converges and
  stays there. **Mechanism**: exhaustive search scores only the *final*
  simulated state after the full lookahead, indifferent to path — it can
  select a sequence whose promised final position looks optimal while
  its first action (the only one actually executed before replanning
  from scratch) sets up a self-reinforcing overshoot-correct cycle. Beam
  search scores every intermediate partial state during its own
  expansion, incidentally biasing toward monotonic progress that happens
  to avoid exactly this pathology.
- **Falsification:** would have been beam search failing to reproduce
  exhaustive search's choice at full beam width (a bug in the
  implementation, ruled out directly in `tests/test_planning.py` before
  this experiment), or the oscillation finding disappearing once both
  methodological bugs were fixed (meaning it was an artifact, not a real
  phenomenon) — neither happened. Full numbers, the traced pathological
  case, and what isn't tested (whether a smarter tie-breaking rule would
  close the gap; deeper depths without ground truth; whether the
  oscillation pathology is specific to this dynamics structure) in
  [experiments/exp14_beam_search_planning/RESULTS.md](../experiments/exp14_beam_search_planning/RESULTS.md).

## 7l. Experiment 15 — Nonlinear world-model dynamics (Phase 6, continued)

**Status: run.** Every world model built in this program
(`LinearDynamicsModel`, experiments 11-14) assumes the true transition is
linear in state. This tests whether that assumption's cost — already
established for causal effect estimation (experiment 10) and single-step
value estimation (experiment 11's model-free baseline) — recurs a third
time, for world-model dynamics prediction itself, and whether a
correctly-specified nonlinear dynamics model recovers it.

- **Hypothesis:** a nonlinear dynamics model (fitting
  `next_state ~ intercept(a) + b1(a)*state + b2(a)*state*|state|` per
  action, matching the true functional form) recovers near-oracle
  performance under genuinely nonlinear dynamics, while
  `LinearDynamicsModel`, reused completely unchanged, shows a real,
  measurable regret gap.
- **The confound this needed to control for:** showing a nonlinear
  world model beat a linear one would only prove nonlinear features can
  help *somewhere*, not that they capture the actual functional form —
  `nonlinear_world_model` uses `state*abs(state)` (an odd, sign-aware
  function), matching `NonlinearControlEnv`'s actual restoring-force
  construction exactly, not a generic `state**2` that would still be
  blind to the force's sign.
- **A boundary condition this experiment's first run found, checked
  rather than reported as a null result:** the initial `GAMMA=0.05`
  (chosen so the nonlinear term's magnitude at the state range's edge is
  comparable to a single nudge) showed almost no gap between the linear
  and nonlinear models — the same pattern experiment 13 found for a mild
  regime shift: a nonlinearity too small to change which discrete action
  ranks best doesn't produce a measurable gap. Swept `GAMMA` up rather
  than stopping there; `GAMMA=0.30` produces a clean, dramatic gap and
  is the value used for the main result.
- **Design:** `NonlinearControlEnv`
  (`environments/transworld/nonlinear_control_env.py`) adds
  `-GAMMA*state*abs(state)` to `ResourceControlEnv`'s (experiment 11)
  dynamics — the same structural role experiment 10's `gamma2*X**2` term
  played for a causal effect, now for a state's own evolution.
  **Verified by hand before being trusted for anything**: the
  environment's dynamics were checked against a hand computation
  (`state=4.0`, `action=large_up` → `next_state=5.2` exactly), and
  `NonlinearWorldModelAgent`'s fitted coefficients were checked against a
  noiseless synthetic case, recovered to floating-point precision, in
  `tests/test_exp15_nonlinear_world_model.py`.
- **Result (GAMMA=0.30): `nonlinear_world_model` matched the oracle
  ceiling almost exactly (regret 0.0036 vs. the oracle's 0.0014), while
  `linear_world_model`'s regret (0.6933) was roughly 193x larger** —
  despite identical state access and the identical `ordinary_least_squares`
  tool, differing only in feature set. `linear_world_model` still
  meaningfully beat the `state_blind` floor (0.6933 vs. 0.8769) — wrong
  but not worthless, the same pattern experiment 11's model-free linear
  baseline showed.
- **Falsification:** would have been `nonlinear_world_model` failing to
  reach the oracle ceiling despite ample data (a bug in the fitting
  mechanism), or the linear/nonlinear gap remaining negligible even at
  `GAMMA=0.30` (meaning the claimed misspecification doesn't actually
  matter for decision quality) — neither happened. Full numbers and what
  isn't tested (a single nonlinear functional form; the correct feature
  handed to the agent, not discovered; single-step decisions only, not
  composed with the multi-step planner or beam search; the exact
  boundary-severity crossover point unmapped) in
  [experiments/exp15_nonlinear_world_model/RESULTS.md](../experiments/exp15_nonlinear_world_model/RESULTS.md).

## 7m. Experiment 16 — Combining regime-change detection with multi-step planning (Phase 6, continued)

**Status: run.** A second synthesis experiment (after experiment 13):
does `CUSUMTemporalReasoner`-triggered adaptation still work when the
planner is a multi-step `RecedingHorizonPlanner` (experiments 12/14)
instead of experiment 13's single-step greedy lookahead? Nothing new is
implemented — every mechanism is reused exactly as already verified.

- **Hypothesis:** detection-triggered adaptation composes with
  multi-step planning the way it composed with single-step planning in
  experiment 13.
- **The confound this needed to control for:** a 2×2 design (planning
  horizon × adaptation) isolates whether multi-step planning and regime
  detection interfere with, are independent of, or amplify each other,
  rather than testing only "everything on."
- **Finding 1, an unprompted replication:** the first run's `oracle`
  condition (given the true dynamics exactly) scored -25.42 post-shift —
  worse than the *learned*, adaptive `greedy_cusum_adapts`'s -4.52, which
  should be impossible for a true oracle. Investigated rather than
  accepted: the cause was exactly experiment 14's oscillation pathology,
  independently reproduced in a new environment built for a different
  purpose — exhaustive search's terminal-only scoring, combined with this
  environment's delayed/pending dynamics, selects plans whose first
  action sets up a self-reinforcing oscillation once replanned. Beam
  search (`beam_width=2`, the value experiment 14 found most reliable)
  fixed it directly: the oracle's post-shift reward became -0.0284,
  matching its pre-shift performance almost exactly.
- **Finding 2, the actual answer to the original question:** fixing the
  search-strategy pathology did NOT make `mpc_beam_cusum_adapts` (-33.06)
  competitive with `greedy_cusum_adapts` (-4.52), despite identical
  adaptation and dynamics-fitting mechanisms. Traced across the whole
  post-shift window in 8 chunks: the gap is persistent, not a shrinking
  startup transient (chunk 7, with hundreds of post-reset samples, is no
  better than chunk 1). A "reduced exploration" hypothesis (multi-step
  planning converges to a narrow region, starving the model of diverse
  training data) was checked directly and **refuted** — `mpc_beam`'s
  visited positions post-shift had a *larger* spread than greedy's
  (stdev 4.25 vs. 1.87). Best-supported remaining explanation, stated as
  a hypothesis not a certainty: multi-step lookahead chains two
  predictions from the same learned model, and the estimation error each
  carries compounds in a way single-step lookahead never pays for — a
  cost that need not shrink with more data, unlike a small-sample
  transient.
- **Falsification:** would have been the oscillation pathology failing
  to replicate in this new environment (meaning experiment 14's finding
  was environment-specific, not general), or `mpc_beam_cusum_adapts`
  matching `greedy_cusum_adapts` once beam search was applied (meaning
  the two mechanisms simply compose, no new finding) — neither happened.
  Full numbers, the chunked trace, and what isn't tested (the
  compounding-estimation-error mechanism wasn't directly manipulated to
  confirm it; only one lookahead depth and beam width; only one
  regime-shift severity; not combined with experiment 15's nonlinear
  dynamics) in
  [experiments/exp16_regime_shift_multistep_planning/RESULTS.md](../experiments/exp16_regime_shift_multistep_planning/RESULTS.md).

## 7n. Experiment 17 — Validating `MonteCarloSimulator` against experiment 16's already-established finding (Phase 6, closing)

**Status: run.** The validating experiment for `transintelligence/simulation/`'s
new `MonteCarloSimulator` (§13/§19 `Simulator`, the last of Phase 6's
three remaining gaps after experiment 14 — nonlinear dynamics and
regime-adaptation + multi-step planning were the first two, closed by
experiments 15 and 16). Rather than a fresh, unrelated claim, this is a
genuine pre-registered check against a result already established by a
completely different methodology.

- **Hypothesis:** `MonteCarloSimulator.compare_policies`, given only
  experiment 16's two already-trained, frozen policies
  (`greedy_cusum_adapts`, `mpc_beam_cusum_adapts` — learning switched
  off) and the environment's true post-shift dynamics formula plus its
  real observation noise, recovers via short stochastic rollouts alone
  the same ranking experiment 16 found via full multi-seed environment
  rollouts averaged over 400 episodes each: `greedy_cusum_adapts` beats
  `mpc_beam_cusum_adapts` post-shift.
- **Why this is a real risk, not a formality:** the answer was already
  known, so a mismatch would have been a real, reportable finding —
  either a bug in `MonteCarloSimulator`, or evidence that experiment 16's
  full-environment result doesn't generalize to arbitrary starting
  states, not just its own `reset()` distribution.
- **Verified against a hand-computed case before trusting it here:** a
  deterministic (noise-free) canonical case first
  (`tests/test_simulation.py`) — always-up beats always-down by exactly
  the hand-computed margin, with zero spread, and swapping which policy
  is passed first flips the reported winner — before running it on
  anything stochastic or previously-unseen.
- **Result:** greedy wins 58/60 Monte Carlo comparisons (96.7%) across
  12 seeds × 5 starting states, with mean simulated rewards (-2.29 vs.
  -32.98) landing close to experiment 16's own full-environment numbers
  (-4.52 vs. -33.06) despite the completely different methodology.
- **Sensitivity control built into the same run:** dropping `n_rollouts`
  from 200 to 5 (same trained agents, same start states) reduced the win
  rate to 88.3% — noisier, as expected, but still correctly favoring
  greedy in the large majority of cases, confirming `n_rollouts` does
  real work rather than being cosmetic.
- **Falsification:** would have been a win rate near 50% (no recoverable
  signal) or favoring `mpc_beam_cusum_adapts` (contradicting experiment
  16 outright) — neither happened. What isn't tested (only one policy
  pair, one environment, one severity; `n_rollouts` sensitivity checked
  at only two values; pre-shift, where both agents perform
  near-identically, not separately validated) in
  [experiments/exp17_monte_carlo_simulator/RESULTS.md](../experiments/exp17_monte_carlo_simulator/RESULTS.md).

## 7o. Experiment 18 — Testing experiment 16's compounding-estimation-error hypothesis directly with `CalibrationVerifier` (Phase 6, closing / meta-intelligence, opening)

**Status: run.** Fills `Verifier` (`transintelligence/reasoning/interfaces.py`),
the last reasoning-protocol stub from before Phase 4 left unbuilt, with
`CalibrationVerifier` (`transintelligence/verification/model.py`) —
unlike experiment 17, this is a genuine new hypothesis test, not a
validation against an already-known answer.

- **Hypothesis:** experiment 16's central, explicitly *unconfirmed*
  explanation — multi-step lookahead chains two predictions from the
  same learned dynamics model, and each carries estimation error that
  compounds in a way single-step lookahead never pays for — implies
  `mpc_beam_cusum_adapts`'s dynamics model should show more excess
  estimation error, beyond the environment's true noise floor
  (`NOISE_SIGMA`), than `greedy_cusum_adapts`'s does, post-shift. Both
  agents fit the identical one-step OLS model; the only difference is
  how many steps ahead each searches when choosing an action — but each
  agent's own planning strategy steers it into different regions of
  state space, feeding back into what data its own model trains on.
- **The confound this needed to control for:** pre-shift residuals
  (stable data, no regime confusion) checked as a baseline in the same
  run — if pre-shift were also miscalibrated for both agents, that would
  point to a cause unrelated to the regime shift.
- **Result:** both agents' one-step models are flagged `miscalibrated`
  in every window (observed coverage ~66% vs. 68.27% claimed) — but the
  gap between `greedy` and `mpc_beam` is under 0.002 post-shift, smaller
  than either agent's own pre-to-post-shift shift (~0.007), and the
  small miscalibration is present pre-shift too, identically for both
  agents. A robustness check restricting to only the first 500
  post-shift steps per seed shows the same pattern.
- **What this establishes:** the "self-steered training distribution
  differentially degrades the model" alternative explanation is **not
  supported** — `mpc_beam`'s wider post-shift position spread
  (established in experiment 16) does not measurably degrade its own
  one-step model's calibration relative to `greedy`'s. The universal
  small miscalibration is a baseline property of the fitting procedure
  (plausibly in-sample-adjacent OLS residuals understating true
  out-of-sample variance), not shift- or strategy-specific. By
  elimination, this leaves experiment 16's original
  compounding-across-chained-predictions hypothesis as the more
  plausible remaining explanation for the reward gap, though the
  chaining mechanism itself still was not directly manipulated — the
  hypothesis remains unconfirmed, now with one fewer competing
  explanation.
- **Falsification:** would have been `mpc_beam` showing measurably worse
  one-step calibration than `greedy` post-shift specifically (supporting
  the self-steered-degradation alternative instead), or pre-shift being
  well-calibrated while post-shift wasn't (implicating the shift itself
  rather than a baseline fitting property) — neither happened. Full
  numbers and what isn't tested (the chaining mechanism itself; only one
  environment, severity, and lookahead depth; only one calibration
  threshold) in
  [experiments/exp18_calibration_verifier/RESULTS.md](../experiments/exp18_calibration_verifier/RESULTS.md).

## 7p. Experiment 19 — Extending regime-adaptation to a nonlinear dynamics model (Phase 6, continued)

**Status: run.** A third synthesis experiment (after 13 and 16): does
`CUSUMTemporalReasoner`-triggered adaptation (experiment 13) still work
when the learned model is `NonlinearDynamicsModel` (generalized from
experiment 15 into `transintelligence/world_models/`) instead of
`LinearDynamicsModel`? Structurally identical to experiment 13
(`environments/transworld/nonlinear_regime_shift_env.py` combines
experiment 15's restoring force with experiment 13's silent actuator
rescaling), one substitution only.

- **Hypothesis:** the composition holds the same way it did for the
  linear model in experiment 13 — no benefit under a mild shift, a clear
  benefit under a severe one.
- **Why it wasn't obvious:** `CUSUMTemporalReasoner` monitors residuals
  agnostic to whether the underlying model is linear, but
  `NonlinearDynamicsModel` needs 3 observations per action to fit at
  all, one more than `LinearDynamicsModel`'s 2 — no a priori guarantee a
  pipeline implicitly calibrated around a linear model transfers
  unchanged.
- **Finding 1:** under a mild shift, `oracle_adapts` performs *worse*
  than `never_adapts` (-0.5278 vs. -0.3630) — not merely
  indistinguishable, as experiment 13 found for its linear model
  (-5.0996 vs. -5.0823, ~0.3% gap). Confirmed directly, not inferred:
  instrumentation shows a post-shift agent's `known_actions()` stays
  completely empty for the entire first 19-trial refit window (forced
  pure random action selection) — the nonlinear model's higher
  per-action data requirement gives a hard-reset strategy a real,
  measurable cold-start cost a linear model's shorter cold start doesn't
  produce.
- **Finding 2:** under a severe shift, `oracle_adapts` still clearly
  beats `never_adapts` (-0.4151 vs. -1.6340), but `cusum_detects_and_adapts`
  (-1.3364) fails the confound-controlled test experiment 13's design
  passed — it does not beat `sliding_window_baseline` (-0.4255).
  Detection itself is measurably less reliable (9/15 seeds detected vs.
  experiment 13's 15/15; mean latency 125.2 vs. 23.0 trials) — the
  natural "noisier residuals" explanation was checked directly and
  **refuted** (pre-shift residual stdev ~0.30 in both linear and
  nonlinear settings, 5 matched seeds), leaving the true cause
  unidentified. `sliding_window_baseline`, which never fully empties its
  training data, avoids the cold-start cliff entirely and becomes the
  most practically competitive strategy in both severities — reversing
  experiment 13's own preference ordering.
- **Falsification:** would have been the composition holding unchanged
  (matching experiment 13's pattern with no new effect) — it didn't.
  Full numbers and what isn't tested (the cause of degraded detection
  reliability; only one nonlinearity and GAMMA value; the cold-start cost
  not mitigated or engineered around; not combined with multi-step
  planning) in
  [experiments/exp19_nonlinear_regime_shift/RESULTS.md](../experiments/exp19_nonlinear_regime_shift/RESULTS.md).

## 7q. Experiment 20 — Directly confirming experiment 16's compounding-estimation-error hypothesis via a noise sweep (Phase 6, continued)

**Status: run.** Nothing new implemented — `Agent`,
`DelayedRegimeShiftControlEnv`, and `run_episode` reused exactly as
experiments 16-18 already verified, with `noise_sigma` swept instead of
fixed. The most literal possible test of experiment 16's own stated
mechanism: multi-step lookahead's persistent post-shift gap comes from
chained predictions' estimation error, which never fully vanishes
*because the environment has real observation noise* — does the gap
shrink toward zero as that noise shrinks toward zero?

- **Hypothesis:** the `greedy_cusum_adapts` vs. `mpc_beam_cusum_adapts`
  post-shift reward gap should shrink substantially as `NOISE_SIGMA` ->
  0 — a deterministic environment gives a correctly-specified model
  nothing left to compound.
- **The confound this needed to control for:** detection reliability
  could plausibly vary across noise levels (experiment 19 already found
  detection sensitivity to a model's residual characteristics) — if the
  two agents detected the shift at very different rates at some noise
  level, that alone could produce a reward gap unrelated to
  estimation-error compounding. Detection counts/latencies for both
  agents, at every noise level, reported in the same run.
- **Result:** the gap scales strongly with `NOISE_SIGMA` — 1.44 at
  `NOISE_SIGMA=0.0` vs. 24.72-30.92 at `NOISE_SIGMA in [0.1, 0.4]`, a
  ~17-21x difference between the noise-free and noisy ends of the
  sweep. Detection reliability tracks closely between the two agents at
  every noise level, ruling out differential detection as an
  alternative explanation. A 10-seed first pass showed an apparent
  non-monotonic dip at `NOISE_SIGMA=0.2` (gap 6.91) that a targeted
  20-seed re-check did not replicate (19.75) — investigated before being
  trusted, attributed to sample noise, not left unexamined.
- **What this establishes:** the first direct, positive, manipulation-
  based confirmation of experiment 16's central hypothesis — experiment
  18 only ruled out one alternative explanation without testing the
  mechanism's own stated cause; this test does, and it holds.
- **Falsification:** would have been the gap staying roughly constant
  across the sweep (a noise-independent, structural cause instead) — it
  didn't. Full numbers and what isn't tested (the noise -> gap
  relationship's exact functional form; `NOISE_SIGMA=0.0`'s small
  residual gap not further investigated; only one environment, severity,
  lookahead depth, and beam width) in
  [experiments/exp20_noise_sweep_compounding_error/RESULTS.md](../experiments/exp20_noise_sweep_compounding_error/RESULTS.md).

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
3. ~~Experiment 1~~ — **done**, see §5. Result: `rf_aware` beats flat,
   a fair (thrice-strengthened) learned-embedding baseline, and two oracle
   controls — but the advantage vanishes under high observation noise
   (σ≳0.2) and collapses when the active frame isn't in `rf_aware`'s known
   candidate list (−0.181 accuracy). Positive-with-real-caveats, not
   simply positive. Formal calibration (Brier=0.191) and the held-out-frame
   test both run. Full results and five rounds of fixes/strengthening in
   [experiments/exp01_frame_conditioning/](../experiments/exp01_frame_conditioning/RESULTS.md).
4. ~~Experiment 3~~ — **done**, see §7, with the isomorphism control plus
   an additional confound control (`random_differentiated`) the original
   design didn't call for but turned out to be necessary. Result: weak,
   partial support for cross-domain transfer — a real but small
   structure-specific effect, substantially smaller than a naive
   transfer-vs-scratch comparison suggests once the "any non-random start
   helps" confound is subtracted out. Full decomposition in
   [experiments/exp03_cross_domain_transfer/](../experiments/exp03_cross_domain_transfer/RESULTS.md).
5. ~~Experiment 4~~ — **done**, see §7a. Result: the cleanest positive
   result of the four experiments — fitted discovery recovers held-out
   accuracy to 0.914 (from 0.701 pre-discovery), beating the
   `RandomDiscoveryAgent` confound control (0.786) by +0.128, with the
   fitted frames confirmed close to the true held-out regime. Built the
   control in the same commit as the treatment, per the lesson from
   experiment 3. Full results in
   [experiments/exp04_frame_discovery/](../experiments/exp04_frame_discovery/RESULTS.md).
6. ~~Experiment 5~~ — **done, five follow-ups deep**, see §7b. First
   Phase 4 work, and the first genuinely reusable kernel primitive from
   this research program (`transintelligence/reasoning/temporal/`,
   previously an empty stub). CUSUM change detection degrades gracefully
   under noise — direct, independent evidence for the self-calibrating-
   threshold fix experiment 4's results named but never built — has a
   real, mechanistically clean structural limit (regimes must persist at
   least `burn_in` steps), segments its output about ~2-2.6x worse than a
   ground-truth oracle regardless of noise level, correctly implements DTW
   for trajectory comparison, gets a genuine bounded win from multi-key
   evidence combination, is meaningfully more fragile to non-Gaussian
   noise (specificity, not power), and turned out to handle gradual drift
   fine despite a specific, reasoned hypothesis that it wouldn't. Full
   results in
   [experiments/exp05_regime_change_detection/](../experiments/exp05_regime_change_detection/RESULTS.md).
7. ~~Experiment 6~~ — **done**, see §7c. First Phase 5 work, and the
   second genuinely reusable kernel primitive from this research program
   (`transintelligence/reasoning/causal/`, previously an empty stub).
   The backdoor criterion, computed from graph structure alone with no
   data, exactly predicts which adjustment sets are unbiased — naive
   regression finds a spurious effect of 0.881 where the truth is 0.0,
   the valid adjustment recovers ~0, and adjusting for a collider instead
   produces its own distinct bias that gets *worse*, not neutral, when
   layered onto an otherwise-correct model. Full results in
   [experiments/exp06_confounding_bias/](../experiments/exp06_confounding_bias/RESULTS.md).
8. ~~Experiment 7~~ — **done**, see §7d. Fills `CounterfactualReasoner`,
   the last previously-empty reasoning stub before Phase 4 started. With
   true coefficients, per-unit counterfactual recovery is exact (error
   0.0000) while the naive population-plug-in shortcut's error (0.2413)
   matches the theoretical mean absolute exogenous noise almost exactly —
   confirming it's wrong for the precise reason expected: it silently
   discards each unit's own residual. With estimated coefficients,
   abduction still wins ~9x, and naive's error barely moves between the
   two conditions, because its error source is structurally independent
   of estimation quality. Full results in
   [experiments/exp07_counterfactual_queries/](../experiments/exp07_counterfactual_queries/RESULTS.md).
9. ~~Experiment 8~~ — **done**, see §7e. Constraint-based causal discovery
   (PC skeleton + collider orientation) recovers the correct skeleton as
   sample size grows (recall 0.750→1.000 across n=100→1000), correctly
   orients a dedicated unshielded collider exactly at every sample size
   tested, correctly declines to orient experiment 6's shielded collider
   at every sample size (0/80 false orientations), and a negative control
   confirms the false-edge rate shrinks rather than grows with sample
   size. A follow-up added three of Meek's four orientation-propagation
   rules (R1-R3, R4 provably inapplicable without background knowledge),
   extending recall from a hard 0.500 ceiling to 1.000 on a
   fully-identifiable graph with zero wrong orientations given a correct
   skeleton, and zero spurious orientations on a genuinely undetermined
   negative-control chain. Full results in
   [experiments/exp08_causal_discovery/](../experiments/exp08_causal_discovery/RESULTS.md).
10. ~~Experiment 9~~ — **done**, see §7f. Two-stage least squares and
    front-door adjustment both recover the true effect despite a
    genuinely unobserved confounder (2SLS: bias 0.0131 vs. naive's
    0.4118 at instrument strength 0.9; front-door: bias 0.0094 vs.
    naive's 0.8807), and both fail in their theory-predicted ways when
    stress-tested: 2SLS becomes wildly unstable below a weak-instrument
    threshold (stdev grows ~390x), and front-door adjustment becomes
    nearly as biased as naive when its "confounder doesn't reach the
    mediator" assumption is violated (~99x bias increase). Full results
    in
    [experiments/exp09_iv_and_frontdoor/](../experiments/exp09_iv_and_frontdoor/RESULTS.md).
11. ~~Experiment 10~~ — **done**, see §7g. The last of the three stated
    gaps. `StructuralEquation` generalizes to nonlinear structural
    functions with zero changes to `abduct()`/`counterfactual()`
    themselves — verified exact on a hand-computed quadratic case.
    Linear-adjusted OLS effect estimation, by contrast, is substantially
    biased under true nonlinearity (gap 0.5799 vs. an exact nonlinear
    reference, vs. 0.0121 in a linear control) and, more sharply, a
    single linear coefficient cannot represent a heterogeneous effect at
    all — the true per-unit shift effect ranged from -1.5 to +3.3 and
    even flipped sign across five reference points, while the linear
    model predicted the same constant number at every one of them. Full
    results in
    [experiments/exp10_nonlinear_scm/](../experiments/exp10_nonlinear_scm/RESULTS.md).
12. ~~Experiment 11~~ — **done**, see §7h. The first Phase 6 (World
    Models) work. An agent that learns forward dynamics and plans by
    simulating candidate actions matches the true oracle's performance
    ceiling almost exactly (regret -0.0002), while an identically-tooled,
    equally state-aware model-free baseline that fits reward directly
    falls far short (regret 3.5421) — because the true reward is a
    non-monotonic function of state a linear value fit cannot represent,
    while the true dynamics are linear and therefore exactly learnable.
    Full results in
    [experiments/exp11_world_model_planning/](../experiments/exp11_world_model_planning/RESULTS.md).
    A follow-up confirmed the mechanism directly: a quadratic (correctly-
    specified) model-free baseline closed almost the entire gap (regret
    3.5421 → 0.0144), matching `world_model`'s -0.0002.
13. ~~Experiment 12~~ — **done**, see §7i. A 2×2 factorial design
    (`greedy`/`mpc` × `learned`/`oracle`) isolates planning-horizon from
    model-quality on `DelayedControlEnv`, a genuine multi-step
    credit-assignment environment (unlike experiment 11's fresh-state-
    per-trial design). Multi-step (receding-horizon) planning beat greedy
    1-step lookahead by almost exactly the same margin in both the
    learned and oracle pairs, confirming the effect is about horizon, not
    model quality — a real, consistent (14/15 seeds), but modest (~17%
    relative) effect, reported at its actual size. Full results in
    [experiments/exp12_multistep_planning/](../experiments/exp12_multistep_planning/RESULTS.md).
14. ~~Experiment 13~~ — **done**, see §7j. A synthesis experiment
    combining `CUSUMTemporalReasoner` (Phase 4) with `LinearDynamicsModel`
    (Phase 6). The first, simpler version of this experiment found a
    null result (adaptation didn't help under a mild regime shift) that
    was investigated rather than reported as-is, revealing the actual
    mechanism (mild miscalibration rarely changes which action is
    locally best) and turning the experiment into a severity sweep. At
    a severe (sign-flipping) shift, detection-triggered adaptation
    beat both a do-nothing baseline (post-shift reward -2.62 vs. -15.49)
    and a naive always-use-recent-data heuristic (-2.62 vs. -3.00),
    landing close to the oracle ceiling (-2.36). Also surfaced a real,
    traced limitation: a 20% false-positive rate, notably higher than
    experiment 5's ~6%, caused by monitoring a periodically-refit
    model's residuals rather than a stationary raw signal. Full results
    in
    [experiments/exp13_regime_shift_world_model/](../experiments/exp13_regime_shift_world_model/RESULTS.md).
15. ~~Experiment 14~~ — **done**, see §7k. Set out to test the expected,
    small result (beam search approximates exhaustive search's quality
    at lower cost) and found something sharper by investigating a result
    that looked wrong rather than reporting it: at depth 3, exhaustive
    search's terminal-only scoring converges to target in only 2/15
    starting states under receding-horizon replanning, while beam search
    converges in 13-15/15 using 12-22x fewer `transition_fn` calls —
    beam search's per-step scoring is more robust to a genuine
    receding-horizon oscillation pathology, not just cheaper. Two real
    methodological bugs (a single-step evaluation; tie-breaking
    degeneracy in a fine action grid) were caught and fixed before this
    finding could be trusted. Full results in
    [experiments/exp14_beam_search_planning/](../experiments/exp14_beam_search_planning/RESULTS.md).
16. ~~Experiment 15~~ — **done**, see §7l. A weak nonlinear perturbation
    to `ResourceControlEnv`'s dynamics (`GAMMA=0.05`) showed no gap
    between linear and nonlinear world models — the same
    doesn't-change-the-optimal-action pattern experiment 13 found for a
    mild regime shift — but a stronger one (`GAMMA=0.30`) produced a
    clean, ~193x regret gap, with the correctly-specified nonlinear
    model matching the oracle ceiling almost exactly. Full results in
    [experiments/exp15_nonlinear_world_model/](../experiments/exp15_nonlinear_world_model/RESULTS.md).
17. ~~Experiment 16~~ — **done**, see §7m. A second synthesis experiment
    surfaced two findings only visible by testing the actual
    combination: an unprompted, independent replication of experiment
    14's exhaustive-search oscillation pathology in a new environment
    (fixed by beam search, exactly as before), and a genuinely new
    result that fixing it wasn't enough — learned multi-step planning
    persistently underperforms learned single-step planning once
    combined with regime-adaptation (-33.06 vs. -4.52 post-shift,
    unchanged across the whole post-shift window), with a "reduced
    exploration" explanation checked directly and refuted. Full results
    in
    [experiments/exp16_regime_shift_multistep_planning/](../experiments/exp16_regime_shift_multistep_planning/RESULTS.md).
18. ~~Experiment 17~~ — **done**, see §7n. Closed Phase 6's last
    remaining gap, `Simulator`, with `MonteCarloSimulator`
    (`transintelligence/simulation/`). Not a fresh claim — a validation
    against experiment 16's already-established finding, using a
    completely different methodology (short stochastic rollouts from
    arbitrary starting states, not full 400-episode environment
    averages): independently recovered the same ranking
    (`greedy_cusum_adapts` beats `mpc_beam_cusum_adapts` post-shift) in
    58/60 comparisons, with a built-in sensitivity control confirming
    the number of rollouts genuinely affects verdict reliability. Full
    results in
    [experiments/exp17_monte_carlo_simulator/](../experiments/exp17_monte_carlo_simulator/RESULTS.md).
19. ~~Experiment 18~~ — **done**, see §7o. Filled `Verifier`, the last
    reasoning-protocol stub from before Phase 4, with
    `CalibrationVerifier` (Kupiec 1995's unconditional-coverage test).
    Unlike experiment 17, a genuine new hypothesis test: does
    experiment 16's own best-supported but unconfirmed explanation
    (chained multi-step predictions compound a learned model's
    estimation error) hold up when checked directly against each
    agent's actual one-step residuals? It does not show `mpc_beam`'s
    model as measurably worse-calibrated than `greedy`'s post-shift —
    ruling out one plausible alternative explanation and, by
    elimination, leaving the original chaining hypothesis more
    plausible, still not directly confirmed. Full results in
    [experiments/exp18_calibration_verifier/](../experiments/exp18_calibration_verifier/RESULTS.md).
20. ~~Experiment 19~~ — **done**, see §7p. Extended regime-adaptation to
    a nonlinear dynamics model (`NonlinearDynamicsModel`, generalized
    from experiment 15's follow-up) — a third synthesis experiment,
    structurally identical to experiment 13. The composition does NOT
    transfer for free: a mild shift, an honest null result for the
    linear model, actively HURTS here (a cold-start cost traced
    directly via instrumentation); a severe shift still favors
    adaptation, but CUSUM detection reliability itself measurably
    degrades for a reason checked and refuted but not otherwise
    identified. Full results, plus two follow-ups (narrowing the
    detection-degradation cause, and a partial fix for the cold-start
    cost), in
    [experiments/exp19_nonlinear_regime_shift/](../experiments/exp19_nonlinear_regime_shift/RESULTS.md).
21. ~~Experiment 20~~ — **done**, see §7q. Directly confirmed experiment
    16's central compounding-estimation-error hypothesis — where
    experiment 18 only ruled out one alternative explanation, this
    swept the environment's own observation noise from zero to 4x its
    original value and found the persistent multi-step-planning gap
    scales strongly with it (1.44 at zero noise vs. 24.72-30.92 at or
    above the original level), with detection reliability tracked and
    shown not to differ between agents. The first direct, positive
    confirmation of the mechanism in this program's Phase 6 arc. Full
    results, plus a follow-up characterizing (partially) the noise→gap
    functional form, in
    [experiments/exp20_noise_sweep_compounding_error/](../experiments/exp20_noise_sweep_compounding_error/RESULTS.md).

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
  context's own §21 and §33 already say this. Now that experiment 3 has
  run: still don't lead with it — the result is weak/partial and
  substantially confound-corrected, the weakest of the four findings, not
  a capstone result.
- **Experiment 4 is the strongest result, and the safest to lead with if
  one must be chosen** — the only one of the four where the confound
  control confirmed the effect rather than mostly explaining it away, and
  the hyperparameters were derived analytically before running rather than
  tuned after seeing a result. Still frame it precisely: it recovers from
  a *narrow, hand-designed* failure mode (a two-parameter frame family)
  and **only within the same bounded noise regime experiment 1's
  structural claim holds** (the follow-up noise sweep found a specificity
  collapse, not a graceful decline, above σ≈0.1), **and only for regimes
  that are behaviorally distinguishable from current belief** (the
  two-missing-regimes follow-up found one of two equally-distinct-in-
  parameters regimes went undetected 8/10 seeds because it happened to
  agree with existing predictions ~90% of the time). Not a general
  solution to open-world frame discovery — the established theory it
  borrows from (§8a of `related-work.md`) solves a much harder version of
  this problem than what was actually tested here, and both follow-ups
  show exactly where that gap matters.
- **All twenty experiments are now done** (§5-7q). If this program is
  written up externally, the honest headline is: reference-frame
  conditioning helps within a bounded noise/coverage regime (experiment 1),
  a naive frame-dependence detector can fail in exactly the common case and
  the fix is provable not just empirical (experiment 2), cross-domain
  transfer shows a real but small effect once a necessary confound control
  is applied (experiment 3), a genuinely novel regime outside the known
  candidate set — the specific failure mode experiment 1 found — can be
  detected and largely recovered from by a deliberately crude heuristic
  (experiment 4), a properly self-calibrated detector (built for an
  unrelated Phase 4 kernel capability) degrades far more gracefully than
  experiment 4's fixed-threshold trigger did, retroactively validating the
  fix experiment 4 named but didn't build (experiment 5), a purely
  graph-theoretic criterion computed with no data at all exactly predicts
  which covariate adjustments remove confounding bias and which introduce
  a *different* bias instead (experiment 6), the same graph-theoretic
  machinery extends cleanly to exact per-unit counterfactual recovery,
  with a naive shortcut's error traced to a precise, predicted cause
  rather than just observed to be worse (experiment 7), graph structure
  itself doesn't have to be assumed — a constraint-based discovery
  procedure recovers it from data, with a negative control ruling out the
  obvious way that could have been an illusion of statistical power
  (experiment 8), and even an unobserved confounder that discovery could
  never find doesn't block identification, given a valid instrument or
  mediator — though both of those alternative strategies fail in their
  own theory-predicted ways once their assumptions don't hold (experiment
  9), and the linear-only simplification every one of these causal/
  counterfactual mechanisms made from the start turns out to matter in
  exactly the way it should: abduction stays exact under nonlinearity
  with no code changes needed, while linear effect estimation is
  substantially biased there, unable to represent an effect that
  genuinely depends on where a unit starts (experiment 10), and the
  first Phase 6 result shows that decomposing "learn the dynamics, apply
  the known reward formula" beats directly fitting a value function with
  the identical tool and identical state access — not because context
  helps (trivial), but because the value surface is non-monotonic in
  state in a way no linear fit can represent, echoing experiment 10's
  lesson in a planning setting instead of an effect-estimation one
  (experiment 11) — a follow-up confirmed the mechanism directly by
  showing a correctly-specified (quadratic) model-free baseline closes
  almost the entire gap — and multi-step planning beats greedy 1-step
  lookahead by a real, consistent, but honestly modest margin, with a
  2x2 design confirming the advantage is specifically about planning
  horizon rather than an accidental model-quality difference (experiment
  12), and combining two already-verified Phase 4/6 primitives
  (CUSUM change detection, learned dynamics) works as expected but only
  conditionally: a mild regime shift showed no adaptation benefit at
  all (a real, investigated null result, not a bug), while a severe one
  showed detection-triggered adaptation beating both a do-nothing
  baseline and a naive recency heuristic decisively (experiment 13), and
  a fourteenth set out to test a small, expected scalability fix (beam
  search vs. exhaustive search for the planner) and instead found, by
  investigating a result that looked wrong rather than reporting it,
  that beam search isn't just cheaper — it's measurably more robust to a
  genuine receding-horizon oscillation pathology exhaustive search's
  terminal-only scoring is vulnerable to (experiment 14), and a
  fifteenth found the linear-cannot-represent-a-nonlinearity lesson
  recurring for a third distinct object — world-model dynamics
  prediction itself, after causal effect estimation and single-step
  value estimation — with a directly-checked boundary condition
  mirroring experiment 13's: a weak nonlinearity that doesn't change
  which discrete action ranks best shows no gap, a strong one produces a
  clean ~193x regret gap (experiment 15), and a sixteenth, a second
  synthesis experiment, independently replicated experiment 14's
  exhaustive-search oscillation pathology in an unrelated environment
  and then, after fixing it, found a genuinely new result that could
  only be found by testing the actual combination: learned multi-step
  planning persistently underperforms learned single-step planning once
  combined with regime-adaptation, with the obvious "reduced
  exploration" explanation checked directly and refuted (experiment 16),
  and a seventeenth closed out Phase 6 with a validation rather than a
  new claim: `MonteCarloSimulator`, a newly-built kernel primitive using
  a completely different methodology (short stochastic rollouts from
  arbitrary starting states, not full 400-episode environment averages),
  independently recovered experiment 16's exact ranking in 58/60
  comparisons, with a built-in sensitivity control confirming the number
  of rollouts genuinely mattered rather than being cosmetic (experiment
  17), and an eighteenth took experiment 16's own best-supported but
  explicitly unconfirmed explanation — chained multi-step predictions
  compound a learned model's estimation error — and tested it directly
  with a newly-built calibration check: `mpc_beam`'s one-step dynamics
  model is NOT measurably worse-calibrated than `greedy`'s post-shift,
  ruling out one plausible alternative explanation (a self-steered,
  degraded training distribution) and, by elimination, leaving the
  original chaining hypothesis the more plausible remaining one, still
  not directly confirmed (experiment 18), and a nineteenth extended
  regime-adaptation to a nonlinear dynamics model (the last of Phase 6's
  named remaining-gaps items) and found the composition does NOT
  transfer for free: a mild shift, an honest null in experiment 13's
  linear version, actively HURTS here (oracle_adapts worse than
  never_adapts), traced directly to a cold-start cost the nonlinear
  model's higher per-action data requirement creates; a severe shift
  still favors adaptation overall, but CUSUM detection itself is
  measurably less reliable (9/15 vs. experiment 13's 15/15 seeds
  detected) for a reason checked and ruled out (steady-state residual
  noise) but not identified, and a simpler sliding-window heuristic ends
  up more robust than CUSUM detection here, reversing experiment 13's
  own preference ordering (experiment 19), and a twentieth went back and
  directly confirmed experiment 16's central hypothesis rather than only
  ruling out an alternative to it (experiment 18's contribution): sweeping
  the environment's own observation noise from 0 up to 4x its original
  value shows the persistent multi-step-planning gap scale strongly with
  it — 1.44 at zero noise vs. 24.72-30.92 at noise levels at or above the
  original, a ~17-21x difference — with detection reliability tracked
  and shown to stay closely matched between both agents throughout,
  ruling out differential detection as an alternative explanation for
  the pattern (experiment 20). That's a coherent, modest,
  defensible set of claims — resist the
  temptation to round any of them up, experiments 4 through 20 included.
- **Experiments 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, and
  20 are also the first results from this program that are reusable
  kernel capabilities, not RL research scripts** — worth leading with in any framing aimed at the "is any of
  this actually usable" question, separate from the reference-frame-
  conditioning experiments' own framing. Experiments 6 and 7 together
  remain the cleanest, most textbook-dramatic results: a spurious effect
  of 0.881 where the truth is 0.0, corrected to ~0 by the graph-predicted
  adjustment (6), and an exact per-unit counterfactual recovery whose
  only error source (finite-sample estimation) is precisely characterized
  (7) — worth leading with if only one or two results can be shown.
  Experiment 8 is the one that shows the graph itself need not be a given
  input at all, and experiment 9 is the one that shows identification
  survives even the case discovery can't solve — a confounder that never
  appears in the data — while still being honest that both of its
  strategies have real, demonstrated breaking points of their own.
  Experiment 10 closes out Phase 5's stated gaps by showing precisely
  where the linearity simplification the whole phase made from the start
  does and doesn't matter: not at all for counterfactual abduction, a
  great deal for effect estimation. Experiment 11 opens Phase 6 by
  showing that same lesson recurs in a genuinely different setting
  (planning, not estimation) — not a coincidence so much as the same
  underlying mathematical fact (a linear function cannot represent a
  relationship with a peak) showing up wherever it's structurally
  relevant. Experiment 12 is the first Phase 6 result that isn't a
  variation on that theme — it tests a structurally different question
  (planning horizon, not function-class mismatch) with its own dedicated
  2x2 confound-isolation design, and reports a genuinely modest effect
  size honestly rather than reaching for a more dramatic-sounding number.
  Experiment 13 is the first result in this program built entirely from
  two already-verified pieces rather than any new mechanism — and its
  most valuable moment wasn't the positive result at severe shift
  severity, it was catching and explaining the *null* result at mild
  severity instead of quietly re-running the experiment until a cleaner
  number appeared. Experiment 14 is the sharpest instance yet of that
  same discipline: it caught two of its own methodological bugs (a
  single-step evaluation, tie-breaking degeneracy in a fine action grid)
  before trusting a result that initially looked impossible (beam search
  "beating" exhaustive search), and what survived after fixing both was
  more interesting than the originally-planned "cheaper at a small
  quality cost" story — a genuine, quantified robustness advantage the
  experiment wasn't designed to find. Experiment 15 closes out the
  three-object arc this program's linearity theme has now traced end to
  end: causal effect estimation (experiment 10), single-step value
  estimation (experiment 11), and now world-model dynamics prediction
  itself, each independently confirming a linear fit cannot represent a
  relationship it wasn't built to represent, and each independently
  needing the misspecification to be severe enough to change an actual
  decision before the effect became visible at all. Experiment 16 is the
  clearest demonstration yet of why this program tests syntheses, not
  just individual mechanisms: experiments 12 and 14 independently
  established multi-step planning's value, and experiment 13
  independently established regime-adaptation's value, but combining
  them revealed a real, persistent interaction cost neither predicted —
  the kind of finding that only exists at the seam between two
  separately-validated capabilities, not inside either one.
