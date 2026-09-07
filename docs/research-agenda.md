# Research Agenda

## 1. Purpose

This document formalizes the first falsifiable claims TransIntelligence is
willing to stand behind, and specifies the smallest experiments that could
support or kill each one. It intentionally covers less ground than the
project vision (see `docs/architecture.md`, `docs/intelligence-model.md`):
vision motivates the program, this document constrains near-term work to
what can actually be measured.

All three experiments below have now run. For a standalone summary of what
they actually established — without reading this document's incremental
updates or three separate `RESULTS.md` files — see
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
  explicable results. Full numbers, the calibration sweep, and what
  isn't yet tested (only `change_points()` benchmarked directly, not
  `regime_segments()`'s per-segment accuracy; no non-Gaussian/gradual-drift
  transitions, which CUSUM isn't designed to detect cleanly) in
  [experiments/exp05_regime_change_detection/RESULTS.md](../experiments/exp05_regime_change_detection/RESULTS.md).

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
6. ~~Experiment 5~~ — **done**, see §7b. First Phase 4 work, and the first
   genuinely reusable kernel primitive from this research program
   (`transintelligence/reasoning/temporal/`, previously an empty stub).
   CUSUM change detection degrades gracefully under noise — direct,
   independent evidence for the self-calibrating-threshold fix experiment
   4's results named but never built — and has a real, mechanistically
   clean structural limit (regimes must persist at least `burn_in`
   steps). Full results in
   [experiments/exp05_regime_change_detection/](../experiments/exp05_regime_change_detection/RESULTS.md).

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
- **All five experiments are now done** (§5-7b). If this program is
  written up externally, the honest headline is: reference-frame
  conditioning helps within a bounded noise/coverage regime (experiment 1),
  a naive frame-dependence detector can fail in exactly the common case and
  the fix is provable not just empirical (experiment 2), cross-domain
  transfer shows a real but small effect once a necessary confound control
  is applied (experiment 3), a genuinely novel regime outside the known
  candidate set — the specific failure mode experiment 1 found — can be
  detected and largely recovered from by a deliberately crude heuristic
  (experiment 4), and a properly self-calibrated detector (built for an
  unrelated Phase 4 kernel capability) degrades far more gracefully than
  experiment 4's fixed-threshold trigger did, retroactively validating the
  fix experiment 4 named but didn't build (experiment 5). That's a
  coherent, modest, defensible set of claims — resist the temptation to
  round any of them up, experiment 4 and 5 included.
- **Experiment 5 is also the first result from this program that is a
  reusable kernel capability, not an RL research script** — worth leading
  with in any framing aimed at the "is any of this actually usable"
  question, separate from the reference-frame-conditioning experiments'
  own framing.
