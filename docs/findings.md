# Findings

A standalone summary of what the first four experiments in
[research-agenda.md](research-agenda.md) actually established, for anyone
who wants the result without reading four `RESULTS.md` files and the
incremental updates to the agenda itself. Each section below is a compressed
version of a much more detailed writeup — follow the links for the numbers,
the code, and the caveats a one-paragraph summary can't carry.

## The one-sentence version

Reference-frame conditioning works, but only within a bounded regime; every
positive result that looked clean on first pass got smaller once the
control built to potentially kill it actually ran — except one, where the
control confirmed the effect instead, and that asymmetry is itself worth
noticing.

## Experiment 2 — Does `sensitivity()` detect frame-dependent conclusions?

**Falsified in the general case, then fixed.** The original
`sensitivity(x, R1, R2)` reduced algebraically to `abs(baseline2 -
baseline1)` whenever the two compared frames shared a `direction` — the
entity's own raw value canceled out of the subtraction entirely, so the
statistic carried *zero* per-entity signal in exactly the common case
(all three frames in `examples/finance_demo.py` share a direction). This
was provable, not just observed at one seed, and confirmed by a synthetic
noise sweep: AUC sat at chance regardless of noise level in that regime.

Fixed by comparing the *signs* of the two `evaluate()` results instead of
their difference (each individually depends on the raw value regardless of
direction). Re-running the same experiment confirmed the fix: both
regimes now behave identically, AUC 1.000 at zero noise degrading smoothly
to chance as noise grows.

→ [experiments/exp02_frame_dependence/RESULTS.md](../experiments/exp02_frame_dependence/RESULTS.md)

## Experiment 1 — Does explicit frame structure beat a flat or opaque baseline?

**Positive, with two real limits.** An agent (`rf_aware`) that maintains a
Bayesian belief over a small set of known `ReferenceFrame` objects and
predicts via `evaluate()` beats:

- a flat single-rule baseline with no frame structure (+0.218 accuracy,
  10/10 seeds),
- a fair, capacity-matched *opaque* multi-hypothesis baseline
  (`learned_embedding`, strengthened across three rounds of fixes to
  +0.837 accuracy before losing to `rf_aware` by +0.080, the narrowest and
  most meaningful of these margins),
- and two oracle controls given the true active frame directly, including
  one (`true_oracle`) that isolates "cost of inference" as a clean +0.043.

But sweeping the environment surfaced two boundary conditions invisible at
a single fixed configuration:

- **The advantage vanishes under high observation noise** — it shrinks
  monotonically and is statistically indistinguishable from zero by
  σ=0.4. The claim holds within a tested regime (σ≲0.2), not universally.
- **The advantage inverts into a real liability when the world produces a
  regime outside the agent's known candidate list** — `rf_aware` loses
  0.181 accuracy when the active frame isn't one it was told about, while
  the flat and opaque baselines (which never assumed a fixed candidate
  set) show almost no gap. The same explicitness that makes `rf_aware`
  strong when the world matches its assumptions makes it brittle exactly
  when it doesn't.

→ [experiments/exp01_frame_conditioning/RESULTS.md](../experiments/exp01_frame_conditioning/RESULTS.md)

## Experiment 3 — Does a learned strategy transfer across structurally analogous domains?

**Weak, partial support — smaller than it first looked.** Training an
agent's learned decision rules on a finance-shaped domain and
transplanting them into a structurally analogous knowledge-shaped domain
(same numeric frame structure, different surface labels, built to satisfy
Gentner's 1983 structure-mapping distinction) beat training from scratch
by a wide, clean-looking margin (+0.218 accuracy, 10/10 trials).

Adding a control the original design didn't call for — an agent
initialized at the same *magnitude* as the trained one but never trained
on anything — showed that **most of that margin was a confound**: any
non-degenerate starting point beats a near-zero cold start regardless of
whether the domains match, because a cold start has no way to break
symmetry early on. Subtracting that confound out left a real
structure-specific effect, larger for the true analogy than for a
non-isomorphic control (roughly double), but far smaller than the naive
number, and it mostly disappears by the end of a full run rather than
persisting.

→ [experiments/exp03_cross_domain_transfer/RESULTS.md](../experiments/exp03_cross_domain_transfer/RESULTS.md)

## Experiment 4 — Can an agent discover a frame it wasn't told about?

**Positive, and the cleanest result of the four — the control confirmed
the effect instead of shrinking it.** Directly motivated by experiment 1's
held-out-frame failure: an agent that monitors its own reward rate,
statistically detects when its known frames stop fitting, and fits a new
`(baseline, direction)` candidate from recent data recovers accuracy on
the held-out regime from 0.701 (before discovery) to 0.914 (after) —
close to, though still short of, the ceiling for an agent that knew the
frame from the start.

The confound control here (`RandomDiscoveryAgent`, same trigger, but
appends a random frame instead of a fitted one) is what makes this
trustworthy rather than a coincidence: it also improves on the
pre-discovery baseline (0.701 → 0.786, confirming "any extra candidate
helps a little," the same kind of confound experiment 3 found) but falls
far short of the fitted version (0.786 vs. 0.914) — so most of the
recovery is specifically attributable to fitting the *right* frame, not
just having more of them. Spot-checking the fitted frames directly
confirmed they land close to the true held-out one. The trigger's
hyperparameters were derived analytically (exact binomial tail
probabilities) before anything was run, and the observed false-positive
rate (zero, across ~750 window-checks) matched the ~4.6e-5/window
prediction.

Scope: this is a narrow, two-parameter fit, not a general solution to
open-world frame discovery — the established theory it borrows a crude
heuristic from (Bayesian online changepoint detection, Dirichlet process
mixtures, open-set recognition) solves a much harder version of this
problem than what was actually tested here. Two follow-ups, run in the
same session, sharpened exactly how much harder:

- **A noise sweep found the fixed trigger threshold doesn't degrade
  gracefully.** Within the noise level it was calibrated for (and below),
  the result holds and false-discovery rate stays at 0.000. Above it, the
  dominant failure isn't reduced detection (the original prediction) but a
  **specificity collapse** — false triggers rise sharply with noise, and
  by σ≈0.2 the mechanism has stopped helping at all because the candidate
  list fills with noise-driven junk before a useful frame can be fit. This
  tracks the same noise boundary experiment 1 found for the underlying
  structural claim.
- **A two-simultaneously-missing-regimes test found detection depends on
  *behavior*, not parameters.** One held-out frame was discovered in 9/10
  seeds; a second, chosen to be just as parametrically distinct from the
  known ones, was discovered in only 2/10 — because its true rule happened
  to agree with the known (wrong) frames' predictions ~90% of the time by
  structural coincidence (checked directly), versus 40% for the reliably
  discovered one. The mechanism detects regimes that are *behaviorally*
  distinguishable under the available reward, not regimes that merely have
  different numbers.

→ [experiments/exp04_frame_discovery/RESULTS.md](../experiments/exp04_frame_discovery/RESULTS.md)

## The meta-finding

Across the four experiments, the same discipline applied every time: build
the control that could kill the result, then run it, and don't stop at the
first configuration that looks clean. Every single result got a real
qualifier once that happened. Three times the headline *number* shrank —
noise sweeps and the held-out-frame test both punctured otherwise-clean
numbers in experiment 1, and a confound control cut experiment 3's result
roughly in half. Experiment 4's core confound control (fitted vs. random
discovery) went the other way and *confirmed* the effect — but its own
follow-up stress tests (a noise sweep, a two-simultaneously-missing-
regimes test) still found real, structural boundaries: a specificity
collapse outside the calibrated noise level, and a dependence on
behavioral rather than parametric distinctness. No experiment that was
actually pushed on came back unqualified. None of the four hypotheses were
fully falsified, but none survived untouched either — that's the intended
outcome of the experimental discipline in
[research-agenda.md](research-agenda.md) §21, not a failure of it. A
result that survives its own strongest test is worth more than one that
was never tested against one, and the one experiment where the *core*
control confirmed rather than shrank the effect is still the one worth
the most trust — its remaining caveats are about scope, not about whether
the central claim is real.

## What isn't tested yet

- A learned-embedding baseline that matches prior art's actual mechanism
  (an encoder-decoder trained on reward/dynamics prediction) rather than
  the hard-EM mixture of linear experts built here.
- Discovery beyond a single missing regime from a known two-parameter
  family — two or more simultaneously missing regimes, a continuously
  drifting regime, or a richer frame structure than `(baseline,
  direction)` would all break the current grid-search fit.
- Any of the master context's later phases (temporal, causal,
  counterfactual reasoning; world models; agency; meta-intelligence) — all
  still pre-formalization, per `research-agenda.md`'s own sequencing.
