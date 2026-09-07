# Findings

A standalone summary of what the first five experiments in
[research-agenda.md](research-agenda.md) actually established, for anyone
who wants the result without reading five `RESULTS.md` files and the
incremental updates to the agenda itself. Each section below is a compressed
version of a much more detailed writeup — follow the links for the numbers,
the code, and the caveats a one-paragraph summary can't carry.

## The one-sentence version

Reference-frame conditioning works, but only within a bounded regime; every
positive result that looked clean on first pass got smaller once the
control built to potentially kill it actually ran — except one, where the
control confirmed the effect instead, and a fifth experiment on an
unrelated kernel capability then independently validated the specific fix
that would have helped the one result that didn't hold up cleanly.

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

A fourth baseline, built specifically to close the "not prior art's actual
mechanism" gap in `learned_embedding`: a genuine recurrent context
encoder (`rnn_embedding`), trained end-to-end via truncated
backpropagation through time — architecturally the closest thing in this
codebase to what the cited multi-task RL papers actually do. **It scored
worse than the flat baseline** (0.630 vs. 0.700), and lengthening the
truncation window from 1 to 20 steps changed nothing — a textbook
vanishing-gradient failure in vanilla RNNs, the exact problem LSTM/GRU
gating was invented to fix decades ago. This is a genuine, well-diagnosed
negative result, and it strengthens the overall picture rather than
weakening it: "opaque learned embedding" turns out to be a family with
very different members, one of which (the discrete mixture) clears a real
bar and one of which (the vanilla RNN) fails outright for reasons
unrelated to the frame-conditioning question itself.

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

Across the five experiments, the same discipline applied every time: build
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
actually pushed on came back unqualified. None of the five hypotheses were
fully falsified, but none survived untouched either — that's the intended
outcome of the experimental discipline in
[research-agenda.md](research-agenda.md) §21, not a failure of it. A
result that survives its own strongest test is worth more than one that
was never tested against one, and the one experiment where the *core*
control confirmed rather than shrank the effect is still the one worth
the most trust — its remaining caveats are about scope, not about whether
the central claim is real.

Experiment 5 adds a second kind of validation to this pattern: not a
control on its own result, but independent confirmation of a fix proposed
for an *earlier* experiment's failure. Experiment 4's noise sweep found
that a fixed detection threshold collapses sharply outside the noise
level it was calibrated for, and named the fix (recalibrate from local
data) without building it. Experiment 5, built for an unrelated kernel
capability, happened to use exactly that kind of self-calibrating
threshold — and it degraded gracefully under the identical stress test.
That's not a coincidence to wave away; it's the kind of cross-experiment
consistency that makes the whole research log more credible than any one
result in isolation.

## Experiment 5 — Can the kernel detect when a regime changed, not just which one is active?

**Positive, and the first result from this program that's a reusable
kernel capability rather than an RL research script.** Experiments 1-4
all live under `experiments/` as standalone research code.
`transintelligence/reasoning/temporal/` was, before this, a single-line
docstring stub — no change detection, no regime segmentation, nothing
beyond the point-in-time `state_at`/`trajectory` lookups Phase 1 already
had. `CUSUMTemporalReasoner` fills it in with a self-calibrating CUSUM
change detector (Page, 1954) operating directly on `StateHistory`.

A real calibration finding came first, the same way it did in earlier
experiments: the "textbook" statistical-process-control starting point
produced a **47% false-positive rate on genuinely stationary data**
(measured directly), because a short, self-calibrated burn-in window
gives an unreliable noise estimate and the test runs at every step, not
once. Recalibrated the defaults (`burn_in=30`, `h_sigma=8.0`, ~6%
false-positive rate) and documented the finding in the class itself, not
just here.

With that fixed, two sweeps: **noise degrades the detector gracefully**
(recall/precision ~0.97-0.99 up to σ=0.10, only crossing into serious
degradation past ~6x that noise level) — a materially different pattern
from experiment 4's sharp specificity collapse, and direct, independent
evidence that experiment 4's own suggested fix (a threshold that adapts
to locally observed noise instead of one fixed at design time) actually
works when built. **Regime length has a sharp, mechanistically correct
threshold at the detector's `burn_in` parameter** — recall is exactly
zero for regimes a third as long as `burn_in`, and jumps to ~1.0 the
moment regimes reach or exceed it. Not a bug: a regime shorter than the
calibration window can never be calibrated on before the next change
happens, and the clean transition right at the parameter boundary
confirms the mechanism does exactly what it's designed to do.

→ [experiments/exp05_regime_change_detection/RESULTS.md](../experiments/exp05_regime_change_detection/RESULTS.md)

## What isn't tested yet

- A learned-embedding baseline that matches prior art's actual mechanism
  more closely than either the hard-EM mixture or the vanilla RNN built
  here — both were tried; an LSTM/GRU-gated version is the concrete next
  lever, not attempted (the vanilla RNN failed on vanishing gradients, a
  well-understood and unrelated problem).
- Discovery beyond a single missing regime from a known two-parameter
  family — two or more simultaneously missing regimes, a continuously
  drifting regime, or a richer frame structure than `(baseline,
  direction)` would all break the current grid-search fit.
- `regime_segments()`'s per-segment accuracy under noise (only
  `change_points()` was benchmarked directly), temporal comparison of
  whole trajectories rather than point-to-point state diffs (dynamic time
  warping is the established method, not implemented), and non-Gaussian
  or gradual-drift regime transitions, which CUSUM isn't designed to
  detect cleanly.
- The master context's remaining later phases (causal, counterfactual
  reasoning; world models; agency; meta-intelligence) — all still
  pre-formalization, per `research-agenda.md`'s own sequencing.
