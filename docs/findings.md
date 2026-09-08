# Findings

A standalone summary of what the first nine experiments in
[research-agenda.md](research-agenda.md) actually established, for anyone
who wants the result without reading nine `RESULTS.md` files and the
incremental updates to the agenda itself. Each section below is a compressed
version of a much more detailed writeup — follow the links for the numbers,
the code, and the caveats a one-paragraph summary can't carry.

## The one-sentence version

Reference-frame conditioning works, but only within a bounded regime; every
positive result that looked clean on first pass got smaller once the
control built to potentially kill it actually ran — except one, where the
control confirmed the effect instead, a fifth experiment on an unrelated
kernel capability then independently validated the specific fix that would
have helped the one result that didn't hold up cleanly, a sixth showed
that a purely graph-theoretic criterion, computed before touching any
data, exactly predicts which statistical adjustments are safe and which
quietly make things worse, a seventh extended the same machinery to
per-unit questions with an exact, not just approximately correct, answer,
an eighth showed the graph itself doesn't have to be assumed — it can
be recovered from data, with a negative control ruling out the obvious way
that could have been a statistical-power illusion — and a ninth showed
even an unobserved confounder doesn't block identification given a valid
instrument or mediator, while directly demonstrating both of those
alternative strategies' own theory-predicted breaking points rather than
just asserting them.

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

Across the nine experiments, the same discipline applied every time: build
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
behavioral rather than parametric distinctness. Experiments 6, 7, and 8's
confound controls (adjusting for a collider instead of the true
confounder; a naive population plug-in instead of per-unit abduction; a
negative control of mutually independent variables) all went the other
way — they *confirmed*, with textbook clarity, that the graph-theoretic
machinery is doing real work, and each surfaced a sharper nuance than "it
works": combining a valid and an invalid adjustment is worse than the
invalid one alone (6), a naive shortcut's error is structurally immune to
more data in a way the correct method's isn't (7), and a discovery
procedure's false-edge rate shrinks rather than grows as sample size
increases, ruling out the "just finds more structure with more
statistical power" failure mode directly (8). Experiment 9 did both at
once, in the same experiment: its two positive results (2SLS, front-door
adjustment, both against a genuinely unobserved confounder) held up, but
its two matching stress tests (a weak-instrument sweep, a violated
front-door assumption) each found a real, sharp breaking point rather
than graceful degradation — 2SLS becomes wildly unstable, not just more
biased, below a threshold instrument strength, and front-door adjustment
becomes nearly as biased as doing nothing once its specific structural
precondition fails. No experiment that was actually pushed on came back
unqualified. None of the nine hypotheses were fully falsified, but none
survived untouched either — that's the intended
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

Two follow-ups extended this rather than overturning it. **Per-segment
accuracy** (not just detection recall/precision): `regime_segments()`'s
mean estimates stay consistently ~2-2.6x worse than a ground-truth
oracle across every noise level tested, not diverging as noise grows —
imperfect detection costs a roughly constant penalty, not a
compounding one. **Temporal comparison via Dynamic Time Warping** (Sakoe
& Chiba, 1978): constructed the technique's own classic motivating case
directly — a time-shifted-but-identically-shaped trajectory and a
same-position-but-differently-shaped one. Naive same-index comparison
ranks these backwards starting at just a 2-step shift; DTW ranks them
correctly throughout an 8-step window where naive is wrong, recognizing
the shifted trajectory as a perfect match regardless of delay.

Three more follow-ups pushed on the module's remaining stated gaps.
**Multi-key tracking** (Crosier, 1988): combining two keys' evidence into
one statistic genuinely beats a union-of-independent-detectors control in
a moderate signal-strength band (paired wins 9:2, 6:1, 3:0) — a real gain
from combining evidence, not the "more chances helps" confound this
program has learned to check for — but the advantage vanishes at the
weakest signal tested, a real boundary stated plainly rather than
smoothed over. **Non-Gaussian noise**: a standard heavy-tailed
contamination more than quadruples the false-positive rate (0.08 → 0.34)
while barely touching detection power (1.00 → 0.967) — the same
specificity-over-power fragility pattern found in experiment 4's trigger,
now confirmed a second time in a completely different mechanism.
**Gradual drift**: the working hypothesis going in was that a slow enough
drift might evade detection entirely. It was wrong — recall stayed
perfect even for a drift that never completes within the observed
series, because the detector's calibration is fixed once per cycle rather
than continuously re-chased, so any persistent drift mathematically
crosses threshold eventually. Worth stating as "the hypothesis was wrong"
rather than quietly filing it as confirmed.

→ [experiments/exp05_regime_change_detection/RESULTS.md](../experiments/exp05_regime_change_detection/RESULTS.md)

## Experiment 6 — Does the backdoor criterion correctly predict which adjustments remove confounding bias?

**Positive, and the cleanest, most textbook-dramatic result of the six.**
The first work on Phase 5 (causal reasoning): `transintelligence/reasoning/causal/`
was an empty stub, same starting point as `temporal/` before Phase 4.
Built `CausalGraph` (d-separation via path enumeration, verified against
the three canonical structures — chain, fork, collider — before trusting
it for anything) and the backdoor criterion (Pearl, 1995) on top of it,
plus a small linear-regression utility for effect estimation.

Constructed the classic "correlation is not causation" demonstration
directly, the same way Experiment 5 built DTW's motivating case: a
confounder `Z` drives both a treatment `X` and an outcome `Y`, `X`'s true
causal effect on `Y` is fixed at exactly zero, and a fourth variable `W`
is a *collider* — a common effect of `X` and `Y`, not a cause of either.

The graph-theoretic prediction, computed with no data at all, matched the
empirical result exactly: naive regression finds a large, entirely
spurious effect (0.881, when the truth is 0.0) from confounding alone;
adjusting for `Z` — the only backdoor-valid set — recovers ~0 (0.0065);
adjusting for the collider `W` instead produces its own distinct bias
(0.281). The sharpest finding: **adding `W` on top of the already-correct
`{Z}` adjustment makes the estimate worse, not neutral** (bias rises from
0.031 to 0.191, even flipping sign) — the same lesson experiments 3-5
already established with completely different mechanisms: including more
or any extra information is not a safe default, and the graph-theoretic
validity check is doing real, load-bearing work.

→ [experiments/exp06_confounding_bias/RESULTS.md](../experiments/exp06_confounding_bias/RESULTS.md)

## Experiment 7 — Can a per-unit counterfactual be recovered exactly, not just on average?

**Positive, and mechanically exact — the same clean pattern as experiment
6, extended from a population question to a per-unit one.** Experiment 6
asked "does X affect Y on average, adjusting for confounders." This asks
a genuinely different question: "what would *this specific unit's* Y have
been, had its X been different." Filled `CounterfactualReasoner`, the
last reasoning protocol stub that predated Phase 4 — `Predictor`,
`Simulator`, `Planner`, `Verifier` remain unbuilt.

`StructuralCausalModel` implements Pearl's three-step abduction-action-
prediction procedure: infer a unit's own exogenous noise from what was
observed, fix the intervened variable, recompute everything else using
that *same* unit-specific noise — not the population average. The
obvious shortcut (skip abduction, just plug the new treatment value into
the fitted population regression) is an unbiased estimator of the
*average* effect, so the real test had to be per-unit accuracy, not
average accuracy, or the naive shortcut could look deceptively fine.

With the true structural coefficients, abduction recovers the exact
per-unit counterfactual (error 0.0000) while the naive shortcut's error
(0.2413) matches the theoretical average magnitude of the simulated
noise almost exactly — confirming precisely *why* it's wrong: its error
literally equals the residual it silently assumes is zero for every unit.
With estimated (not true) coefficients — the realistic case — abduction
still wins by roughly 9x, and the naive error barely moves between the
two conditions, because discarding a unit's own residual is a structural
problem no amount of additional data can fix, unlike abduction's small
remaining error, which does shrink with more data.

→ [experiments/exp07_counterfactual_queries/RESULTS.md](../experiments/exp07_counterfactual_queries/RESULTS.md)

## Experiment 8 — Can causal structure be recovered from data instead of assumed?

**Positive, with a negative control that specifically rules out the way
this result could have been an illusion of statistical power.**
Experiments 6 and 7 both assumed the graph was given. This asks the prior
question: can constraint-based discovery (Spirtes & Glymour's PC
algorithm, 1991 — skeleton recovery via Fisher-z partial-correlation
independence tests, then collider/v-structure orientation for unshielded
triples) recover it at all?

A structural subtlety mattered before any code was written: experiment
6's confounding graph has a collider `W` that's **shielded** (`X→Y` is
also a direct edge), so v-structure orientation cannot and should not
fire there. That graph is still valid for testing skeleton recovery
(reused with a nonzero `X→Y` effect this time, since discovery is purely
statistical and a true-zero coefficient would make the edge genuinely
vanish), but a **separate, dedicated unshielded-collider graph**
(`A→B←C`, `A` and `C` independent) was built specifically to test
orientation — the confounding graph could never provide a positive case
for it.

Across sample sizes from 100 to 3000 (20 seeds each): skeleton precision
on the confounding graph is ≈1.000 throughout, and recall climbs from
0.750 (n=100) to 1.000 (n≥1000) as weaker edges become statistically
distinguishable from noise; `W` is never falsely oriented as a collider,
0/80 trials at any sample size. On the dedicated unshielded graph,
skeleton recovery and correct orientation are both exact (20/20) even at
the smallest sample size tested. **The confound control — four mutually
independent variables, swept across the same sample sizes — shows the
false-edge rate shrinking as N grows (0.100 mean false edges/trial at
n=100, down to 0.000 at n=3000), not climbing the way it would if the
positive results above were just an artifact of growing statistical
power.**

→ [experiments/exp08_causal_discovery/RESULTS.md](../experiments/exp08_causal_discovery/RESULTS.md)

## Experiment 9 — Do IV and front-door adjustment work when a confounder is never observed, and fail how theory predicts when their assumptions don't hold?

**Positive on both mechanisms, and both positives come with a matching,
directly-demonstrated breaking point.** Experiments 6 and 8 both assumed
or discovered a graph where every relevant variable was in the dataset.
This asks what happens when the confounder itself is never observed at
all — the case neither the backdoor criterion nor discovery can touch —
using the two classic identification strategies for exactly that
situation: two-stage least squares (via a valid instrument) and
front-door adjustment (via a fully-mediating observed variable).

With a strong instrument, 2SLS cuts bias from an unobserved confounder by
roughly 30x relative to naive regression (0.0131 vs. 0.4118). But sweeping
instrument strength down to near-zero found the textbook weak-instrument
pathology in its most dramatic form: at instrument strength 0.05,
individual-seed estimates ranged from **-96.3 to +29.4** — a ~390x
increase in standard deviation over the strong-instrument case, and the
mean estimate ends up *more* biased than naive, not less. 2SLS doesn't
degrade gracefully; below a threshold it becomes actively worse and
wildly unstable.

Front-door adjustment recovers the true effect almost exactly when its
structural assumption holds (bias 0.0094 vs. naive's 0.8807). Giving the
confounder a direct effect on the mediator too — violating exactly the
assumption front-door identification requires — produces bias nearly as
bad as naive (0.9304, a ~99x jump), using the identical adjustment code
on data that now breaks its precondition. The same "adjusting incorrectly
is worse, not neutral" lesson experiment 6 established for the backdoor
criterion, now confirmed for a second, structurally different mechanism.

→ [experiments/exp09_iv_and_frontdoor/RESULTS.md](../experiments/exp09_iv_and_frontdoor/RESULTS.md)

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
- Correlated cross-key covariance in the joint-detection statistic (the
  current combination assumes independence across keys); non-i.i.d. noise
  beyond the single contaminated-Gaussian case tested; more than 5 keys
  tracked at once.
- Nonlinear structural equations (both `reasoning/causal/` and
  `reasoning/counterfactual/` assume linearity throughout).
- Meek's further orientation-propagation rules (UAI 1995) for causal
  discovery — only skeleton recovery plus direct collider orientation
  were implemented and tested; nonlinear dependencies that produce zero
  *linear* partial correlation would be missed entirely by the Fisher-z
  independence test used; graphs larger than 4-5 nodes weren't tested;
  no comparison against a score-based discovery method (e.g. GES).
- Overidentified 2SLS (more instruments than endogenous regressors) and
  overidentification tests that could detect instrument invalidity from
  data; an automated weak-instrument diagnostic (e.g. a first-stage F
  test); front-door adjustment's other two structural preconditions
  (full mediation, no unblocked backdoor path from mediator to outcome
  other than through treatment) weren't separately stress-tested, only
  "confounder also affects the mediator" was; sensitivity-analysis
  approaches to unobserved confounding that don't require a valid
  instrument or mediator at all (e.g. Rosenbaum bounds).
- Multi-step or sequential interventions, and a computational comparison
  against Rubin's potential-outcomes framework (noted as the alternative
  formalization, not implemented or benchmarked against).
- The master context's remaining later phases (world models; agency;
  meta-intelligence) — all still pre-formalization, per
  `research-agenda.md`'s own sequencing.
