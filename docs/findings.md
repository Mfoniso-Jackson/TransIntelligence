# Findings

A standalone summary of what the first twenty experiments in
[research-agenda.md](research-agenda.md) actually established, for anyone
who wants the result without reading twenty `RESULTS.md` files and the
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
that could have been a statistical-power illusion, a ninth showed even an
unobserved confounder doesn't block identification given a valid
instrument or mediator, while directly demonstrating both of those
alternative strategies' own theory-predicted breaking points rather than
just asserting them, and a tenth closed out the linearity question every
one of the causal/counterfactual mechanisms had left open: exact for
counterfactual abduction under a nonlinear structural equation, but
substantially biased — and structurally unable to represent a
heterogeneous effect at all — for linear effect estimation under one,
an eleventh (opening Phase 6, world models) found the same lesson
recurring in a genuinely different setting: an agent that learns forward
dynamics and plans by simulating actions matches an oracle's performance
ceiling almost exactly, while an equally state-aware, identically-tooled
model-free baseline falls far short, because the true value surface has
a peak that no linear fit can represent, no matter how much data it
gets — confirmed directly by a follow-up showing a correctly-specified
nonlinear baseline closes almost the entire gap — a twelfth found
that planning further ahead (not just having a model at all) gives a
real, consistent, but honestly modest advantage over greedy single-step
lookahead, with a 2×2 design confirming the advantage is specifically
about how far ahead the agent looks, not an accidental difference in
model quality, a thirteenth combined two already-verified mechanisms
(change detection, learned dynamics) and found the honest answer is
conditional — no benefit under a mild regime shift, a real, investigated
null result rather than a smoothed-over one, but a clear,
confound-controlled benefit under a severe one, and a fourteenth set out
to test a small, expected scalability fix (beam search vs. exhaustive
search for the planner) and found something sharper by investigating a
result that looked wrong rather than reporting it: beam search isn't
just cheaper, it's measurably more robust to a genuine receding-horizon
oscillation pathology exhaustive search's terminal-only scoring is
vulnerable to, and a fifteenth closed out a three-object arc this
program's linearity theme has now traced end to end — causal effect
estimation, single-step value estimation, and world-model dynamics
prediction — each confirming a linear fit cannot represent a
nonlinearity, and each needing the misspecification to be severe enough
to actually change a decision before the gap became visible at all, and
a sixteenth, a second synthesis experiment, independently replicated the
fourteenth's oscillation pathology in an unrelated environment before
finding something new only visible in combination: learned multi-step
planning persistently underperforms learned single-step planning once
combined with regime-adaptation, with the obvious "reduced exploration"
explanation checked directly and refuted, and a seventeenth closed out
Phase 6 not with a new claim but a validation: a newly-built Monte Carlo
policy comparator, using short stochastic rollouts rather than full
environment averages, independently recovered the sixteenth's exact
ranking in 58 of 60 comparisons, with a built-in sensitivity control
confirming the rollout count itself was doing real work, and an
eighteenth took the sixteenth's own best-supported but unconfirmed
explanation and tested it directly with a newly-built calibration check:
the multi-step planner's dynamics model is not measurably worse-
calibrated than the single-step planner's, ruling out one plausible
alternative explanation and, by elimination, leaving the original
chained-prediction hypothesis the more plausible remaining one, still
not directly confirmed, and a nineteenth extended regime-adaptation to a
nonlinear dynamics model and found the composition does not transfer for
free from the linear case: a mild shift, an honest null result before,
actively hurts here, traced to a real cold-start cost; a severe shift
still favors adaptation, but change detection itself becomes measurably
less reliable for a reason checked and ruled out but not identified, and
a twentieth went back and closed the loop the sixteenth and eighteenth
both left open — directly manipulating the environment's own noise level
and finding the persistent multi-step-planning gap scale strongly with
it, near-zero at zero noise and largest at the highest noise tested, the
first direct positive confirmation (not just an alternative ruled out)
of the mechanism this whole sub-thread was chasing.

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

Across the twenty experiments, the same discipline applied every time: build
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
precondition fails. Experiment 10's control condition (a genuinely linear
truth) confirmed its own setup was sound before the nonlinear condition's
bias could be trusted as evidence of anything — the same discipline
applied to a mechanism-generalization result rather than a treatment-
vs-control one. Experiment 11's confound control (an equally
state-aware, identically-tooled model-free baseline, not just a
state-blind floor) is what turned "the world-model agent scored higher"
into a specific, mechanistic claim rather than a demonstration that
context helps at all — and the follow-up investigation into *why* the
fair baseline still fell short (rather than reporting the gap and moving
on) is what surfaced the actual reason: a non-monotonic value surface a
linear fit can never represent, not a fixable data problem. Experiment
12's 2×2 factorial design (planning horizon × model source) is its own
confound control, built in rather than added after: without it, "the
multi-step planner scored higher" could have meant either "planning
ahead helps" or "this run's learned model happened to be better," and
the design makes those two possibilities separable rather than
conflated — the near-identical gap size in the learned and oracle pairs
is what makes the horizon claim trustworthy. Experiment 13's first run
produced a genuine null result (mild-shift adaptation showed no benefit
at all) that got investigated rather than discarded or re-run until it
looked better — the explanation it surfaced (miscalibration only matters
when it changes which action ranks best) is what justified testing a
second, more severe condition, where the confound-controlled positive
result (beating a naive recency heuristic, not just doing nothing)
actually held up. Experiment 14 caught two of its own methodological
bugs (a single-step evaluation that couldn't measure the objective it
claimed to; tie-breaking degeneracy in a fine action grid) before
trusting a result that initially looked impossible (beam search
"beating" exhaustive search by construction) — and what survived after
fixing both was a real, quantified finding the experiment wasn't
designed to find. Experiment 15's own boundary-condition sweep (a weak
nonlinearity, checked directly, showed no gap before a stronger one
did) is the same discipline applied a third time to the same underlying
theme — not assuming the expected result would appear just because the
mechanism was technically misspecified. Experiment 16 pushed the
discipline furthest yet: not content that beam search fixed the
oscillation pathology it replicated from experiment 14, it kept
investigating why the resulting numbers still didn't match expectations,
directly checked and refuted its own first hypothesis (reduced
exploration), and reported the best-supported remaining explanation as a
hypothesis rather than dressing it up as a confirmed finding. No
experiment that was actually pushed on came back unqualified. None of
the sixteen hypotheses were fully falsified, but none survived untouched
either — that's the intended
outcome of the experimental discipline in
[research-agenda.md](research-agenda.md) §21, not a failure of it. A
result that survives its own strongest test is worth more than one that
was never tested against one, and the one experiment where the *core*
control confirmed rather than shrank the effect is still the one worth
the most trust — its remaining caveats are about scope, not about whether
the central claim is real.

Experiment 17 is a different kind of check than the sixteen hypothesis
tests above it — not "does this claim about the world hold up," but
"does a newly-built tool (`MonteCarloSimulator`) reproduce, by an
independent method, a result already established by hypothesis-testing
discipline." It carried the same real risk of failure (the answer was
known in advance, so a mismatch would have been reportable either way),
and it passed — the closest thing this program has to an external
validity check on its own experimental machinery, not just on its
claims.

Experiment 18 goes back to being a genuine hypothesis test — its answer
was not known in advance, unlike experiment 17's — but built on the tool
experiment 17's sibling primitive made possible: `CalibrationVerifier`
gave experiment 16's own best-supported-but-unconfirmed explanation a
real chance to be *directly* checked rather than only argued from reward
patterns, and one competing explanation for it was ruled out as a
result, not just left unaddressed. Experiment 19 is a hypothesis test
too, and the clearest demonstration yet of the "investigate a result
that looks wrong instead of reporting it" discipline applied to a
*negative* surprise: an oracle performing worse than an agent that never
adapts at all is exactly the kind of result that invites either
suspicion of a bug or a quiet rerun with different settings — instead it
was traced to a specific, confirmed mechanism (a nonlinear model's
cold-start data requirement) via direct instrumentation, not just
argued from the pattern of final rewards. Experiment 20 closes out this
particular thread: where experiment 18 tested and ruled out one
*alternative* to experiment 16's hypothesis, experiment 20 manipulated
the hypothesis's own stated cause directly (the environment's noise
level) and watched the predicted pattern actually appear — the
difference between narrowing a field of explanations and confirming the
survivor positively. That makes nineteen genuine hypothesis tests total
(the sixteen above, plus 18, 19, and 20), with experiment 17 as the one
deliberate exception — a validation rather than a claim about the world.

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

**Follow-up: Meek's orientation-propagation rules.** Collider orientation
alone can only ever find direct v-structures — a hard 0.500 recall
ceiling on a fully-identifiable collider-then-chain graph (`A→C←B,
C→D→F`), confirmed exactly. Adding three of Meek's four rules (R1-R3;
R4 provably cannot fire without a background-knowledge mechanism this
implementation doesn't have) extends recall to 1.000 once the skeleton
is reliably correct. An apparent wrong orientation at one seed, on
investigation, turned out to be a skeleton-recovery error propagating
downstream, not a bug in the rules themselves — given a correct
skeleton, they never oriented wrongly across 80 trials. A negative
control (a plain collider-free chain, genuinely undetermined by any
amount of data) confirmed zero spurious orientations at any sample size.

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

## Experiment 10 — Does abduction stay exact under nonlinearity, and does linear effect estimation actually break?

**Both predictions held, and the nonlinear case's mismatch was more
dramatic than a simple "biased estimate."** Every causal/counterfactual
mechanism in this program assumed linear structural equations from the
start. This is the last of the three stated gaps: does that assumption
matter for counterfactual abduction (which was always argued to need
only additive noise, not linearity), and does it matter for
`reasoning/causal/`'s OLS-based effect estimation (which does assume
linearity)?

`StructuralEquation` now accepts an arbitrary nonlinear function in place
of its linear coefficients, and `abduct()`/`counterfactual()` needed
**zero changes** to support it — verified exact on a hand-computed
quadratic case first. The experiment then reused experiments 6/7/9's
confounding-graph shape with a quadratic `X→Y` relationship, and computed
the **true average shift effect exactly** (not estimated) by exploiting
the same noise-cancellation trick experiment 7 used for the linear case.

A linear control condition (the truth really is linear) confirmed
linear-adjusted OLS matches the true effect closely there (gap 0.0121) —
ruling out "the experimental setup itself is broken" before trusting the
positive result. In the nonlinear condition, the gap grew to **0.5799**,
but more strikingly: the linear-adjusted estimate itself barely moved
from the control condition's value, because the treatment variable's
near-zero skew makes it nearly blind to the entire quadratic
contribution rather than reporting a scaled-down version of it. At five
fixed reference points, the true effect ranged from **-1.5 to +3.3**,
even flipping sign, while the linear model predicted the identical
number at every one of them — not a quantitative miss, a category
error: a single coefficient cannot represent an effect that depends on
where a unit starts.

→ [experiments/exp10_nonlinear_scm/RESULTS.md](../experiments/exp10_nonlinear_scm/RESULTS.md)

## Experiment 11 — Does explicit dynamics modeling beat model-free value estimation? (Phase 6, the first world-models result)

**Positive, and the same lesson experiment 10 found for effect
estimation reappears here in a planning setting instead.** The master
context formalizes a world model as `M(S_t, A_t) → S_{t+1}` — a
*transition* model, useful because predicted outcomes of different
actions from the same state can be compared. This is Phase 6's first
result, filling the `Predictor` protocol stub that had been empty since
Phase 1.

Showing a state-aware agent beat a state-blind one would only prove
using context helps at all — trivial. The real test needed a second,
equally state-aware baseline that uses the identical regression tool but
never models the transition: `model_free_linear_q` fits reward directly
as a linear function of state per action, while `world_model` fits the
(truly linear) state *transition* per action and applies the known
(quadratic) reward formula to the result.

`world_model` matched the oracle's true performance ceiling almost
exactly (regret **-0.0002**), while the identically-tooled
`model_free_linear_q` fell far short (regret **3.5421**) — roughly 3.5x
further from optimal despite seeing the same states and using the same
tool. Investigated *why*, not just reported: each action's true reward
is a downward parabola in state, peaking exactly where that action lands
the state on target. A linear fit is forced to be monotonic — it tracks
the parabola's rising side but has no way to represent the
peak-then-decline, so it systematically misranks actions once the state
passes an action's optimal zone. **The value surface isn't just
nonlinear, it's non-monotonic — which no amount of data lets a linear
model represent, the exact structural reason experiment 10 found a
linear coefficient can't capture an effect that depends on where a unit
starts.**

→ [experiments/exp11_world_model_planning/RESULTS.md](../experiments/exp11_world_model_planning/RESULTS.md)

**Follow-up confirmed the mechanism directly**: giving the model-free
baseline a quadratic (correctly-specified) feature set — a function
class that *can* represent the true parabola — closed almost the entire
gap (regret 3.5421 → 0.0144, landing at essentially `world_model`'s
level). This rules out an unaccounted-for confound between the two
conditions as the real explanation.

## Experiment 12 — Does multi-step planning beat greedy lookahead, and is the advantage really about horizon?

**Positive, real, consistent — and honestly modest, not talked up.**
Experiment 11's environment has no delayed effects, so greedy 1-step
lookahead was already optimal there; nothing to test multi-step planning
against. This experiment builds the smallest environment where that
stops being true: each action's effect splits between landing
immediately and landing one step later, and episodes now persist across
steps with reward only at the end — a genuine multi-step
credit-assignment problem.

A 2×2 factorial design (greedy vs. multi-step receding-horizon planning,
crossed with learned vs. oracle dynamics) is what makes the result
trustworthy: multi-step planning beat greedy lookahead by almost
exactly the same margin whether the dynamics model was learned (0.0246 →
0.0204) or known exactly (0.0241 → 0.0200) — confirming the advantage is
really about how far ahead the agent looks, not an accident of one
condition's model happening to be better. The effect held in 14 of 15
seeds. **The size of the effect (~17% relative reduction in error) is
reported honestly, not inflated**: because every step gives the agent
full feedback and lets it replan, a myopic policy already self-corrects
fairly well over several free steps — multi-step planning's real
advantage here is efficiency, not preventing catastrophic mistakes.

→ [experiments/exp12_multistep_planning/RESULTS.md](../experiments/exp12_multistep_planning/RESULTS.md)

## Experiment 13 — Does combining a world model with regime-change detection work as expected?

**Conditional — a real null result at mild severity, a real
confound-controlled positive at severe severity, both reported
honestly.** This is the first synthesis experiment in this program: it
combines two already-verified mechanisms (`CUSUMTemporalReasoner` from
Phase 4, `LinearDynamicsModel` from Phase 6) rather than testing a new
one, asking only whether they compose as expected when wired together.

The first version tested one fixed, mild regime shift and found
`oracle_adapts` (told the true shift trial exactly, zero detection
delay) performed statistically indistinguishably from `never_adapts` —
the opposite of the hypothesis. Investigated rather than reported as a
clean result: this environment's state range is wide relative to its
nudge magnitudes, so a stale and a correctly-calibrated model usually
pick the *same* largest-available action anyway — mild miscalibration
rarely changes which action ranks best, while discarding a large body of
converged prior data for a small, noisy post-shift refit has a real,
visible cost (confirmed directly by inspecting the reward trajectory:
`oracle_adapts` starts *worse* than `never_adapts` right after the
shift, before the two converge).

That turned the experiment into a two-severity sweep. At a severe
shift (the actuator's effect reverses direction entirely),
`never_adapts` collapses (post-shift reward ~8x worse than pre-shift),
while `cusum_detects_and_adapts` recovers to near the oracle ceiling and
**beats `sliding_window_baseline`** — the confound control that
mattered: explicit detection measurably outperforms the simpler
always-use-recent-data heuristic, not just "doing something." A real
limitation surfaced along the way, reported rather than glossed over: a
20% false-positive detection rate, notably higher than experiment 5's
~6% under the identical default calibration — traced to a specific,
understood cause (this experiment's residual stream comes from a
periodically-refit model, whose own re-fits introduce small genuine
jumps a stationary raw signal wouldn't have).

**Follow-up confirmed the diagnosis, and found it's a tradeoff, not a
free fix**: refitting the dynamics model twice as often roughly halved
the false-positive rate (4/15 → 2/15), but also cost two missed true
detections under the severe shift (15/15 → 13/15) — successful
detections did get much faster (23.0 → 2.5 trials), but which side of
that tradeoff is worth it depends on the application, not something this
experiment resolves to a single answer.

→ [experiments/exp13_regime_shift_world_model/RESULTS.md](../experiments/exp13_regime_shift_world_model/RESULTS.md)

## Experiment 14 — Does beam search scale `RecedingHorizonPlanner` past exhaustive search's limits, and at what cost?

**The expected, small result (cheaper at a small quality cost) turned
out not to be the real finding — investigating a result that looked
wrong instead of reporting it surfaced something sharper.**
`RecedingHorizonPlanner`'s exhaustive search was flagged from the start
as not scaling past small action sets and shallow lookahead. Adding beam
search was meant to be a routine fix.

Two real methodological bugs were caught before any result could be
trusted. First, scoring only a single executed step (not the multi-step
rollout the search was actually optimizing for) made beam search look
like it was "beating" exhaustive search — impossible by construction,
since exhaustive search checks every sequence at a given depth. Fixed by
scoring a full 5-step receding-horizon rollout with replanning at every
step. Second, even after that fix, a fine 21-action vocabulary produced
massive score ties (one state had 19 different 3-step sequences all
scoring exactly optimal, with wildly different first actions), and
exhaustive search's tie-breaking rule always favored the most extreme
one — fixed by reusing experiment 12's original, widely-spaced 6-action
set instead.

**What remained after both fixes was real: at depth 3, exhaustive search
converges to the target in only 2 of 15 starting states, while every
beam width tested converges in 13-15 of 15 — using 12-22x fewer
`transition_fn` calls.** Traced one misbehaving state step by step:
exhaustive search's policy enters a persistent oscillation (position
stuck away from target, forever alternating), while beam search from
the identical starting state converges and stays at target. The
mechanism: exhaustive search scores only the *final* simulated state
after the full lookahead, blind to the path — it can select a sequence
whose promised final position looks optimal while its first action (the
only one ever executed before replanning from scratch) sets up a
self-reinforcing overshoot cycle. Beam search scores every intermediate
partial state during its own expansion, incidentally biasing toward
monotonic progress that happens to avoid exactly this pathology. Beam
search isn't just a cheaper approximation of exhaustive search here —
it's a better match for what receding-horizon control actually needs.

→ [experiments/exp14_beam_search_planning/RESULTS.md](../experiments/exp14_beam_search_planning/RESULTS.md)

## Experiment 15 — Does a linear world model fail under genuine nonlinearity, and does a correctly-specified nonlinear one recover it?

**Positive, with the same boundary-condition discipline experiment 13
established, applied a third time to the same underlying theme.**
Every world model in this program (`LinearDynamicsModel`, experiments
11-14) assumes linear dynamics. This tests the cost of that assumption
for a third distinct object — after causal effect estimation (experiment
10) and single-step value estimation (experiment 11) — for world-model
dynamics prediction itself.

A weak version of a quadratic restoring-force nonlinearity showed almost
no gap between the linear and a correctly-specified nonlinear dynamics
model — checked directly, not assumed, and matching the exact pattern
experiment 13 found for a mild regime shift: a nonlinearity too small to
change which discrete action ranks best doesn't produce a measurable
gap. Swept the nonlinearity's strength up rather than stopping at the
null result. A stronger version produced a clean, dramatic gap: the
correctly-specified nonlinear model matched the oracle ceiling almost
exactly (regret 0.0036 vs. the oracle's 0.0014), while the linear
model's regret (0.6933) was roughly **193x larger** — despite identical
state access and the identical `ordinary_least_squares` tool, differing
only in which features were fit. The linear model still meaningfully
beat ignoring state entirely, though — wrong, but not worthless, the
same pattern experiment 11's model-free linear baseline showed.

→ [experiments/exp15_nonlinear_world_model/RESULTS.md](../experiments/exp15_nonlinear_world_model/RESULTS.md)

## Experiment 16 — Does regime-change detection compose with multi-step planning?

**Two findings, neither predictable from testing the pieces separately —
exactly why this experiment was built.** A second synthesis experiment
(after experiment 13): does `CUSUMTemporalReasoner`-triggered adaptation
still work when the planner is multi-step (`RecedingHorizonPlanner`,
experiments 12/14) instead of single-step greedy? Nothing new is
implemented; every mechanism is reused exactly as already verified.

**Finding 1, an unprompted replication.** The first run's `oracle`
condition — given the true dynamics exactly — scored worse post-shift
than a *learned*, adaptive greedy policy, which should be impossible for
a true oracle. Investigated rather than accepted: the cause was exactly
experiment 14's oscillation pathology, independently reproduced in a new
environment built for a different purpose. Beam search fixed it the same
way it did in experiment 14: the oracle's post-shift performance went
from dramatically degraded to matching its pre-shift quality almost
exactly.

**Finding 2, the actual answer, and the more interesting one.** Fixing
the search pathology did *not* make multi-step planning competitive with
single-step planning once both used the same learned, CUSUM-adaptive
dynamics model (post-shift reward -33.06 vs. -4.52). Traced across the
whole post-shift window in 8 chunks: the gap never closes, even with
hundreds of post-reset samples accumulated by the end of the run — not a
shrinking startup transient. The natural explanation (multi-step
planning converges to a narrow region, starving the model of diverse
training data) was checked directly and **refuted**: the multi-step
policy's visited positions had a *larger* spread than the single-step
policy's, not smaller. The best-supported remaining explanation,
reported as a hypothesis rather than a confirmed cause: multi-step
lookahead chains two predictions from the same learned model, and the
estimation error each one carries compounds in a way single-step
lookahead never has to pay — a cost that doesn't shrink with more data,
unlike a small-sample transient.

→ [experiments/exp16_regime_shift_multistep_planning/RESULTS.md](../experiments/exp16_regime_shift_multistep_planning/RESULTS.md)

## Experiment 17 — Does `MonteCarloSimulator`, a newly-built kernel primitive, independently recover experiment 16's already-established finding?

**A validation, not a new claim — closing Phase 6's last remaining gap
(`Simulator`).** `transintelligence/simulation/`'s new
`MonteCarloSimulator` compares two given candidate policies via
stochastic rollouts, rather than predicting one state (`Predictor`) or
searching internally for one best action (`Planner`). Hand-verified
first against a deterministic canonical case (always-up beats
always-down by exactly the hand-computed margin, zero spread, and
swapping argument order flips the reported winner), then run on a real,
previously-unseen question: reusing experiment 16's own trained agents
and environment exactly, freezing them (learning switched off), does
`MonteCarloSimulator.compare_policies` — given only short stochastic
rollouts from fresh starting states — recover the same ranking
experiment 16 found via full 400-episode environment averages
(`greedy_cusum_adapts` beats `mpc_beam_cusum_adapts` post-shift)?

**Result: yes, cleanly.** 58 of 60 comparisons (12 seeds × 5 starting
states) favored greedy, with mean simulated rewards (-2.29 vs. -32.98)
landing close to experiment 16's own numbers (-4.52 vs. -33.06) despite
the completely different methodology. A built-in sensitivity control —
dropping `n_rollouts` from 200 to 5 — reduced the win rate to 88.3%,
confirming the rollout count genuinely affects verdict reliability rather
than being cosmetic, while also showing the underlying effect is large
enough that even a small sample rarely gets the ranking backwards. This
was a genuine risk, not a formality: the answer was already known, so a
mismatch would have meant either a bug in the new code or a real
limitation of experiment 16's own result.

→ [experiments/exp17_monte_carlo_simulator/RESULTS.md](../experiments/exp17_monte_carlo_simulator/RESULTS.md)

## Experiment 18 — Does `CalibrationVerifier` find direct evidence for experiment 16's compounding-estimation-error hypothesis?

**Fills `Verifier`, the last reasoning-protocol stub from before Phase 4
left unbuilt — and, unlike experiment 17, a genuine new hypothesis test,
not a validation against an already-known answer.** Experiment 16's
central finding came with a best-supported but explicitly *unconfirmed*
explanation: multi-step lookahead chains two predictions from the same
learned dynamics model, and each carries estimation error that compounds
in a way single-step lookahead never pays for. That was inferred from
the pattern of final rewards, never checked against either agent's
actual model error directly. Both agents fit the identical one-step OLS
model — the only difference is how far ahead each searches — but each
agent's own planning strategy steers it into different regions of state
space (experiment 16 already measured `mpc_beam`'s wider post-shift
position spread), feeding back into what data its own model trains on.
`CalibrationVerifier` (Kupiec's 1995 unconditional-coverage test, applied
to each agent's one-step residuals against the environment's true noise
floor) asks directly: does `mpc_beam`'s model carry more excess
estimation error than `greedy`'s, post-shift?

**Result: no.** Both agents' one-step models show the same small,
universal miscalibration (~66% observed vs. 68.27% claimed coverage) —
present pre-shift too, identically for both agents, ruling it out as a
regime-shift artifact (a baseline property of the fitting procedure,
most plausibly ordinary in-sample-adjacent OLS residuals understating
true out-of-sample variance). The gap between `greedy` and `mpc_beam`
specifically is under 0.002 post-shift, smaller than either agent's own
pre-to-post-shift shift (~0.007) — not a real, attributable difference.
A robustness check restricting to only the first 500 post-shift steps
per seed (closer to experiment 16's largest-gap window) shows the same
pattern.

**What this establishes, and what it doesn't.** The "self-steered
training distribution differentially degrades the model" alternative
explanation is ruled out directly — genuine new evidence, not just an
absence of evidence. By elimination, this leaves experiment 16's
original chained-prediction-compounding hypothesis the more plausible
remaining explanation for the reward gap, but the chaining mechanism
itself still was not directly manipulated — the hypothesis is narrower
now, with one fewer competitor, not confirmed.

→ [experiments/exp18_calibration_verifier/RESULTS.md](../experiments/exp18_calibration_verifier/RESULTS.md)

## Experiment 19 — Does regime-adaptation still work when the dynamics model is nonlinear?

**A third synthesis experiment (after 13 and 16), and a genuinely
new, mechanistically-explained finding — not the "it just works the same
way" result the setup might suggest.** Structurally identical to
experiment 13, with one substitution: `NonlinearDynamicsModel`
(generalized from experiment 15 into a kernel primitive alongside
`LinearDynamicsModel`) in place of the linear one. `CUSUMTemporalReasoner`
monitors residuals agnostic to whether the underlying model is linear —
but a nonlinear model needs one more observation per action to fit at
all (3, not 2), and there was no guarantee a detection/adaptation
pipeline implicitly calibrated around a linear model's behavior would
transfer unchanged.

**It doesn't, in two distinct ways.** Under a mild shift — an honest
null result for the linear model in experiment 13 — adaptation here
actively *hurts*: an oracle given the true shift trial exactly performs
worse than an agent that never adapts at all. Confirmed directly, not
inferred: instrumentation shows a post-shift agent's known actions stay
completely empty for the entire first refit window, forcing pure random
action selection — the nonlinear model's higher data requirement gives
any hard-reset adaptation strategy a real, measurable cold-start tax a
linear model's shorter cold start doesn't produce. Under a severe shift,
adaptation still wins overall, matching experiment 13 — but change
detection itself becomes measurably less reliable (barely more than half
the seeds detect the shift at all, and detection that does happen takes
roughly five times as long) for a reason checked directly and *refuted*
(residual noise in steady state is statistically identical between the
linear and nonlinear settings) but not otherwise identified — reported
as an open question, not dressed up with an unverified explanation. A
follow-up ruled out a second candidate too (residual autocorrelation)
and localized the real difference to post-shift, pre-detection behavior
specifically — plausibly the stale model's own action choices
interacting with the now-shifted environment more erratically than the
linear model's do — without fully explaining why that interaction is
noisier for a nonlinear model. A simpler heuristic that never fully
empties its training data sidesteps the cold-start cliff entirely and
ends up more practically robust than
change detection here, reversing experiment 13's own preference.

→ [experiments/exp19_nonlinear_regime_shift/RESULTS.md](../experiments/exp19_nonlinear_regime_shift/RESULTS.md)

## Experiment 20 — Does experiment 16's central hypothesis hold up when the environment's own noise is directly manipulated?

**The most literal possible test of a claim this program had been
circling since experiment 16, and the first one to confirm it directly
rather than only rule out a competitor.** Experiment 16 proposed that
multi-step planning's persistent post-shift underperformance comes from
chaining two predictions from the same learned model, each carrying
estimation error that never fully vanishes *because the environment has
real observation noise*. Experiment 18 tested a specific alternative
explanation and ruled it out, but never touched the stated cause itself.
This experiment does: sweep the environment's own noise level from zero
up to four times its original value and watch what happens to the gap.

**It scales strongly, close to the predicted shape.** Near zero at zero
noise (a deterministic environment gives a correctly-specified model
nothing left to compound), roughly 17-21x larger at or above the
original noise level used throughout experiments 16-18. Detection
reliability — a plausible confound, given experiment 19 already found it
sensitive to a model's residual behavior — was tracked at every noise
level for both agents and stays closely matched throughout, ruling it
out as an alternative explanation for the pattern. A first pass at 10
seeds showed an apparent dip at one noise level that looked like it
might break the trend; doubling the seed count made it mostly disappear,
and the result reported here is the checked version, not the first one
run.

→ [experiments/exp20_noise_sweep_compounding_error/RESULTS.md](../experiments/exp20_noise_sweep_compounding_error/RESULTS.md)

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
- Non-additive-noise structural equations (`reasoning/counterfactual/`'s
  abduction now handles nonlinear-but-additive-noise functions exactly,
  but genuinely non-additive noise, or non-monotonic/discontinuous
  functional forms beyond the single quadratic case tested, would break
  the closed-form residual entirely); a nonlinear effect-estimation
  method for `reasoning/causal/` to replace/complement OLS under known
  nonlinearity; nonlinear functional forms discovered from data rather
  than given (interacting with causal discovery's own linear-only
  independence test, below).
- Meek's fourth orientation rule (R4, UAI 1995) — not a gap so much as a
  rule that provably cannot fire in this no-background-knowledge
  pipeline; R1-R3 are implemented and are established to be complete for
  the CPDAG without background knowledge. Also untested: nonlinear
  dependencies that produce zero *linear* partial correlation, which
  would be missed entirely by the Fisher-z independence test used;
  graphs larger than 4-5 nodes; no comparison against a score-based
  discovery method (e.g. GES).
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
- Nonlinear dynamics (the world-model's advantage in experiment 11 rests
  specifically on the true transition being exactly linear); whether
  model-free methods find the right feature set unprompted, in a real
  environment where the reward's functional form isn't known in advance
  (experiment 11's follow-up handed the model-free baseline the
  exactly-correct quadratic features rather than having it discover
  them).
- Multi-step planning beyond a 2-step lookahead and a single fixed
  1-step-lag structure (experiment 12); a nonlinear or momentum-based
  (rather than simple linear-lag) delay structure, which might show a
  more dramatic planning advantage than experiment 12's modest ~17% one.
  `RecedingHorizonPlanner`'s exhaustive-search scaling limit now has a
  beam-search alternative (`transintelligence/planning/`, experiment
  14), which turned out to be more robust to a receding-horizon
  oscillation pathology, not just cheaper — but whether a smarter
  exhaustive-search tie-breaking rule (e.g. preferring minimal-effort
  actions among ties) would close that gap, and whether the pathology is
  specific to this dynamics structure or general, are both untested;
  `Simulator` remains fully unbuilt.
- A continuous regime-shift-severity sweep, rather than experiment 13's
  two fixed severities (mild attenuation, full sign flip) — where
  adaptation's benefit actually crosses over from negligible to real is
  unmapped; a longer post-shift window (whether `never_adapts`'s
  mixed-data model eventually catches up as more post-shift data
  dilutes the stale pre-shift fit is untested); combining
  regime-change-adaptive world models with the multi-step planner
  (experiment 12) or the nonlinear dynamics model (experiment 15) —
  experiment 13 only tested the single-step, linear case. A follow-up
  found that refitting the dynamics model more often trades false
  positives for missed true detections (4/15 → 2/15 false positives, but
  15/15 → 13/15 true-detection rate) — a genuine precision/recall
  tradeoff, not a single "correct" refit interval; recalibrating CUSUM's
  own `h_sigma`/`burn_in` parameters specifically for this context,
  rather than only varying refit frequency, remains untested.
- Experiment 15's nonlinear dynamics fitting is an experiment-local
  class, not yet a kernel primitive the way `LinearDynamicsModel` is;
  only one nonlinear functional form was tested (a quadratic restoring
  force, with the correct feature handed to the agent rather than
  discovered); whether it composes with the multi-step planner
  (experiment 12) or regime-adaptation (experiment 13/16) is untested;
  the exact crossover point between "nonlinearity too weak to matter"
  and "matters a lot" wasn't mapped, the same limitation experiment 13's
  severity sweep had.
- Experiment 16's compounding-estimation-error hypothesis was not
  directly confirmed — the "reduced exploration" alternative was ruled
  out, but the proposed mechanism itself wasn't independently tested
  (e.g. by varying observation noise and checking whether the
  learned-vs-oracle gap scales with it); only one lookahead depth and
  beam width were tested; only one regime-shift severity (experiment
  13's dramatic sign-flip, reused rather than re-derived); combining
  regime-adaptation with the nonlinear dynamics model (experiment 15)
  remains untested.
- The master context's remaining later phases (agency; meta-intelligence;
  strange loops; cross-domain transfer) — all still pre-formalization,
  per `research-agenda.md`'s own sequencing. World models (Phase 6) has
  now started (experiments 11-16).
