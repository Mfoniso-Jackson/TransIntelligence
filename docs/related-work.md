# Related Work

## 1. Purpose

`docs/research-agenda.md` compresses the established-vs-novel question into
one table. This document expands it, with real citations, so that claims in
this repo can be checked rather than taken on faith. Every entry below was
verified by search before being written down here — nothing is cited from
memory. If a citation looks wrong, that's a bug in this file, file it as
such.

Classification key (same as `research-agenda.md`):

- **Established theory** — solved/standard, we just reuse it.
- **Synthesis** — a recombination of established ideas, not itself new.
- **Hypothesis** — a claim we believe but haven't tested.
- **Novel proposal** — we believe this is new; treat with the most
  skepticism, and preferentially assume it is synthesis until an experiment
  says otherwise.

## 2. Reference frames as conditioning context

**Closest prior art, and the one that matters most for Experiment 1.**

- Contextual bandits: Li, Chu, Langford, Schapire, *A Contextual-Bandit
  Approach to Personalized News Article Recommendation*, WWW 2010.
  `evaluate(x, context)` in a bandit setting, with regret guarantees, is
  decades-old and this is one of the canonical applied references
  (`ir.webis.de/anthology/2010.wwwconf_conference-2010.67`).
- POMDPs / belief states: Kaelbling, Littman, Cassandra, *Planning and
  Acting in Partially Observable Stochastic Domains*, Artificial
  Intelligence 101, 1998. A "reference frame the agent must infer" is
  structurally a belief state over a latent variable — this is the
  formal apparatus for exactly that problem
  (`people.csail.mit.edu/lpk/papers/aij98-pomdp.pdf`).
- Goal-conditioned value functions: Schaul, Horgan, Gregor, Silver,
  *Universal Value Function Approximators*, ICML 2015. `V(s, g)` /
  `Q(s, a, g)` is the RL-native version of "evaluate relative to a
  parameter," with a learned (not hand-specified) conditioning object.
- Multi-task RL, explicit vs. implicit context representation — **read
  this one closely, it is the paper closest to falsifying or supporting
  our exact claim**: *Multi-Task Reinforcement Learning with Context-based
  Representations* (arXiv:2102.06177) contrasts simplistic task-ID
  conditioning against richer, learned task embeddings derived from
  reward/dynamics differences across tasks, and against MATE-style
  encoder-decoder task embeddings learned from trajectories (arXiv
  2207.02249). This is the existing literature's version of "implicit
  flat context" vs. "structured context representation" — precisely
  Experiment 1's baseline-vs-treatment axis. Read before finalizing
  Experiment 1's design: if this literature already shows structured
  task representations reliably beat flat task-ID conditioning under
  shift, our contribution narrows further, to whether an *explicit,
  human-legible, compositional `ReferenceFrame` object* (vs. a learned
  opaque embedding) adds anything beyond what a learned embedding gets
  for free — interpretability and inspectability, not necessarily reward.
  That's a different and more modest claim than "explicit frames improve
  reward," and should be stated as such if the reward gap turns out to be
  small.
- Meta-RL: Duan, Schulman, Chen, Bartlett, Sutskever, Abbeel, *RL²: Fast
  Reinforcement Learning via Slow Reinforcement Learning*, arXiv:1611.02779,
  2016; Finn, Abbeel, Levine, *Model-Agnostic Meta-Learning for Fast
  Adaptation of Deep Networks*, ICML 2017. Both adapt to a hidden task
  variable from few samples — "reference frame changes at unannounced
  change-points" is a special case of the meta-RL adaptation problem,
  specifically the non-stationary / changepoint variant.

**Classification: established theory for the mechanism (conditioning on a
latent context variable, inferring it from feedback, adapting quickly).
Where TransIntelligence's `ReferenceFrame` differs is only in being an
explicit, structured, human-legible dataclass (`baseline`, `observer`,
`objective`, `time_window`, `domain`, `scale`, `assumptions`,
`constraints`) rather than a learned embedding — this is an engineering
and interpretability choice, not a new mechanism. Do not claim mechanism
novelty. The open question worth an experiment is narrower: does the
interpretable/explicit representation cost or gain anything in decision
quality relative to the learned-embedding versions already in the
literature above, and does it gain anything in calibration/auditability
that a black-box embedding can't offer even if reward is tied.**

**Follow-up (Experiment 1, `docs/experiments/exp01_frame_conditioning/RESULTS.md`)**:
building the more literal version of this literature's mechanism — a
trained recurrent context encoder, not a discrete mixture — hit a second,
independent layer of established theory. Bengio, Simard, Frasconi,
*Learning Long-Term Dependencies with Gradient Descent is Difficult*,
IEEE Transactions on Neural Networks, 1994, identifies exactly the failure
mode observed (vanishing gradients in a vanilla `tanh` RNN, insensitive to
truncated-BPTT length); Williams, Peng, *An Efficient Gradient-Based
Algorithm for On-Line Training of Recurrent Network Trajectories*, Neural
Computation 2(4), 1990, is the truncated-BPTT training method used;
Hochreiter, Schmidhuber, *Long Short-Term Memory*, Neural Computation
9(8), 1997, is the established fix (gating) not implemented here.
**Classification: established theory for all three — the failure, the
training method, and the fix are all textbook, not novel findings. What's
worth reporting is that a plausible opaque-embedding *implementation
choice* (vanilla RNN vs. discrete mixture) determines success or failure
independently of the frame-conditioning question itself — treat "learned
embedding" as a family with very different members, not a single
baseline.**

## 3. Causal framing of "evaluate relative to a frame"

- Pearl, *Causality: Models, Reasoning, and Inference*, 2nd ed.,
  Cambridge University Press, 2009. `evaluate(x, R)` where `R` fixes a
  baseline/comparator is formally a conditional query, and comparing
  `evaluate(x, R1)` vs `evaluate(x, R2)` is close to comparing outcomes
  under different conditioning sets — do-calculus is the established
  machinery for when such comparisons support causal (not merely
  associational) claims. Current `BaselineRelativeReasoner` (`raw - baseline`,
  sign-flipped by direction) makes no causal claim and shouldn't be
  described as one — it's descriptive/associational scoring dressed as
  "reasoning." Flag this explicitly in any external write-up.

**Classification: established theory. This was true when written, and
`transintelligence/reasoning/causal/` has since been built (Phase 5,
`docs/research-agenda.md` #7c, grounded in §3a below) — `evaluate(x, R)`
itself still makes no causal claim, but the codebase now has a real
place to make one, separate from the relative-reasoning module.**

## 3a. The backdoor criterion and confounding bias (Experiment 6, Phase 5)

- Pearl, *Causal Diagrams for Empirical Research*, Biometrika 82(4),
  669-688, 1995. The backdoor criterion — the graphical test
  `CausalGraph.satisfies_backdoor_criterion()` implements, for whether a
  candidate covariate set is valid for estimating a causal effect from
  observational data.
- Verma, Pearl, *Causal Networks: Semantics and Expressiveness*, UAI
  1988, pp. 69-78. D-separation — the soundness result the backdoor
  criterion's path-blocking logic rests on, implemented directly via path
  enumeration in `CausalGraph.d_separated()` rather than the equivalent
  moralized-ancestral-graph reformulation, verified against the three
  canonical structures (chain, fork, collider) before being trusted for
  anything built on top of it.
- Simpson, *The Interpretation of Interaction in Contingency Tables*,
  JRSS-B 13(2), 238-241, 1951. The classic confounding-reversal
  phenomenon — worth noting precisely rather than overclaiming Simpson
  "discovered" it: the underlying effect was already pointed out by
  Pearson (1899) and Yule (1903); Simpson's paper is the one that gave
  the phenomenon its common name. Experiment 6's synthetic SCM
  constructs a version of this directly (naive regression finds a
  spurious 0.881 effect where the truth is 0.0) rather than citing the
  paradox from authority.
- Rubin, *Estimating Causal Effects of Treatments in Randomized and
  Nonrandomized Studies*, Journal of Educational Psychology, 1974. The
  potential-outcomes framework — an alternative formalization of the same
  causal-effect question Experiment 6 answers via graphs (Pearl's
  structural/graphical tradition) instead of counterfactual random
  variables (Rubin's tradition). Experiment 7 later implements per-unit
  counterfactual queries via Pearl's tradition specifically (see §3b) —
  Rubin's potential-outcomes framework remains the road not taken, not
  computationally compared against.

**Classification: established theory in full for all four citations —
the backdoor criterion, d-separation, Simpson's paradox, and potential
outcomes are all textbook, decades old, and none of the three
implemented mechanisms (d-separation, the backdoor criterion, linear
adjustment via OLS) are novel. What Experiment 6 contributes is a
directly-constructed demonstration (a collider adjustment that makes an
already-correct model *worse*, not just "less good") rather than citing
the standard confounding warning from authority — the same discipline
Experiment 5's DTW comparison used.**

## 3b. Per-unit counterfactuals: abduction-action-prediction (Experiment 7, Phase 5)

- Pearl, Glymour, Jewell, *Causal Inference in Statistics: A Primer*,
  Wiley, 2016. The standard pedagogical source for the three-step
  abduction-action-prediction procedure `StructuralCausalModel.counterfactual()`
  implements: infer a unit's exogenous noise from its observed values,
  fix the intervened variables, recompute everything else using that
  unit's own inferred noise.
- Balke, Pearl, *Counterfactual Probabilities: Computational Methods,
  Bounds and Applications*, UAI 1994, pp. 46-54. The earlier computational
  treatment the 2016 primer's pedagogical version builds on — relevant
  here mainly for the general (non-linear, non-deterministic) case this
  implementation deliberately does not attempt: `abduct()`'s closed-form
  residual only works because the structural equations are linear with
  additive noise; Balke & Pearl's methods (and the general theory) handle
  cases where abduction requires solving for a distribution over exogenous
  variables rather than reading off an exact value.

**Classification: established theory in full — the three-step procedure
and its computational grounding are both textbook, not novel.
`StructuralCausalModel`'s contribution is the same kind of engineering
simplification as `CUSUMTemporalReasoner` and `reasoning/causal/`'s OLS
adjustment: the smallest version of the established mechanism (closed-form
linear abduction) that could produce a falsifiable result, explicitly not
a claim to handle the general nonlinear/non-additive-noise case Balke &
Pearl's methods were built for.**

**Update (Experiment 10, Phase 5): the "not nonlinear" limitation above
turned out to be narrower than it looked.** `StructuralEquation` now
accepts an arbitrary `nonlinear_fn` in place of its linear coefficient
form, and `abduct()`/`counterfactual()` needed *zero changes* to support
it — because Pearl's abduction step only ever requires the noise to be
*additive*, never that the structural function be linear, the same
closed-form residual is exact for any function, verified against a
hand-computed quadratic case before being trusted. What remains
genuinely linear-only is `reasoning/causal/`'s OLS-based *effect
estimation*, which experiment 10 shows is substantially biased under
true nonlinearity (absolute gap 0.5799 against an exact nonlinear
reference, vs. 0.0121 in a linear control) — and, more sharply, that a
single linear coefficient cannot represent a *heterogeneous* effect at
all (the true per-unit shift effect ranged from -1.5 to +3.3 and even
flipped sign across the tested reference points, while the linear model
predicted one constant number everywhere). Non-additive noise remains the
one part of Balke & Pearl's general theory still not attempted here.**

## 3c. Causal discovery: constraint-based structure recovery (Experiment 8, Phase 5)

- Spirtes, Glymour, *An Algorithm for Fast Recovery of Sparse Causal
  Graphs*, Social Science Computer Review 9(1), 62-72, 1991. The PC
  algorithm — the mechanism `discover_skeleton`/`orient_colliders`
  implement: recover an undirected skeleton by removing an edge x-y as
  soon as some conditioning set (drawn from x's and y's neighbors) makes
  them independent, then orient unshielded colliders (v-structures) from
  which conditioning sets did and didn't include the middle node.
- Fisher, *On the "Probable Error" of a Coefficient of Correlation
  Deduced from a Small Sample*, Metron 1, 1921. The z-transform of a
  (partial) correlation coefficient used as the conditional-independence
  test (`fisher_z_independence_test`) — the standard test for
  linear-Gaussian PC-algorithm implementations, chosen over a
  permutation or kernel-based independence test for the same
  "lightweight dependencies, no numpy/scipy" reason `ordinary_least_squares`
  exists in this module already; `partial_correlation` computes the
  partial correlation itself by reusing that same OLS routine for
  residualization rather than a separate covariance-matrix inversion.
- Meek, *Causal Inference and Causal Explanation with Background
  Knowledge*, UAI 1995, pp. 403-410. Meek's four orientation rules can
  direct additional edges beyond direct v-structure detection, using
  acyclicity and no-new-collider constraints to propagate orientation
  information through the graph. **Three of the four (R1-R3) are now
  implemented, as `apply_meek_rules`** (Phase 5 follow-up to experiment
  8, `experiments/exp08_causal_discovery/RESULTS.md` §4) — the fourth,
  R4, is not, and provably could never fire in this pipeline: R4 only
  orients edges by propagating externally-supplied *background
  knowledge*, and this implementation has no mechanism to inject any.
  Perkovic, Textor, Kalisch, Maathuis, *Interpreting and Using CPDAGs
  with Background Knowledge*, UAI 2017, confirms R1-R3 alone are
  established to be complete for recovering the CPDAG (the
  maximally-oriented representation of a DAG's whole Markov equivalence
  class) in exactly the no-background-knowledge setting this module
  operates in — R4 only matters once background knowledge is added.

**Classification: established theory in full for all four citations —
the PC algorithm, Fisher's z-test, Meek's orientation rules, and the
completeness result for the no-background-knowledge case are all
textbook or a well-established paper result, not novel. What experiment 8
contributes is a set of directly-constructed demonstrations that the
mechanism's theoretically-predicted boundaries are real, not just
footnotes: a **shielded** collider (built to mirror experiment 6's own
graph shape) is correctly never oriented despite genuinely being a
collider, a **dedicated unshielded** collider is oriented correctly and
reliably, Meek's rules extend a hard 0.500 recall ceiling (collider
orientation alone) to 1.000 on a fully-identifiable graph while never
producing a wrong orientation given a correct skeleton, and a
collider-free chain (Markov-equivalent to a fork and a reverse chain)
gets zero edges oriented at any sample size, confirming the rules
propagate from real evidence rather than inventing orientations — the
same "don't just show the positive case, show the boundary condition
too" discipline experiment 6's collider adjustment used.**

## 3d. Instrumental variables and front-door adjustment: identification under an unobserved confounder (Experiment 9, Phase 5)

- Wright, P. G., *The Tariff on Animal and Vegetable Oils*, Macmillan,
  1928, Appendix B. The origin of the instrumental-variables estimator
  `two_stage_least_squares` implements as two-stage least squares —
  authorship of the technical appendix is historically disputed between
  Philip Wright and his son Sewall Wright.
- Pearl, *Causal Diagrams for Empirical Research*, Biometrika 82(4),
  1995. The same paper already cited in §3a for the backdoor criterion
  also introduces the front-door criterion `front_door_adjustment`
  implements: identification via a fully-mediating observed variable,
  for exactly the case where a valid backdoor adjustment set doesn't
  exist because the confounder isn't observed.
- Wright, S., *The Method of Path Coefficients*, Annals of Mathematical
  Statistics 5(3), 161-215, 1934. The classical result that grounds this
  module's *linear* implementation of front-door adjustment: for chained
  linear structural equations, the total effect along a mediating path
  is the product of the path's individual coefficients, not their sum or
  either one in isolation. `front_door_adjustment` computes exactly that
  product (treatment→mediator coefficient times mediator→outcome
  coefficient, the latter estimated adjusting for treatment).
- Bound, Jaeger, Baker, *Problems with Instrumental Variables Estimation
  When the Correlation between the Instruments and the Endogenous
  Explanatory Variable is Weak*, Journal of the American Statistical
  Association 90(430), 443-450, 1995. Predicts that IV estimates
  degrade — becoming as biased as, or in this implementation's
  iid-error setup, considerably *more* unstable than, naive OLS — as
  instrument strength approaches zero. Experiment 9's instrument-strength
  sweep tests this prediction directly rather than only showing a
  favorable instrument-strength case.

**Classification: established theory in full for all four citations —
2SLS, the front-door criterion, path analysis, and the weak-instrument
problem are all textbook or foundational-paper results, none of the two
implemented mechanisms are novel. What experiment 9 contributes is two
directly-constructed demonstrations, following the same discipline as
experiment 6's collider-adjustment condition: 2SLS's positive result
(bias reduced roughly 30x relative to naive OLS at strength 0.9) is
paired with a demonstration that it doesn't degrade gracefully — below a
strength threshold it becomes a wildly unstable, worse-than-naive
estimator, not a smoothly weakening one — and front-door adjustment's
positive result (near-exact recovery, bias 0.0094) is paired with a
demonstration that violating its specific structural assumption (the
confounder reaching the mediator) makes it nearly as biased as doing
nothing at all.**

## 3e. World models: learned dynamics for planning (Experiment 11, Phase 6)

- Sutton, *Integrated Architectures for Learning, Planning, and Reacting
  Based on Approximating Dynamic Programming*, ICML 1990; Sutton, *Dyna,
  an Integrated Architecture for Learning, Planning, and Reacting*,
  SIGART Bulletin, 1991. The Dyna architecture — the established
  mechanism `LinearDynamicsModel`
  (`transintelligence/world_models/model.py`) implements the smallest
  version of: plan by simulating candidate actions through a learned
  model of the environment's transition dynamics, rather than only
  acting on cached historical value.
- Ha, Schmidhuber, *World Models*, arXiv:1803.10122, 2018. The paper that
  popularized the term "world model" for this class of methods, using a
  much heavier generative/recurrent neural architecture than anything
  implemented here — cited for the term and the "learn M, then act
  inside it" framing the master context's own Phase 6 description
  (`docs/master-context.md` §13) is written in, not as the mechanism
  this experiment implements.
- Sutton, Barto, *Reinforcement Learning: An Introduction*, 2nd ed., MIT
  Press, 2018 — standard reference for linear function approximation of
  a value function, the mechanism `model_free_linear_q`
  (`experiments/exp11_world_model_planning/run.py`) uses as the fair,
  identically-tooled comparison condition.

**Classification: established theory in full for all three citations —
Dyna-style model-based planning, the "World Models" framing, and linear
value-function approximation are all textbook or foundational-paper
results, none of the mechanisms implemented are novel. What experiment
11 contributes is a directly-constructed demonstration that isolates the
*specific* claim (decomposing dynamics-modeling from reward computation)
from the trivial claim (using state helps at all): a state-aware
model-free baseline, given identical state access and the identical
linear-regression tool, is shown to be quantitatively worse than the
world-model agent for a mechanistically specific reason — the true
reward is a non-monotonic (quadratic) function of state that a linear
value fit cannot represent regardless of data, while the true
*transition* is linear and therefore exactly learnable, the same
linear-cannot-represent-curvature lesson experiment 10 established for
effect estimation, now shown in a planning setting instead. A follow-up
confirmed the mechanism directly: giving the model-free baseline a
quadratic (correctly-specified) feature set closed almost the entire
gap, ruling out an unaccounted-for confound as the real explanation.**

## 3f. Multi-step planning: receding-horizon control (Experiment 12, Phase 6, continued)

- Richalet, Rault, Testud, Papon, *Model Predictive Heuristic Control:
  Applications to Industrial Processes*, Automatica 14(5), 429-445,
  1978. The founding paper for what's now called Model Predictive
  Control / receding-horizon control — the established mechanism
  `RecedingHorizonPlanner`
  (`transintelligence/planning/model.py`, generalized from experiment
  12's original environment-specific `choose_action` function after the
  experiment shipped) implements the smallest version of: simulate
  several steps ahead using a learned model, execute only the first
  action, then replan from the newly observed state at every step.
  Experiment 11's Dyna citation (Sutton 1990/1991, above) already covers
  "plan via simulated rollouts through a learned model" in the
  single-step case; this is the direct multi-step generalization of the
  same idea, from a different but closely related literature.

**Classification: established theory in full — receding-horizon control
is foundational, decades-old control theory, not a novel mechanism.
What experiment 12 contributes is a 2×2 factorial design (planning
horizon × model source) that isolates the specific claim (does horizon
matter) from a confound (does model quality happen to differ between
conditions): a multi-step planner beats greedy 1-step lookahead by
almost exactly the same margin whether using a learned or an oracle
dynamics model, confirming the advantage is genuinely about horizon.
The effect is real, consistent across 14 of 15 seeds, and reported at
its actual (modest, ~17% relative) size rather than overstated — smaller
than experiment 11's dynamics-vs-reward-modeling result, for a
structural reason stated plainly in the results, not glossed over.**

## 3g. Combining world models with regime-change detection (Experiment 13, Phase 6, continued)

This is a synthesis experiment, not a new-mechanism one — it combines
two already-built, already-verified kernel primitives rather than
introducing anything new:

- `CUSUMTemporalReasoner` (Page 1954; already cited and independently
  verified in experiment 5, §9a below) monitors a residual stream for a
  mean-level shift.
- `LinearDynamicsModel` (already cited and independently verified in
  experiment 11, §3e above) is the world model being monitored.

No new citation is needed for the mechanism itself; the only question is
empirical — does composing two independently-correct primitives actually
work as expected when wired together, and under what conditions.

**Classification: established theory in full for both underlying
mechanisms — nothing here is a novel algorithm. What experiment 13
contributes is a directly-constructed test of composition, and an honest
report of where the naive hypothesis (adaptation always helps) failed on
its first, simpler run: a mild regime shift that doesn't change which
action is locally best produced no detectable adaptation benefit, and
adaptation's real cost (discarding a large body of converged prior data
for a small, noisy post-shift sample) briefly outweighed its benefit. A
severe shift (reversing the actuator's direction entirely) is where
detection-triggered adaptation earned its complexity, beating both a
do-nothing baseline and a naive always-use-recent-data heuristic — the
same "isolate the real claim from the trivial one" discipline experiment
11's model-free baseline and experiment 9's weak-instrument sweep both
used, now applied to a synthesis of two Phase 4/6 primitives instead of
a single new mechanism.**

## 3h. Beam search for scalable receding-horizon planning (Experiment 14, Phase 6, continued)

- Lowerre, *The Harpy Speech Recognition System*, PhD thesis, Carnegie
  Mellon University, 1976. The origin of beam search — the established
  mechanism `RecedingHorizonPlanner`'s `beam_width` parameter
  (`transintelligence/planning/model.py`) implements: keep only the
  `beam_width` best-scoring partial candidates at each expansion step
  instead of exploring every possible sequence.

**Classification: established theory in full — beam search itself is
textbook, not novel. What experiment 14 contributes is not the expected
"beam search approximates exhaustive search's quality more cheaply"
result — investigating a result that looked wrong (beam search appeared
to *beat* exhaustive search, impossible by construction) instead of
discarding it surfaced something sharper: in a receding-horizon control
setting where only the first planned action is ever executed before
replanning, exhaustive search's terminal-only scoring is vulnerable to a
genuine, quantified pathology (2/15 states converged to target at
depth 3, vs. 13-15/15 for beam search) — a real finding about the
mismatch between what exhaustive search optimizes and what receding-
horizon control actually needs, not merely a speed/quality tradeoff.**

## 3i. Nonlinear world-model dynamics (Experiment 15, Phase 6, continued)

- Pearl, Glymour, Jewell, *Causal Inference in Statistics: A Primer*,
  Wiley, 2016. Already cited in §3b for the claim that abduction only
  needs additive noise, not linearity — the same underlying principle
  (a nonlinear functional form is fine as long as it's correctly
  specified and fit) motivates `NonlinearWorldModelAgent`'s quadratic
  feature here, no new citation needed for the mechanism itself (plain
  polynomial-feature OLS, already used in experiment 11's quadratic
  follow-up for a value function rather than a transition model).

**Classification: established theory — polynomial-feature regression is
textbook, not novel. What experiment 15 contributes is the same
"linear-cannot-represent-a-nonlinearity" demonstration experiments 10
and 11 already made, now for a third distinct object (world-model
dynamics prediction, after causal effect estimation and single-step
value estimation) — and a directly-checked boundary condition: a weak
nonlinearity that doesn't change which discrete action ranks best
produces no measurable gap, mirroring experiment 13's mild-severity
finding exactly, before a stronger version of the same nonlinearity
produced a clean, ~193x regret gap.**

## 3j. Combining regime-change detection with multi-step planning (Experiment 16, Phase 6, continued)

A second synthesis experiment (after experiment 13) — no new mechanism
is implemented, only `CUSUMTemporalReasoner` (Page 1954, already cited
for experiment 5), the dynamics-fitting approach from experiment 12, and
`RecedingHorizonPlanner` (Richalet et al. 1978 / Lowerre 1976, already
cited for experiments 12/14), combined.

**Classification: established theory in full for every underlying
mechanism — nothing here is novel. What experiment 16 contributes is two
findings only visible by testing the actual combination.** First, an
independent replication: exhaustive multi-step search's oscillation
pathology (experiment 14) recurs in a new environment built for a
different purpose, confirming it as a general property of exhaustive
receding-horizon search under delayed dynamics, not an artifact of
experiment 14's specific setup. Second, a genuinely new finding: even
after fixing that pathology with beam search, combining a *learned*
(not oracle) dynamics model with multi-step lookahead has a real,
persistent cost — traced across the entire post-shift window, not a
shrinking startup transient — that the same model combined with
single-step lookahead does not pay. A "reduced exploration" explanation
was checked directly and refuted; the best-supported remaining
explanation (compounding estimation error across chained predictions) is
stated as a hypothesis, not confirmed by direct manipulation.

## 3k. `MonteCarloSimulator`: validating a Monte Carlo policy comparator against experiment 16's already-established finding (Experiment 17, Phase 6, closing)

- Sutton, ICML 1990 / SIGART Bulletin 1991 (already cited in §3e). The
  same Dyna framing generalized one step further: where
  `LinearDynamicsModel` (`Predictor`) simulates to predict a single next
  state and `RecedingHorizonPlanner` (`Planner`) simulates to search
  internally for the best action, `MonteCarloSimulator`
  (`transintelligence/simulation/model.py`) simulates to *compare* two
  already-chosen candidate policies under real stochasticity, the
  standalone capability the master context's own §13 framing describes
  ("S_t →A_1→ S_t+1 can be compared against S_t →A_2→ S'_t+1") but
  neither `Predictor` nor `Planner` provides directly.

**Classification: established theory in full — Monte Carlo rollout
comparison of candidate policies is textbook, not novel. What experiment
17 contributes is not a new claim but a validation**: rather than
trusting newly-written code on a fresh, unrelated environment, it asks
whether `MonteCarloSimulator`, using a completely different methodology
(short stochastic rollouts from arbitrary starting states) than
experiment 16 used (full multi-seed environment rollouts averaged over
400 episodes each), independently recovers experiment 16's already-known
ranking (`greedy_cusum_adapts` beats `mpc_beam_cusum_adapts` post-shift).
It does — 58/60 comparisons (96.7%) — with a built-in sensitivity control
(dropping `n_rollouts` from 200 to 5 reduces the win rate to 88.3%,
confirming sample size genuinely affects verdict reliability rather than
being cosmetic). This is the same "confirm a new mechanism against an
already-established result before trusting it further" pattern
experiment 5 used for experiment 4's fixed-threshold trigger, applied
here to a newly-built kernel primitive instead of a newly-run experiment.

## 3l. `CalibrationVerifier`: testing experiment 16's compounding-estimation-error hypothesis directly (Experiment 18, Phase 6 closing / meta-intelligence opening)

- Dawid, *The Well-Calibrated Bayesian*, Journal of the American
  Statistical Association 77(379), 605-613, 1982. The general
  definition `CalibrationVerifier`
  (`transintelligence/verification/model.py`) operationalizes: a
  well-calibrated forecaster's stated confidence level should equal the
  long-run frequency with which outcomes actually fall within it.
- Kupiec, *Techniques for Verifying the Accuracy of Risk Measurement
  Models*, The Journal of Derivatives 3(Winter), 73-84, 1995. The
  specific mechanism implemented: the unconditional-coverage /
  "proportion of failures" likelihood-ratio test, originally built to
  backtest whether a risk model's stated confidence intervals are
  honest — is the observed hit rate consistent with the claimed one?
  The test statistic is asymptotically chi-squared with 1 degree of
  freedom, whose CDF has the closed form `erf(sqrt(x/2))` (chi-squared(1)
  is the square of a standard normal), computable from `math.erf` alone
  — no dependency beyond the standard `math` module, matching this
  repo's lightweight-dependencies constraint.

**Classification: established theory in full for both citations — the
definition of calibration and the specific coverage test are both
textbook / foundational, not novel. What experiment 18 contributes is a
newly-built tool applied as a genuine hypothesis test, not a validation
(contrast experiment 17): experiment 16's central finding came with a
best-supported but explicitly unconfirmed explanation (chained
multi-step predictions compound a learned model's estimation error) —
`CalibrationVerifier`, applied to each agent's own one-step prediction
residuals against the environment's true noise floor, tests a specific,
falsifiable alternative directly: does the multi-step planner's own
self-steered training distribution degrade its dynamics model's
calibration relative to the single-step planner's? It does not — the
two agents' one-step models are statistically indistinguishable in
calibration, in both the pre- and post-shift windows, ruling out that
specific alternative mechanism and, by elimination, leaving experiment
16's original chaining hypothesis as the more plausible remaining
explanation, still not directly confirmed.**

## 3m. Extending regime-adaptation to a nonlinear dynamics model (Experiment 19, Phase 6, continued)

A third synthesis experiment (after experiments 13 and 16) — no new
mechanism implemented, only `CUSUMTemporalReasoner` (Page 1954, already
cited for experiment 5) and `NonlinearDynamicsModel` (generalized from
experiment 15's own fitting logic, itself grounded in Pearl, Glymour &
Jewell 2016's additive-noise principle, already cited in §3i), combined.

**Classification: established theory in full for every underlying
mechanism — nothing here is novel. What experiment 19 contributes is a
genuinely new, mechanistically-explained finding only visible by testing
the actual nonlinear+regime-shift combination.** A mild shift, an honest
null result for experiment 13's linear model, actively *hurts* here:
`oracle_adapts` performs worse than `never_adapts`, traced directly (via
instrumentation, not inference) to a cold-start cost `NonlinearDynamicsModel`'s
higher per-action data requirement (3 observations, not `LinearDynamicsModel`'s
2) creates for any hard-reset adaptation strategy. A severe shift still
favors adaptation overall, matching experiment 13's conclusion, but
CUSUM detection reliability itself measurably degrades (9/15 vs.
experiment 13's 15/15 seeds detected, ~5.4x slower mean latency) for a
reason checked directly and refuted (steady-state residual noise is
statistically identical between the linear and nonlinear settings) but
not otherwise identified — an honestly reported open question, not
papered over with a plausible-sounding but unverified explanation. A
simpler sliding-window heuristic, which never fully empties its training
data, ends up more practically robust than CUSUM-triggered detection
here, reversing experiment 13's own preference ordering.

## 3n. Directly confirming experiment 16's compounding-estimation-error hypothesis via a noise sweep (Experiment 20, Phase 6, continued)

No new citation — nothing new is implemented; `Agent`,
`DelayedRegimeShiftControlEnv`, and `run_episode` are reused exactly as
experiments 16-18 already verified, with the environment's own
observation noise (`NOISE_SIGMA`) swept as a parameter instead of held
fixed.

**Classification: a direct empirical test of experiment 16's own
previously-stated, unconfirmed mechanism — not new theory.** Experiment
16 proposed that multi-step lookahead's persistent post-shift
underperformance comes from chaining two predictions from the same
learned model, each carrying estimation error that never fully vanishes
*because the environment has real observation noise*. Experiment 18
tested a specific alternative explanation and ruled it out but never
manipulated this stated cause directly. Experiment 20 does: the
post-shift reward gap between single-step and multi-step planning scales
strongly with the environment's own noise level — negligible at zero
noise, ~17-21x larger at or above the original noise level used in
experiments 16-18 — with detection reliability shown to track closely
between the two agents at every noise level, ruling out differential
detection as an alternative explanation. This is the first direct,
positive confirmation of the mechanism in this program's Phase 6 arc,
closing a loop experiments 16 and 18 both left open.

## 4. Frame-dependence / frame-invariance detection (Experiment 2)

- Invariant Risk Minimization: Arjovsky, Bottou, Gulrajani, Lopez-Paz,
  *Invariant Risk Minimization*, arXiv:1907.02893, 2019. Learns
  representations whose optimal predictor is stable across environments —
  the mirror image of what Experiment 2 wants (detecting which
  *properties*, not representations, are frame-dependent vs. frame-stable).
  Worth noting IRM-v1 has been shown empirically unreliable at its own
  goal in follow-up work (*The Risks of Invariant Risk Minimization*,
  arXiv:2010.05761) — a caution against assuming any variance-based
  invariance detector, including ours, works out of the box; this is
  exactly why Experiment 2 asks for an ROC-AUC / noise-robustness check
  rather than assuming the current deterministic `sensitivity()` just works.
- Domain generalization surveys: Zhou, Liu, Qiao, Xiang, Loy, *Domain
  Generalization: A Survey*, IEEE TPAMI 2023 (arXiv 2021); Wang, Lan, Liu,
  Ouyang, Qin et al., *Generalizing to Unseen Domains: A Survey on Domain
  Generalization*, IEEE TKDE 2022. Background for how "does this property
  hold across contexts" is normally studied at scale; useful for framing
  Experiment 2's synthetic-data design so it isn't reinventing evaluation
  protocol from scratch.

**Classification: established theory (the detection *problem* — telling
invariant from context-dependent structure apart — is well studied).
TransIntelligence's `sensitivity()` as currently implemented
(`abs(evaluate(x,R1) - evaluate(x,R2))`, a raw point-difference with no
variance/noise model) is a naive baseline relative to this literature, not
a novel method. Experiment 2 should be read as "does the naive baseline
work at all," not as testing a new algorithm.**

## 5. Geometric reasoning over non-physical spaces

- Bronstein, Bruna, Cohen, Veličković, *Geometric Deep Learning: Grids,
  Groups, Graphs, Geodesics, and Gauges*, arXiv:2104.13478, 2021. The
  "geometry isn't just physical space, treat embedding/graph/latent spaces
  geometrically" framing in `docs/reasoning.md`-adjacent material is this
  paper's thesis, applied outside deep learning specifically. Cite this
  directly wherever the vision doc's §9 claim appears in any external
  write-up — it is not an original framing.

**Classification: established theory, directly and heavily prior art.
Nothing in the current `reasoning/geometric` stub goes beyond this.**

## 6. Analogical / structural cross-domain transfer (Experiment 3, and the "Trans" claim generally)

- Gentner, *Structure-Mapping: A Theoretical Framework for Analogy*,
  Cognitive Science 7(2), 1983. The formal distinction this project needs
  for Experiment 3's isomorphism control already exists here: **similarity**
  (shared attributes) vs. **analogy** (shared relational structure, few
  shared attributes) vs. mere relabeling (shares neither in a meaningful
  sense). Use Gentner's systematicity principle — a relation is more
  likely to transfer if it belongs to a mutually-interconnected system of
  relations, not in isolation — as the actual design constraint for
  building Experiment 3's two "isomorphic" domains, rather than inventing
  an ad hoc notion of isomorphism.
- Lake, Baroni, *Generalization without Systematicity: On the
  Compositional Skills of Sequence-to-Sequence Recurrent Networks*, ICML
  2018 (SCAN benchmark), and the follow-up *Human-like Systematic
  Generalization through a Meta-Learning Neural Network*, Nature 2023.
  Directly relevant negative and positive results: flat neural networks
  fail at compositional generalization by default (2018 result), but
  meta-learning specifically optimized for compositional tasks can induce
  it (2023 result). This is strong precedent for treating "does explicit
  compositional structure transfer better than a flat model" as an
  empirical question with a known-nontrivial answer, not something to
  assume in either direction.

**Classification: established theory for the distinctions needed
(analogy vs. similarity vs. relabeling; compositional generalization is
possible but not automatic). Whether *TransIntelligence's specific*
representation (typed `Entity`/`Relationship`/`ReferenceFrame` primitives)
transfers reasoning strategies across domains is a hypothesis — genuinely
untested, and the least mature of the three experiments as already flagged
in `research-agenda.md`.**

## 7. Self-modeling, recursive intelligence, "strange loops"

- Hofstadter, *Gödel, Escher, Bach: An Eternal Golden Braid*, 1979;
  *I Am a Strange Loop*, 2007. Source of the term and the core intuition
  (a system whose model of itself feeds back into its own behavior). This
  is a philosophical/expository framing, not a computational mechanism or
  falsifiable claim by itself — treat as motivation only, per
  `research-agenda.md` §2's decision to not test this yet.
- Friston, *The Free-Energy Principle: A Unified Brain Theory?*, Nature
  Reviews Neuroscience, 2010. The nearest thing to a formal, mechanistic
  account of self-referential world-modeling (minimizing prediction error
  about an environment that includes the modeler's own actions). If
  §15–16 of the master context is ever formalized into an experiment, this
  is the literature to formalize it against, not Hofstadter directly —
  Hofstadter motivates the question, Friston's program is closer to a
  testable mechanism.
- Rabinowitz, Perbet, Song, Zhang, Eslami, Botvinick, *Machine Theory of
  Mind*, ICML 2018 (arXiv:1802.07740). ToMnet learns to model other
  agents' latent characteristics/mental states from behavior alone, and
  passes false-belief tests. Relevant precedent for "self-model" and
  "model of another agent's model" as things that have actually been built
  and evaluated computationally (on other agents, not self) — a much more
  concrete starting point than strange-loop language if Phase 9 is ever
  attempted.

**Classification: strange loops / recursive self-modeling as described in
the master context §15–16 is currently pure motivation, not a hypothesis —
there is no proposed measurement. Before it can be labeled even
"hypothesis," it needs a Rabinowitz-style operationalization: what
observable agent behavior would differ if self-modeling were present vs.
absent? Do not build Phase 8-9 code before that operationalization exists.**

## 8. Frame semantics (linguistics) — motivating analogy only

- Fillmore, *Frame Semantics*, 1982 (and the earlier *The Case for Case*,
  1968, which case-grammar work evolved from). Source of the word "frame"
  in the intuitive sense used in `docs/reference-frames.md` — a word's
  meaning evokes a structured background scene. This is a linguistic
  theory about lexical semantics, not a computational reasoning mechanism,
  and has no direct algorithmic content to import. Useful only as the
  etymology/intuition pump for the term "reference frame" — do not cite it
  as methodological support for anything in `reasoning/relative/`.

**Classification: not applicable to classify as established/novel in the
technical sense — it's a naming inspiration, not prior art for a
mechanism. Listed here only so nobody later mistakes vocabulary overlap
for a substantive connection.**

## 8a. Frame discovery: change-point detection, nonparametric mixtures, open-set recognition (Experiment 4)

- Change-point detection: Adams, MacKay, *Bayesian Online Changepoint
  Detection*, arXiv:0710.3742, 2007. Maintains an exact posterior over
  "time since the last changepoint" via online message-passing — the
  principled version of "detect that recent fit quality has dropped
  because the underlying regime changed."
- Growing a hypothesis space online: Ferguson, *A Bayesian Analysis of
  Some Nonparametric Problems*, Annals of Statistics, 1973 (the Dirichlet
  process); Neal, *Markov Chain Sampling Methods for Dirichlet Process
  Mixture Models*, Journal of Computational and Graphical Statistics,
  2000 (practical inference, including the Chinese Restaurant Process
  view). These let a mixture model's number of components grow
  nonparametrically as data demands it, rather than being fixed in
  advance — the established version of "add a new candidate frame when
  the known ones don't fit."
- Recognizing inputs that don't belong to any known class: open-set
  recognition / novelty detection, surveyed in *Managing the Unknown: A
  Survey on Open Set Recognition and Tangential Areas*, arXiv:2312.08785,
  and *A Unified Survey on Anomaly, Novelty, Open-Set, and
  Out-of-Distribution Detection*, arXiv:2110.14051.

**Classification: established theory for all three pieces (changepoint
detection, nonparametric mixture growth, open-set recognition). Experiment
4's planned mechanism (a rolling reward-rate trigger plus a one-shot grid
search for a new `(baseline, direction)` pair) is a deliberately crude,
narrow heuristic version of all three — not an implementation of any of
them. State this plainly in any write-up: the experiment tests whether the
*simplest possible* version of "detect novelty, then discover a new
hypothesis" recovers anything, not whether the established machinery
works (that's already known to work, elsewhere, at far more engineering
cost than this repo's "lightweight dependencies" constraint allows for
right now). If the crude version fails outright, these three citations are
the concrete upgrade path, not a vague gesture at "more sophisticated
methods."**

## 9. Summary table

| Area | Nearest citation(s) | Classification |
|---|---|---|
| Reference frame as conditioning context | Li et al. 2010; Kaelbling et al. 1998; Schaul et al. 2015; arXiv:2102.06177; Duan et al. 2016; Finn et al. 2017 | Established theory — mechanism is old, explicit/legible representation is an engineering choice |
| RNN encoder-decoder condition-B attempt (vanishing gradients) | Bengio, Simard & Frasconi 1994; Williams & Peng 1990; Hochreiter & Schmidhuber 1997 | Established theory in full — the failure mode, the training method, and the fix (unimplemented) are all textbook |
| Causal framing of `evaluate(x,R)` | Pearl 2009 | Established theory; `evaluate()` itself still makes no causal claim |
| Backdoor criterion / confounding bias (Experiment 6, Phase 5) | Pearl 1995; Verma & Pearl 1988; Simpson 1951; Rubin 1974 | Established theory in full — implemented and empirically verified, not novel |
| Per-unit counterfactuals (Experiment 7, Phase 5) | Pearl, Glymour & Jewell 2016; Balke & Pearl 1994 | Established theory in full — closed-form abduction, now confirmed exact for nonlinear (not just linear) additive-noise structural equations too (Experiment 10) |
| Causal discovery (Experiment 8, Phase 5) | Spirtes & Glymour 1991; Fisher 1921; Meek 1995 (R1-R3); Perkovic et al. 2017 | Established theory in full — skeleton, collider orientation, and Meek's R1-R3 propagation rules implemented; R4 provably inapplicable without background knowledge |
| Instrumental variables & front-door adjustment (Experiment 9, Phase 5) | Wright (P.) 1928; Pearl 1995; Wright (S.) 1934; Bound, Jaeger & Baker 1995 | Established theory in full — both mechanisms implemented and empirically verified, including their theory-predicted failure modes |
| Nonlinear structural equations (Experiment 10, Phase 5) | Pearl, Glymour & Jewell 2016 (abduction only needs additive noise, not linearity) | Established theory — abduction confirmed exact for a nonlinear case; linear OLS-based effect estimation confirmed biased under nonlinearity by construction, not a new finding about OLS itself |
| World models for planning (Experiment 11, Phase 6) | Sutton 1990/1991 (Dyna); Ha & Schmidhuber 2018; Sutton & Barto 2018 | Established theory in full — Dyna-style planning and linear value-function approximation are both textbook; the specific comparison isolating dynamics-modeling from reward-fitting is a directly-constructed demonstration, not a new algorithm |
| Multi-step planning (Experiment 12, Phase 6) | Richalet, Rault, Testud & Papon 1978 (MPC/receding-horizon control) | Established theory in full — the 2x2 design isolating planning-horizon from model-quality is a directly-constructed demonstration, not a new algorithm |
| World models + regime-change detection (Experiment 13, Phase 6) | Page 1954 (CUSUM, already cited for experiment 5); no new citation — a synthesis of two already-verified primitives | Established theory in full for both components — the empirical composition test, and the severity-dependent finding it produced, are the contribution, not any mechanism |
| Beam search for planning (Experiment 14, Phase 6) | Lowerre 1976 (beam search) | Established theory in full — the mechanism is textbook; the finding that it's more robust to a receding-horizon oscillation pathology, not just cheaper, is the contribution |
| Nonlinear world-model dynamics (Experiment 15, Phase 6) | Pearl, Glymour & Jewell 2016 (already cited, §3b) | Established theory — polynomial-feature OLS is textbook; the boundary-condition finding (weak nonlinearity shows no gap, strong does) mirroring experiment 13 is the contribution |
| Regime detection + multi-step planning (Experiment 16, Phase 6) | Page 1954; Richalet et al. 1978 / Lowerre 1976 (all already cited) — no new citation, a synthesis | Established theory in full for every component — an independent replication of experiment 14's oscillation pathology, plus a new, only-visible-in-combination finding (learned multi-step planning's persistent cost), are the contribution |
| `MonteCarloSimulator` validation (Experiment 17, Phase 6, closing) | Sutton 1990/1991 (Dyna, already cited, §3e) | Established theory in full — Monte Carlo rollout comparison is textbook; the contribution is a validation, not a new claim: independently recovering experiment 16's already-known ranking via a different methodology, with a built-in rollout-count sensitivity control |
| `CalibrationVerifier` / testing experiment 16's compounding-error hypothesis (Experiment 18, Phase 6 closing) | Dawid 1982 (calibration); Kupiec 1995 (unconditional-coverage LR test) | Established theory in full — a new hypothesis test, not a validation: rules out a self-steered-training-distribution alternative explanation for experiment 16's persistent gap, leaving the original chaining hypothesis the more plausible remaining one, still unconfirmed |
| Nonlinear dynamics + regime-adaptation (Experiment 19, Phase 6) | Page 1954; Pearl, Glymour & Jewell 2016 (both already cited) — no new citation, a synthesis | Established theory in full for every component — the contribution is a mechanistically-explained negative-under-mild-shift finding (a nonlinear model's cold-start cost) and a measured, only-partly-explained detection-reliability degradation, neither predictable from experiments 13 or 15 alone |
| Noise sweep confirming experiment 16's compounding hypothesis (Experiment 20, Phase 6) | No new citation — a direct manipulation of experiment 16's own already-stated mechanism | Not new theory — the contribution is the first direct, positive confirmation of experiment 16's central hypothesis (the persistent multi-step-planning gap scales ~17-21x with environment noise), with detection reliability tracked as a confound control |
| Frame-dependence detection | Arjovsky et al. 2019; arXiv:2010.05761; Zhou et al. 2023; Wang et al. 2022 | Established theory; current `sensitivity()` is a naive baseline against it |
| Geometric reasoning over non-physical spaces | Bronstein et al. 2021 | Established theory, directly prior art |
| Cross-domain structural transfer | Gentner 1983; Lake & Baroni 2018/2023 | Established distinctions; TransIntelligence's specific transfer claim is a hypothesis |
| Frame discovery (Experiment 4) | Adams & MacKay 2007; Ferguson 1973; Neal 2000; arXiv:2312.08785; arXiv:2110.14051 | Established theory in full form; the planned mechanism is a deliberately crude heuristic version, not an implementation of any of these |
| Strange loops / recursive self-model | Hofstadter 1979/2007; Friston 2010; Rabinowitz et al. 2018 | Motivation only, not yet a hypothesis — needs operationalization |
| "Reference frame" terminology | Fillmore 1982 | Naming inspiration, not a mechanism |
| Regime-change detection (Experiment 5, Phase 4) | Page 1954; Hamilton 1989; Rabiner 1989; Sakoe & Chiba 1978; Crosier 1988 | Established theory in full — CUSUM (and its multivariate/DTW extensions) is textbook, chosen for simplicity over HMM/regime-switching alternatives, not because they don't apply |

## 9a. Temporal reasoning: change/regime detection, temporal comparison (Phase 4)

`State`/`StateHistory` (Phase 1) already provide `state_at`/`trajectory` --
point-in-time lookup and range queries. What the master context's Phase 4
list actually asks for beyond that is *derived* temporal reasoning:
detecting *when* something changed, and segmenting a history into
regimes. `transintelligence/reasoning/temporal/` was, until this phase, a
single-line docstring stub (`transintelligence/reasoning/temporal/__init__.py`)
— this is the first content in it.

- Change-point detection: Page, *Continuous Inspection Schemes*,
  Biometrika 41(1-2), 1954. The CUSUM control chart — the mechanism
  `CUSUMTemporalReasoner` implements. Chosen over the fuller Bayesian
  treatment already cited for experiment 4 (Adams & MacKay 2007) for the
  same reason `BaselineRelativeReasoner` started with a location-shift
  formula rather than a probabilistic model in Phase 1: the smallest
  mechanism that can produce a falsifiable result. CUSUM is a much older,
  simpler, and more widely deployed method (statistical process control)
  than the sequential-Bayesian approach, and, notably, self-calibrating
  from local data the way Adams & MacKay's exact posterior does *not*
  require external hyperparameters at all — worth reconsidering if
  `CUSUMTemporalReasoner`'s hand-tuned `h_sigma`/`burn_in` calibration
  (see `experiments/exp05_regime_change_detection/RESULTS.md`) proves too
  fragile in practice.
- Regime-switching in time series, as a formal generative model (distinct
  from this project's unrelated use of "reference frame" as a name):
  Hamilton, *A New Approach to the Economic Analysis of Nonstationary
  Time Series and the Business Cycle*, Econometrica 57(2), 1989. The
  canonical econometric formalization of "the process is generated by one
  of several discrete regimes, with unknown, probabilistically-inferred
  switch points" — exactly the synthetic generative process
  `experiments/exp05_regime_change_detection/run.py` constructs to get
  ground truth, and a substantially richer model (a full hidden Markov
  chain with transition probabilities and maximum-likelihood inference)
  than the CUSUM detector actually built and tested.
- Hidden Markov models generally: Rabiner, *A Tutorial on Hidden Markov
  Models and Selected Applications in Speech Recognition*, Proceedings of
  the IEEE 77(2), 1989 (crediting Baum & Petrie, 1966, for the original
  theory). The general framework Hamilton's regime-switching model is a
  special case of; the natural upgrade path if segmenting into more than
  simple mean-shift regimes (e.g. regimes with different variances or
  transition dynamics) becomes necessary.
- Temporal comparison of sequences that may be time-shifted or of
  different lengths: Sakoe, Chiba, *Dynamic Programming Algorithm
  Optimization for Spoken Word Recognition*, IEEE Transactions on
  Acoustics, Speech, and Signal Processing 26(1), 1978 (Dynamic Time
  Warping). Implemented as `dynamic_time_warp()` and
  `CUSUMTemporalReasoner.trajectory_distance()` — the basic symmetric
  form, no slope constraint (the paper's own refinement, not built here).
  `compare()` remains a separate, simpler pointwise field-diff between two
  `State`s (mirroring `ReferenceFrame.differences()`) for the case where
  DTW's sequence alignment isn't needed.
- Multivariate extension, for tracking more than one key at once: Crosier,
  *Multivariate Generalizations of Cumulative Sum Quality-Control
  Schemes*, Technometrics 30(3), 1988. Implemented as
  `joint_change_points()`, using Crosier's simpler "reduce each
  multivariate observation to a scalar, then run a standard CUSUM on it"
  variant — summing per-key z-scored deviations, normalized by
  `sqrt(n_keys)` — rather than his alternative direct-vector-CUSUM
  procedure. This assumes independence across keys (no cross-covariance
  term), a real simplification relative to a full multivariate treatment.

**Classification: established theory in full for all five citations —
CUSUM, regime-switching HMMs, general HMM theory, DTW, and multivariate
CUSUM are all textbook. The specific choices made here (CUSUM
self-calibrated from a short rolling window and restarting after each
detected change; the simpler scalar-reduction multivariate variant rather
than a full covariance-aware one; DTW without the slope-constraint
refinement) are engineering simplifications of that theory, not novel
methods — and the calibration turned out to be non-trivial in practice at
every step (see the false-positive-rate findings for both the single-key
and multi-key cases, and the non-Gaussian-noise finding, in
`experiments/exp05_regime_change_detection/RESULTS.md`), which is itself
evidence for why the richer alternatives (Hamilton's full regime-switching
MLE, a proper Bayesian changepoint posterior, covariance-aware
multivariate CUSUM, robust/nonparametric CUSUM variants) exist in the
literature rather than everyone using the simplest version everywhere.**

## 10. What this changes in `research-agenda.md`

- Experiment 1's baseline needs to be checked against arXiv:2102.06177
  and arXiv:2207.02249 before being called a fair baseline — if those
  papers' "richer learned task embedding" condition already beats flat
  task-ID conditioning the way our RF-aware agent is expected to, the
  interesting comparison shifts from *(flat vs. structured)* to
  *(learned embedding vs. explicit human-legible frame)*, which is a
  narrower and arguably more interesting claim about interpretability,
  not raw reward.
- Experiment 3's "structurally isomorphic domain" needs to be built using
  Gentner's systematicity criterion explicitly (shared, interconnected
  relational structure; deliberately differing attributes) so that a
  reviewer can't dismiss it as relabeling.
- Phase 8-9 (meta-intelligence, strange loops) should not get engineering
  time until there's a Rabinowitz-style operationalization: a concrete,
  observable behavioral difference a self-model would need to produce.
  Right now there isn't one, so there's nothing to build yet.
