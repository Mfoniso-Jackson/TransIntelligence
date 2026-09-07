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

**Classification: established theory. TransIntelligence's causal reasoning
module doesn't exist yet (`transintelligence/reasoning/causal/__init__.py`
is an empty stub) — nothing to classify as novel until it's built.**

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
| Causal framing of `evaluate(x,R)` | Pearl 2009 | Established theory; not yet implemented here |
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
