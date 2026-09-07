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

## 9. Summary table

| Area | Nearest citation(s) | Classification |
|---|---|---|
| Reference frame as conditioning context | Li et al. 2010; Kaelbling et al. 1998; Schaul et al. 2015; arXiv:2102.06177; Duan et al. 2016; Finn et al. 2017 | Established theory — mechanism is old, explicit/legible representation is an engineering choice |
| Causal framing of `evaluate(x,R)` | Pearl 2009 | Established theory; not yet implemented here |
| Frame-dependence detection | Arjovsky et al. 2019; arXiv:2010.05761; Zhou et al. 2023; Wang et al. 2022 | Established theory; current `sensitivity()` is a naive baseline against it |
| Geometric reasoning over non-physical spaces | Bronstein et al. 2021 | Established theory, directly prior art |
| Cross-domain structural transfer | Gentner 1983; Lake & Baroni 2018/2023 | Established distinctions; TransIntelligence's specific transfer claim is a hypothesis |
| Strange loops / recursive self-model | Hofstadter 1979/2007; Friston 2010; Rabinowitz et al. 2018 | Motivation only, not yet a hypothesis — needs operationalization |
| "Reference frame" terminology | Fillmore 1982 | Naming inspiration, not a mechanism |

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
