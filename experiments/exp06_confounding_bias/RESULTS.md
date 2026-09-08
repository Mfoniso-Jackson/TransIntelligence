# Experiment 6 results — confounding bias and the backdoor criterion

Ran: `PYTHONPATH=. python experiments/exp06_confounding_bias/run.py`.
50 seeds × 500 samples per trial. See `docs/research-agenda.md` #7c for
the hypothesis and `docs/related-work.md` §11a for the established
theory this implements and tests against.

Ground-truth linear structural causal model, deliberately constructed
(not asserted from the literature) to demonstrate the textbook
"correlation is not causation" case directly, the same way experiment 5's
DTW comparison constructed its own motivating counterexample:

```
Z ~ N(0, 1)                          (confounder)
X = 0.8*Z + noise                    (treatment, partly driven by Z)
Y = 0.8*Z + 0.0*X + noise            (outcome; TRUE causal effect of X on Y is exactly ZERO)
W = 0.5*X + 0.5*Y + noise            (collider: a common EFFECT of X and Y, not a cause of either)
```

Graph: `Z→X, Z→Y, X→Y, X→W, Y→W` — exactly the graph
`tests/test_causal_reasoning.py` verifies `CausalGraph`'s d-separation and
backdoor-criterion logic against independently of this experiment.

## Graph-theoretic prediction (computed with no data at all)

| adjustment set | backdoor-valid? |
|---|---|
| `{}` (naive) | **False** |
| `{Z}` | **True** |
| `{W}` | **False** (W is a descendant of X) |
| `{Z, W}` | **False** (still contains a descendant of X) |

## Empirical result: the graph's prediction matches the data exactly

| condition | mean estimated effect | stdev | mean absolute bias (true effect = 0.0) |
|---|---|---|---|
| naive (`Y ~ X`) | **0.8808** | 0.021 | **0.8808** |
| adjusted (`Y ~ X + Z`) | 0.0065 | 0.039 | **0.0310** |
| collider (`Y ~ X + W`) | 0.2814 | 0.040 | 0.2814 |
| both (`Y ~ X + Z + W`) | -0.1914 | 0.044 | 0.1914 |

**This is about as clean as a falsifiable result gets: the only
backdoor-valid adjustment set is also the only one that recovers the
true effect (0.0065 vs. the true 0.0), and every backdoor-invalid set
produces substantial, confidently-estimated bias (stdev ≪ bias in every
biased condition — this isn't noise, it's systematic).**

- **Naive regression finds a large, entirely spurious "effect"** (0.88,
  when the truth is 0.0) — purely from Z driving both X and Y. This is
  the demonstration Simpson (1951) and decades of "confounding" warnings
  in applied statistics are about, built directly rather than cited.
- **Adjusting for the confounder Z correctly recovers ~0** — the
  backdoor criterion's practical payoff: knowing *which* variable to
  control for, from graph structure alone, before touching any data.
- **Adjusting for the collider W instead is not just "less good than Z" —
  it's a different failure mode with its own bias (0.28), correctly
  flagged as invalid by the same graph check.** Conditioning on a common
  effect of X and Y opens a spurious statistical path between them that
  doesn't exist in the naive (unadjusted) case at all — collider bias
  layered on top of whatever confounding bias remains.
- **The most important nuance: adding W to the already-correct {Z}
  adjustment makes things *worse*, not neutral.** {Z} alone: bias 0.031.
  {Z, W} together: bias 0.191, with the estimate even flipping sign
  (-0.19). Including the graph-invalid variable doesn't just fail to
  help — it actively contaminates an otherwise-correct model. **This is
  the same lesson experiments 3, 4, and 5 already established in
  different mechanisms (a confound control, a random-discovery control, a
  union-of-detectors control): "include more/any extra information"
  is not a safe default — the graph-theoretic validity check is doing
  real, load-bearing work, not just formal decoration.**

## What this establishes about the causal reasoning module itself

`CausalGraph.d_separated()` and `.satisfies_backdoor_criterion()` were
verified independently against the three canonical d-separation
structures (chain, fork, collider, plus a collider-with-conditioned-
descendant case) in `tests/test_causal_reasoning.py` *before* this
experiment was built — the graph-theoretic machinery was checked for
correctness on its own terms, not just validated by getting the expected
answer on the one graph this experiment happens to use. `ordinary_least_squares()`
was likewise checked against closed-form linear relationships (exact
recovery to floating-point precision) before being trusted for effect
estimation.

## What this does not establish

- **Linear structural equations only.** Both the true model and the
  estimator assume linearity — a real simplification (matching this
  repo's established practice of starting with the smallest mechanism
  that produces a falsifiable result), not a claim that this approach
  handles nonlinear confounding or effect modification.
- **A single, hand-constructed graph.** D-separation and the backdoor
  criterion were verified on canonical 3-4 node structures; this doesn't
  test causal *discovery* (inferring graph structure from data) or
  behavior on larger, more entangled graphs with multiple confounders,
  mediators, and colliders simultaneously.
- **No instrumental-variable or front-door adjustment paths tested** —
  only the backdoor criterion, the simplest and most commonly-needed
  identification strategy, per the "smallest mechanism" discipline. Front-
  door adjustment (for when a valid backdoor set doesn't exist, e.g. an
  unobserved confounder) is the natural next capability if this module is
  extended further.
- **No counterfactual (per-unit) queries** — this experiment estimates a
  population-level average causal effect via adjustment, not a
  unit-level "what would Y have been for this specific observation had X
  been different" counterfactual (Pearl's abduction-action-prediction
  sequence, or Rubin's potential-outcomes framework) — that's
  `CounterfactualReasoner`, still an empty stub, and the natural next
  Phase 5 experiment.
