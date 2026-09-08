# Experiment 7 results — per-unit counterfactual queries

Ran: `PYTHONPATH=. python experiments/exp07_counterfactual_queries/run.py`.
20 seeds × 200 queried units (plus a 2000-unit disjoint sample for fitting
in the `estimated_coefficients` condition). See `docs/research-agenda.md`
#7d for the hypothesis and `docs/related-work.md` §3b for the grounding
theory.

Reuses experiment 6's confounding graph (`Z→X, Z→Y, X→Y`) exactly, with
one difference: the true causal effect of X on Y here is **0.5** (nonzero),
not experiment 6's deliberate null case — this experiment is about
*per-unit* counterfactual accuracy for a real effect, not about detecting
confounding.

## The question and the confound to control for

`StructuralCausalModel.counterfactual()` implements Pearl's abduction-
action-prediction procedure: infer this unit's own exogenous noise from
its observed values, then recompute the outcome under the intervention
using that *same* noise. The obvious naive shortcut — just plug the new
treatment value into the fitted population regression line — is an
**unbiased estimator of the average effect**, so on its own that
comparison risks being unconvincing (naive could look "close enough" if
only checked on average). The real test is per-unit accuracy: does naive
systematically miss for individual units with nonzero residuals, in a way
abduction doesn't?

## Result

| condition | mean absolute error, abducted | mean absolute error, naive plug-in |
|---|---|---|
| true coefficients (no estimation error) | **0.0000** | 0.2413 |
| estimated coefficients (OLS-fit, realistic) | 0.0284 | 0.2440 |

**With the true coefficients, abduction recovers the exact per-unit
counterfactual (error indistinguishable from floating-point rounding) —
not approximately right, exactly right, because a linear+additive-noise
SCM's abduction step is a closed-form residual, not an approximation.**
Naive's error (0.2413) is not noise-of-measurement — it matches the
theoretical mean absolute value of the `N(0, 0.3)` exogenous noise this
experiment generates (`0.3 · √(2/π) ≈ 0.239`) almost exactly, confirming
precisely *why* naive is wrong: it silently assumes every unit's residual
is zero, so its error **equals that unit's discarded residual, on
average across units**.

**With estimated coefficients, abduction still wins decisively** (0.0284
vs. 0.2440, roughly 9x smaller) **and the gap is not close — naive's
error barely moves at all between the two conditions.** This is the
important structural point: naive's error comes from discarding
information (the unit's own residual), a problem *coefficient estimation
quality cannot fix* — no amount of more observational data would ever
bring naive's error toward zero, whereas abduction's small remaining
error (0.0284) is pure finite-sample estimation noise that *does* shrink
with more data (checked directly in `tests/test_exp07_counterfactual_queries.py`).

## What this establishes about the counterfactual reasoning module

`StructuralCausalModel.abduct()`/`.counterfactual()` were verified against
hand-computed cases in `tests/test_counterfactual_reasoning.py` *before*
this experiment was built — a simple chain with an exact expected residual
and counterfactual value, and a check that two units with the same
observed treatment but different outcomes get genuinely different
counterfactual predictions (the core claim that distinguishes this from a
population-average lookup). The experiment's near-exact match to the
true-coefficient prediction (error `0.0000`) confirms the mechanism
matches the closed-form arithmetic exactly, not just approximately.

## What this does not establish

- **Linear structural equations, additive noise only** — the same
  simplification `reasoning/causal/` already makes for effect estimation.
  Pearl's general counterfactual theory handles nonlinear/non-additive-
  noise SCMs; abduction there requires solving (possibly non-closed-form)
  equations, not a direct residual — not attempted here.
- **The graph and equation structure are given, not discovered or
  validated** — this experiment assumes the analyst already knows the
  correct structural equations (or estimates their coefficients correctly
  via OLS, itself dependent on the backdoor criterion being satisfied for
  the fitting regression — untested interaction between the two modules).
- **A single intervention point, single outcome variable** — multi-step
  or sequential interventions (e.g. "what if X had been different at
  time t, given later variables also changed as a result") aren't tested.
- **No comparison against Rubin's potential-outcomes framework
  computationally** — `docs/related-work.md` notes it as the alternative
  formalization, but no matching estimator was implemented to compare
  against directly.
