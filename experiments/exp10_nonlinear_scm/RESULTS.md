# Experiment 10 results — nonlinear structural equations

Ran: `PYTHONPATH=. python experiments/exp10_nonlinear_scm/run.py`.
30 seeds × 500 samples per condition. See `docs/research-agenda.md` #7g
for the hypothesis and `docs/related-work.md` §3b (updated) for the
grounding theory.

This is the last of the three stated gaps from `docs/findings.md`'s "what
isn't tested yet" list. Two separate questions, using the same
confounding-graph shape as experiments 6, 7, and 9 (`Z→X, Z→Y, X→Y`), but
with a genuinely **quadratic** `X→Y` relationship this time:
`Y = 0.8·Z + 0.3·X + GAMMA2·X² + noise`.

## Graph-theoretic validity is unaffected by functional form

`satisfies_backdoor_criterion("X", "Y", {"Z"})` is `True`, exactly as in
experiment 6 — the backdoor criterion is a purely graph-theoretic test
and doesn't know or care whether the underlying relationship is linear.
What breaks under nonlinearity is downstream, in *effect estimation*
(`ordinary_least_squares`), not the criterion itself.

## Part A: does linear-adjusted OLS misestimate the true effect?

"The effect of X on Y" isn't a single number once the relationship is
nonlinear — shifting a unit's `X` by +1 changes `Y` by an amount that
depends on that unit's own `X` (heterogeneous effect). The **true
average shift effect** here is computed exactly, not estimated: per unit,
`counterfactual(observed, {"X": x+1})["Y"] - observed["Y"]` equals
`f(x+1) - f(x)` precisely, because the unit's own exogenous noise is
identical in both terms and cancels out algebraically — the same
"exact by construction" property experiment 7 used for the linear case.

| condition | mean linear-adjusted OLS | mean true average shift effect (exact) | absolute gap |
|---|---|---|---|
| linear ground truth (GAMMA2=0.0, control) | 0.3121 | 0.3000 | 0.0121 |
| nonlinear ground truth (GAMMA2=0.6) | 0.3255 | 0.9053 | **0.5799** |

**In the control condition (truth really is linear), linear-adjusted OLS
matches the exact true effect closely (gap 0.0121, consistent with
ordinary sampling noise) — confirming any gap in the nonlinear condition
is caused by nonlinearity specifically, not a flaw in the experimental
setup.** In the nonlinear condition, the gap is **0.5799** — roughly 48x
larger. Notably, the linear-adjusted estimate (0.3255) is barely
different from the *control* condition's estimate (0.3121), even though
the true effect nearly tripled (0.30 → 0.91): with `X` roughly symmetric
around 0, `Cov(X, X²) ≈ 0`, so OLS's linear coefficient is almost blind
to the entire quadratic contribution — it isn't reporting "a biased
version of the true effect," it's reporting a number that happens to
land near the *linear-only* answer regardless of how large the true
nonlinear effect actually is.

## Part B: heterogeneity the linear model cannot represent at all

At fixed reference points (`x0 ∈ {-2, -1, 0, 1, 2}`), the true shift
effect and what a single linear coefficient would predict for all of
them:

| x0 | true shift_effect(x0), GAMMA2=0.0 | true shift_effect(x0), GAMMA2=0.6 | linear model predicts (GAMMA2=0.6 condition) |
|---|---|---|---|
| -2.0 | 0.3000 | **-1.5000** | 0.3255 |
| -1.0 | 0.3000 | -0.3000 | 0.3255 |
| 0.0 | 0.3000 | 0.9000 | 0.3255 |
| 1.0 | 0.3000 | 2.1000 | 0.3255 |
| 2.0 | 0.3000 | **3.3000** | 0.3255 |

**Under the true linear relationship, the shift effect is constant across
every reference point (0.3000 everywhere) — exactly what a linear model
is built to represent.** Under the true nonlinear relationship, the shift
effect ranges from **-1.5000 to +3.3000** across the same five points —
it even **flips sign** at `x0=-2` (shifting X up there actually
*decreases* Y) while the linear model predicts the same +0.3255 "effect"
at every single point, including that one. This isn't a quantitative
miss, it's a category error: a linear coefficient cannot represent an
effect that depends on where you start.

## What this establishes

`StructuralEquation.nonlinear_fn` needed **zero changes** to
`abduct()`/`counterfactual()` themselves to support this — only
`predict()` and the parent-name lookup changed
(`transintelligence/reasoning/counterfactual/model.py`). This directly
confirms the claim stated in the module's own docstring before this
experiment was built: abduction only ever requires *additive* noise
(`node = f(parents) + noise`), never that `f` be linear, so the same
closed-form residual is exact for any `f`. Verified against a
hand-computed quadratic case in
`tests/test_counterfactual_reasoning.py` *before* this experiment used
it as its ground-truth reference generator — the same discipline as
every other experiment in this program: verify the mechanism on a
canonical case before trusting it for anything built on top of it.

## What this does not establish

- **Only one nonlinear functional form tested** (a single quadratic) —
  other nonlinearities (interactions, non-monotonic or discontinuous
  functions, non-additive noise) aren't covered; non-additive noise in
  particular would break abduction's closed-form residual entirely (the
  general case Balke & Pearl's methods, cited in §3b, were built for).
- **No nonlinear *effect estimation* alternative was built** — this
  experiment demonstrates linear-adjusted OLS's specific failure mode,
  it doesn't add a nonlinear regression method (e.g. polynomial
  features, kernel regression) to `reasoning/causal/` as a fix.
- **The functional form (`GAMMA1`, `GAMMA2`, the fact that it's
  quadratic) is given, not discovered** — `discover_skeleton`'s Fisher-z
  independence test (experiment 8) is itself linear-Gaussian and would
  likely miss purely nonlinear dependencies with zero linear partial
  correlation, an open gap noted in experiment 8's own RESULTS.md.
- **Single intervention variable, single mediating relationship** — the
  same scope limits experiment 7 already noted for the linear
  counterfactual case apply here too.
