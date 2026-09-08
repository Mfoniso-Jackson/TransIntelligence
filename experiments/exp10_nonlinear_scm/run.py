"""Experiment 10 -- the last of the three stated gaps from
docs/findings.md's "what isn't tested yet" list (Phase 5, continued):
(a) is linear OLS-based backdoor adjustment biased when the true
relationship is nonlinear, and (b) does counterfactual abduction, which
this codebase already argued needs only additive noise, not linearity,
actually stay exact for a nonlinear structural equation?
(docs/research-agenda.md #7g)

Ground-truth SCM, reusing experiments 6/7/9's confounding-graph shape
(Z -> X, Z -> Y, X -> Y):

    Z ~ N(0, 1)                                  (confounder)
    X = ALPHA*Z + eps_X                          (treatment; linear in Z)
    Y = BETA*Z + GAMMA1*X + GAMMA2*X^2 + eps_Y   (outcome; QUADRATIC in X)

`{Z}` is still the only backdoor-valid adjustment set (identical graph
shape to experiment 6) -- nonlinearity changes the *functional form* of
the X->Y relationship, not the graph, and `satisfies_backdoor_criterion`
doesn't care about functional form at all. What breaks is downstream:
`ordinary_least_squares`-based effect estimation silently assumes the
adjusted relationship is linear.

## What "the effect of X on Y" even means once GAMMA2 != 0

For a genuinely nonlinear relationship, there is no single scalar "the
causal effect" -- the effect of shifting a unit's X by +1 depends on
that unit's own X value (heterogeneous effect):
`shift_effect(x) = f(x+1) - f(x) = GAMMA1 + GAMMA2*(2x + 1)`. A linear
model produces exactly one number regardless of x. This experiment
measures the gap between that one number and the true, x-dependent shift
effect -- computed exactly via `StructuralCausalModel.counterfactual()`,
not estimated: `counterfactual(observed, {"X": x+1})["Y"] - observed["Y"]`
equals `f(x+1) - f(x)` exactly, because each unit's own exogenous noise
is identical in both terms and cancels out algebraically, the same
"exact by construction" property experiment 7 exploited for the linear
case (verified in `tests/test_counterfactual_reasoning.py` before this
experiment was built).

## The confound this needed to control for

Showing linear-adjusted OLS fail on a nonlinear model, on its own, could
just mean the experimental setup itself is broken -- not that
nonlinearity specifically is the cause. A `GAMMA2=0.0` control condition
(the true relationship IS linear) has to show linear-adjusted OLS closely
matching the true shift effect there, or the "nonlinearity causes bias"
claim isn't actually demonstrated, just asserted.

Run: PYTHONPATH=. python experiments/exp10_nonlinear_scm/run.py
"""
from __future__ import annotations

import random
import statistics

from transintelligence.reasoning.causal import CausalGraph, ordinary_least_squares
from transintelligence.reasoning.counterfactual import StructuralCausalModel, StructuralEquation

ALPHA = 0.8   # Z -> X
BETA = 0.8    # Z -> Y
GAMMA1 = 0.3  # linear part of X -> Y
NOISE_SIGMA = 0.3
N_PER_TRIAL = 500
SEEDS = list(range(30))
REFERENCE_POINTS = [-2.0, -1.0, 0.0, 1.0, 2.0]

GRAPH = CausalGraph((("Z", "X"), ("Z", "Y"), ("X", "Y")))


def make_scm(gamma2: float) -> StructuralCausalModel:
    return StructuralCausalModel(GRAPH, {
        "X": StructuralEquation({"Z": ALPHA}),
        "Y": StructuralEquation(
            nonlinear_fn=lambda p, g2=gamma2: BETA * p["Z"] + GAMMA1 * p["X"] + g2 * p["X"] ** 2,
            nonlinear_parents=("Z", "X"),
        ),
    })


def simulate(seed: int, gamma2: float, n: int = N_PER_TRIAL) -> list[dict[str, float]]:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        z = rng.gauss(0, 1)
        x = ALPHA * z + rng.gauss(0, NOISE_SIGMA)
        y = BETA * z + GAMMA1 * x + gamma2 * x * x + rng.gauss(0, NOISE_SIGMA)
        rows.append({"Z": z, "X": x, "Y": y})
    return rows


def linear_adjusted_effect(rows: list[dict[str, float]]) -> float:
    """The backdoor-adjusted OLS estimate reasoning/causal/ produces --
    the standard operationalization used throughout experiments 6, 7, 9."""
    features = [[1.0, row["X"], row["Z"]] for row in rows]
    targets = [row["Y"] for row in rows]
    return ordinary_least_squares(features, targets)[1]


def true_average_shift_effect(rows: list[dict[str, float]], gamma2: float) -> float:
    """The exact population-average effect of shifting every unit's X by
    +1, computed per-unit via counterfactual abduction (noise cancels
    exactly, no estimation error from this side at all) and averaged."""
    scm = make_scm(gamma2)
    shifts = []
    for row in rows:
        observed = {"Z": row["Z"], "X": row["X"], "Y": row["Y"]}
        cf = scm.counterfactual(observed, {"X": row["X"] + 1.0})
        shifts.append(cf["Y"] - observed["Y"])
    return statistics.mean(shifts)


def true_shift_effect_at(x0: float, gamma2: float) -> float:
    """Closed-form shift_effect(x0) = f(x0+1) - f(x0), for reference-point
    comparison against the single linear-model coefficient."""
    f = lambda x: GAMMA1 * x + gamma2 * x * x
    return f(x0 + 1.0) - f(x0)


def main() -> None:
    print("=== Backdoor-valid adjustment set, independent of functional form ===")
    print(f"  satisfies_backdoor_criterion(X, Y, {{Z}}) = {GRAPH.satisfies_backdoor_criterion('X', 'Y', {'Z'})}")

    for label, gamma2 in [("linear ground truth (GAMMA2=0.0, control)", 0.0), ("nonlinear ground truth (GAMMA2=0.6)", 0.6)]:
        print(f"\n=== {label} ===")
        linear_estimates, true_effects = [], []
        for seed in SEEDS:
            rows = simulate(seed, gamma2)
            linear_estimates.append(linear_adjusted_effect(rows))
            true_effects.append(true_average_shift_effect(rows, gamma2))
        mean_linear = statistics.mean(linear_estimates)
        mean_true = statistics.mean(true_effects)
        print(f"  mean linear-adjusted OLS estimate:        {mean_linear:.4f}")
        print(f"  mean true average shift effect (exact):   {mean_true:.4f}")
        print(f"  absolute gap:                              {abs(mean_linear - mean_true):.4f}")

        print(f"  true shift_effect(x0) at fixed reference points (heterogeneity check):")
        for x0 in REFERENCE_POINTS:
            print(f"    x0={x0:>5.1f}  true_shift_effect={true_shift_effect_at(x0, gamma2):.4f}  linear_model_predicts={mean_linear:.4f}")


if __name__ == "__main__":
    main()
