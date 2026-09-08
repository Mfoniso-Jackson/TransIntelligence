"""Experiment 7 -- does abduction-action-prediction correctly recover a
per-unit counterfactual, and does it genuinely beat the common shortcut of
just plugging the new treatment value into the population-average
regression equation? (docs/research-agenda.md #7d)

Reuses experiment 6's confounding graph and SCM exactly (Z->X, Z->Y,
X->Y) so the two experiments' numbers are directly comparable -- only
here we ask a per-unit question ("what would THIS unit's Y have been had
its X been different") instead of a population-average one ("does X
affect Y on average").

The confound this experiment needs to control for, following the
standing lesson from experiments 3-6: a naive "plug x' into the fitted
regression line" estimate will usually be *close* to the true
counterfactual on average across many units (it's an unbiased estimator
of the average treatment effect, by construction, if the regression is
correctly specified) -- so a good demonstration has to show it is
systematically WRONG for *individual* units with nonzero residuals, not
just show one lucky/unlucky example. Two conditions:
  - `true_coefficients`: abduction and the naive baseline both use the
    exact true structural equation coefficients (no estimation error) --
    isolates whether the abduction *mechanism* is correct in principle.
  - `estimated_coefficients`: both use OLS-fit coefficients from
    observational data (the same fitting experiment 6 already does) --
    the realistic end-to-end case.

Run: PYTHONPATH=. python experiments/exp07_counterfactual_queries/run.py
"""
from __future__ import annotations

import random
import statistics

from transintelligence.reasoning.causal import CausalGraph, ordinary_least_squares
from transintelligence.reasoning.counterfactual import StructuralCausalModel, StructuralEquation

ALPHA = 0.8   # Z -> X (true)
BETA = 0.8    # Z -> Y (true)
TRUE_EFFECT = 0.5  # X -> Y (true) -- nonzero here, unlike experiment 6's null case,
                   # to show abduction recovers a real effect's per-unit consequences too
NOISE_SIGMA = 0.3
N_FIT = 2000       # observational sample used to fit coefficients (estimated_coefficients condition)
N_UNITS = 200      # units whose counterfactuals are actually queried
NEW_X = 2.0        # the counterfactual intervention: do(X = NEW_X)
SEEDS = list(range(20))

# Same graph as experiment 6: Z confounds X and Y, X also causes Y directly.
# This experiment is about counterfactual computation, not adjustment-set
# validity, so the graph is only used to encode each node's parents.
GRAPH = CausalGraph((("Z", "X"), ("Z", "Y"), ("X", "Y")))


def true_scm() -> StructuralCausalModel:
    return StructuralCausalModel(GRAPH, {
        "X": StructuralEquation({"Z": ALPHA}),
        "Y": StructuralEquation({"Z": BETA, "X": TRUE_EFFECT}),
    })


def simulate_units(seed: int, n: int) -> list[dict[str, float]]:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        z = rng.gauss(0, 1)
        x = ALPHA * z + rng.gauss(0, NOISE_SIGMA)
        y = BETA * z + TRUE_EFFECT * x + rng.gauss(0, NOISE_SIGMA)
        rows.append({"Z": z, "X": x, "Y": y})
    return rows


def fit_scm(rows: list[dict[str, float]]) -> StructuralCausalModel:
    """OLS-estimate the structural equations from observational data --
    the realistic case where the true coefficients aren't known."""
    x_features = [[1.0, row["Z"]] for row in rows]
    x_targets = [row["X"] for row in rows]
    x_coef = ordinary_least_squares(x_features, x_targets)  # [intercept, Z-coef]

    y_features = [[1.0, row["Z"], row["X"]] for row in rows]
    y_targets = [row["Y"] for row in rows]
    y_coef = ordinary_least_squares(y_features, y_targets)  # [intercept, Z-coef, X-coef]

    return StructuralCausalModel(GRAPH, {
        "X": StructuralEquation({"Z": x_coef[1]}, intercept=x_coef[0]),
        "Y": StructuralEquation({"Z": y_coef[1], "X": y_coef[2]}, intercept=y_coef[0]),
    })


def naive_counterfactual(scm: StructuralCausalModel, observed: dict[str, float], new_x: float) -> float:
    """The common shortcut: plug new_x into the fitted population
    regression line, WITHOUT abduction -- implicitly assumes this unit's
    residual is exactly 0, i.e. it IS the population average."""
    y_eq = scm.equations["Y"]
    return y_eq.predict({"Z": observed["Z"], "X": new_x})


def true_counterfactual(observed: dict[str, float], new_x: float) -> float:
    """Ground truth: literally recompute Y for this exact unit's true
    exogenous noise (available here because we control the simulator),
    which the reasoner never sees directly -- only the observed values."""
    eps_y = observed["Y"] - (BETA * observed["Z"] + TRUE_EFFECT * observed["X"])
    return BETA * observed["Z"] + TRUE_EFFECT * new_x + eps_y


def run_condition(seed: int, scm: StructuralCausalModel) -> tuple[float, float]:
    """Returns (mean_abs_error_abducted, mean_abs_error_naive) across N_UNITS units."""
    units = simulate_units(seed, N_UNITS)
    abducted_errors, naive_errors = [], []
    for unit in units:
        truth = true_counterfactual(unit, NEW_X)
        abducted = scm.counterfactual(unit, {"X": NEW_X})["Y"]
        naive = naive_counterfactual(scm, unit, NEW_X)
        abducted_errors.append(abs(abducted - truth))
        naive_errors.append(abs(naive - truth))
    return statistics.mean(abducted_errors), statistics.mean(naive_errors)


def main() -> None:
    print(f"{'condition':<22} {'mean_abs_error_abducted':>24} {'mean_abs_error_naive':>22}")

    scm_true = true_scm()
    abducted_true = [run_condition(seed, scm_true)[0] for seed in SEEDS]
    naive_true = [run_condition(seed, scm_true)[1] for seed in SEEDS]
    print(f"{'true_coefficients':<22} {statistics.mean(abducted_true):>24.4f} {statistics.mean(naive_true):>22.4f}")

    abducted_est, naive_est = [], []
    for seed in SEEDS:
        fit_rows = simulate_units(seed + 10_000, N_FIT)  # disjoint sample from the queried units
        scm_est = fit_scm(fit_rows)
        a, n = run_condition(seed, scm_est)
        abducted_est.append(a)
        naive_est.append(n)
    print(f"{'estimated_coefficients':<22} {statistics.mean(abducted_est):>24.4f} {statistics.mean(naive_est):>22.4f}")


if __name__ == "__main__":
    main()
