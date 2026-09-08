"""Experiment 6 -- does the backdoor criterion correctly predict which
adjustment sets remove confounding bias, using a directly-constructed
version of the classic "correlation is not causation" demonstration
(Simpson 1951; Pearl 1995) rather than citing it from authority?
(docs/research-agenda.md #7c)

Ground-truth linear structural causal model:
    Z ~ N(0, 1)                          (confounder)
    X = alpha*Z + eps_X                  (treatment, partly driven by Z)
    Y = beta*Z + TRUE_EFFECT*X + eps_Y   (outcome; TRUE_EFFECT = 0.0 --
                                           X has NO real causal effect on Y)
    W = d*X + e*Y + eps_W                (collider: a common EFFECT of
                                           X and Y, not a cause of either)

Graph: Z->X, Z->Y, X->Y, X->W, Y->W. This is exactly the canonical graph
tested in tests/test_causal_reasoning.py.

Four conditions, each just "which covariates does the OLS regression of Y
on X include":
  - naive:      Y ~ X            (no adjustment -- confounded by Z)
  - adjusted:   Y ~ X + Z        (backdoor-valid: Z blocks the only
                                   backdoor path X<-Z->Y)
  - collider:   Y ~ X + W        (backdoor-INVALID: W is a descendant of
                                   X, and conditioning on a collider opens
                                   a spurious path -- the confound control
                                   this experiment needs, per the lesson
                                   from experiments 3/4/5: don't just show
                                   the positive case, show what happens
                                   when you adjust for the WRONG thing)
  - both:       Y ~ X + Z + W    (does adding the collider on top of the
                                   correct adjustment reintroduce bias?)

Run: PYTHONPATH=. python experiments/exp06_confounding_bias/run.py
"""
from __future__ import annotations

import random
import statistics

from transintelligence.reasoning.causal import CausalGraph, ordinary_least_squares

ALPHA = 0.8   # Z -> X
BETA = 0.8    # Z -> Y
TRUE_EFFECT = 0.0  # X -> Y (the actual causal effect being estimated)
D = 0.5       # X -> W
E = 0.5       # Y -> W
NOISE_SIGMA = 0.3
N_PER_TRIAL = 500
SEEDS = list(range(50))

GRAPH = CausalGraph((("Z", "X"), ("Z", "Y"), ("X", "Y"), ("X", "W"), ("Y", "W")))


def simulate(seed: int, n: int = N_PER_TRIAL) -> list[dict[str, float]]:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        z = rng.gauss(0, 1)
        x = ALPHA * z + rng.gauss(0, NOISE_SIGMA)
        y = BETA * z + TRUE_EFFECT * x + rng.gauss(0, NOISE_SIGMA)
        w = D * x + E * y + rng.gauss(0, NOISE_SIGMA)
        rows.append({"Z": z, "X": x, "Y": y, "W": w})
    return rows


def estimate_x_coefficient(rows: list[dict[str, float]], covariates: list[str]) -> float:
    features = [[1.0, row["X"]] + [row[c] for c in covariates] for row in rows]
    targets = [row["Y"] for row in rows]
    coefficients = ordinary_least_squares(features, targets)
    return coefficients[1]  # index 0 is the intercept, 1 is X's coefficient


CONDITIONS: dict[str, list[str]] = {
    "naive": [],
    "adjusted_Z": ["Z"],
    "collider_W": ["W"],
    "both_Z_and_W": ["Z", "W"],
}


def main() -> None:
    print("Graph-theoretic validity (satisfies_backdoor_criterion), independent of any data:")
    for label, covariates in CONDITIONS.items():
        valid = GRAPH.satisfies_backdoor_criterion("X", "Y", set(covariates))
        set_str = str(set(covariates)) if covariates else "{}"
        print(f"  {label:<14} adjustment_set={set_str:<12} backdoor_valid={valid}")

    print(f"\nEmpirical estimate of X's coefficient on Y (true effect = {TRUE_EFFECT}), {len(SEEDS)} seeds x {N_PER_TRIAL} samples:")
    print(f"{'condition':<14} {'mean_estimate':>14} {'stdev':>8} {'mean_abs_bias':>14}")
    for label, covariates in CONDITIONS.items():
        estimates = [estimate_x_coefficient(simulate(seed), covariates) for seed in SEEDS]
        mean_est = statistics.mean(estimates)
        bias = statistics.mean(abs(e - TRUE_EFFECT) for e in estimates)
        print(f"{label:<14} {mean_est:>14.4f} {statistics.pstdev(estimates):>8.4f} {bias:>14.4f}")


if __name__ == "__main__":
    main()
