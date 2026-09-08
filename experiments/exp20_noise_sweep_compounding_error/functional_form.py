"""Follow-up to experiment 20 -- what functional form best describes how
the post-shift reward gap scales with `NOISE_SIGMA`? Experiment 20
established the gap scales strongly and mostly-monotonically with noise,
but explicitly did not characterize the relationship's exact shape.
Fills that gap: refits the same 6 noise levels with more seeds (reducing
the point-estimate noise that obscured the relationship in the original
6-point sweep) and fits three simple candidate forms via
`ordinary_least_squares` (already used elsewhere in this codebase, no
new dependency) -- linear, quadratic, and power-law -- reporting R^2 for
each rather than assuming any one is correct.

Run: PYTHONPATH=. python experiments/exp20_noise_sweep_compounding_error/functional_form.py
"""
from __future__ import annotations

import math
import statistics

from experiments.exp20_noise_sweep_compounding_error.run import CONDITIONS, NOISE_SIGMAS, run_condition
from transintelligence.reasoning.causal import ordinary_least_squares

SEEDS = list(range(25))  # more than experiment 20's original 20, for more stable per-level gap estimates


def measure_gaps() -> list[tuple[float, float]]:
    """Returns [(noise_sigma, gap), ...] -- gap = mean(greedy_post) -
    mean(mpc_post), the same definition experiment 20 used."""
    results = []
    for noise_sigma in NOISE_SIGMAS:
        greedy_posts = [run_condition(CONDITIONS["greedy_cusum_adapts"], s, noise_sigma)[1] for s in SEEDS]
        mpc_posts = [run_condition(CONDITIONS["mpc_beam_cusum_adapts"], s, noise_sigma)[1] for s in SEEDS]
        gap = statistics.mean(greedy_posts) - statistics.mean(mpc_posts)
        results.append((noise_sigma, gap))
    return results


def r_squared(actual: list[float], predicted: list[float]) -> float:
    mean_actual = statistics.mean(actual)
    ss_res = sum((a - p) ** 2 for a, p in zip(actual, predicted))
    ss_tot = sum((a - mean_actual) ** 2 for a in actual)
    return 1.0 - ss_res / ss_tot if ss_tot else float("nan")


def fit_linear(points: list[tuple[float, float]]) -> tuple[list[float], float]:
    """gap = a + b*sigma"""
    features = [[1.0, sigma] for sigma, _ in points]
    targets = [gap for _, gap in points]
    coeffs = ordinary_least_squares(features, targets)
    predicted = [coeffs[0] + coeffs[1] * sigma for sigma, _ in points]
    return coeffs, r_squared(targets, predicted)


def fit_quadratic(points: list[tuple[float, float]]) -> tuple[list[float], float]:
    """gap = a + b*sigma^2"""
    features = [[1.0, sigma ** 2] for sigma, _ in points]
    targets = [gap for _, gap in points]
    coeffs = ordinary_least_squares(features, targets)
    predicted = [coeffs[0] + coeffs[1] * (sigma ** 2) for sigma, _ in points]
    return coeffs, r_squared(targets, predicted)


def fit_power_law(points: list[tuple[float, float]]) -> tuple[list[float], float]:
    """log(gap) = log(a) + b*log(sigma) -- excludes sigma=0 (undefined
    log) and any non-positive gap (also undefined)."""
    nonzero = [(sigma, gap) for sigma, gap in points if sigma > 0 and gap > 0]
    features = [[1.0, math.log(sigma)] for sigma, _ in nonzero]
    log_targets = [math.log(gap) for _, gap in nonzero]
    coeffs = ordinary_least_squares(features, log_targets)
    predicted_log = [coeffs[0] + coeffs[1] * math.log(sigma) for sigma, _ in nonzero]
    return coeffs, r_squared(log_targets, predicted_log)


def main() -> None:
    points = measure_gaps()
    print(f"{'noise_sigma':>11} {'gap':>10}")
    for sigma, gap in points:
        print(f"{sigma:>11.2f} {gap:>10.4f}")
    print()

    (a, b), r2_linear = fit_linear(points)
    print(f"linear:     gap = {a:.4f} + {b:.4f}*sigma        R^2 = {r2_linear:.4f}")

    (a, b), r2_quadratic = fit_quadratic(points)
    print(f"quadratic:  gap = {a:.4f} + {b:.4f}*sigma^2      R^2 = {r2_quadratic:.4f}")

    (log_a, b), r2_power = fit_power_law(points)
    print(f"power law:  gap = {math.exp(log_a):.4f} * sigma^{b:.4f}   R^2 (log-log) = {r2_power:.4f}")


if __name__ == "__main__":
    main()
