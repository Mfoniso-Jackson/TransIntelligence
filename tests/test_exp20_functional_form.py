"""Regression tests for experiment 20's functional-form follow-up
(docs/research-agenda.md #7q,
experiments/exp20_noise_sweep_compounding_error/RESULTS.md "Follow-up"
section).
"""
from experiments.exp20_noise_sweep_compounding_error.functional_form import (
    fit_linear, fit_power_law, fit_quadratic, r_squared,
)


def test_r_squared_is_exactly_one_for_a_perfect_fit():
    actual = [1.0, 2.0, 3.0, 4.0]
    assert r_squared(actual, actual) == 1.0


def test_r_squared_is_zero_when_predictions_equal_the_mean():
    """Predicting the mean for everything is the "no explanatory power"
    baseline -- R^2 should be exactly 0, not negative or positive."""
    actual = [1.0, 2.0, 3.0, 5.0]
    mean = sum(actual) / len(actual)
    assert r_squared(actual, [mean] * len(actual)) == 0.0


def test_fit_functions_recover_an_exact_synthetic_relationship():
    """Hand-verified: points generated EXACTLY from gap = 2.0 + 3.0*sigma
    should be recovered by fit_linear with R^2 == 1.0 and the exact
    coefficients, not just approximately."""
    points = [(sigma, 2.0 + 3.0 * sigma) for sigma in (0.0, 0.1, 0.2, 0.3, 0.4)]
    (a, b), r2 = fit_linear(points)
    assert abs(a - 2.0) < 1e-9
    assert abs(b - 3.0) < 1e-9
    assert abs(r2 - 1.0) < 1e-9


def test_fit_quadratic_recovers_an_exact_synthetic_relationship():
    points = [(sigma, 1.0 + 4.0 * sigma ** 2) for sigma in (0.0, 0.1, 0.2, 0.3, 0.4)]
    (a, b), r2 = fit_quadratic(points)
    assert abs(a - 1.0) < 1e-9
    assert abs(b - 4.0) < 1e-9
    assert abs(r2 - 1.0) < 1e-9


def test_fit_power_law_recovers_an_exact_synthetic_power_relationship():
    """Hand-verified: points generated EXACTLY from gap = 5.0 * sigma^0.5
    (sigma=0 excluded, matching fit_power_law's own exclusion) should be
    recovered with R^2 == 1.0 and the exact exponent."""
    import math
    points = [(0.0, 0.0)] + [(sigma, 5.0 * sigma ** 0.5) for sigma in (0.01, 0.04, 0.09, 0.16, 0.25)]
    (log_a, b), r2 = fit_power_law(points)
    assert abs(math.exp(log_a) - 5.0) < 1e-6
    assert abs(b - 0.5) < 1e-6
    assert abs(r2 - 1.0) < 1e-9
