"""Regression tests for Experiment 10 (docs/research-agenda.md #7g,
experiments/exp10_nonlinear_scm/RESULTS.md). Small-scale versions of the
real experiment (fewer seeds/samples, for test-suite speed).
"""
import statistics

from experiments.exp10_nonlinear_scm.run import (
    GRAPH,
    linear_adjusted_effect,
    simulate,
    true_average_shift_effect,
    true_shift_effect_at,
)


def test_backdoor_valid_set_is_unaffected_by_functional_form():
    assert GRAPH.satisfies_backdoor_criterion("X", "Y", {"Z"}) is True


def test_linear_adjustment_matches_true_effect_when_the_truth_is_linear():
    """Control condition: GAMMA2=0.0 means the true relationship really
    is linear, so linear-adjusted OLS should closely match the exact
    true average shift effect -- this is what confirms any gap seen in
    the nonlinear condition is caused by nonlinearity specifically."""
    linear_estimates, true_effects = [], []
    for seed in range(15):
        rows = simulate(seed, gamma2=0.0, n=200)
        linear_estimates.append(linear_adjusted_effect(rows))
        true_effects.append(true_average_shift_effect(rows, gamma2=0.0))
    gap = abs(statistics.mean(linear_estimates) - statistics.mean(true_effects))
    assert gap < 0.1


def test_linear_adjustment_is_substantially_biased_when_the_truth_is_nonlinear():
    linear_estimates, true_effects = [], []
    for seed in range(15):
        rows = simulate(seed, gamma2=0.6, n=200)
        linear_estimates.append(linear_adjusted_effect(rows))
        true_effects.append(true_average_shift_effect(rows, gamma2=0.6))
    gap = abs(statistics.mean(linear_estimates) - statistics.mean(true_effects))
    assert gap > 0.4


def test_true_shift_effect_is_heterogeneous_under_nonlinearity_but_constant_under_linearity():
    """The core qualitative claim: a nonlinear relationship has a
    genuinely different effect at different points, a linear one
    doesn't."""
    linear_effects = [true_shift_effect_at(x0, gamma2=0.0) for x0 in (-2.0, 0.0, 2.0)]
    assert len(set(round(e, 6) for e in linear_effects)) == 1  # constant

    nonlinear_effects = [true_shift_effect_at(x0, gamma2=0.6) for x0 in (-2.0, 0.0, 2.0)]
    assert nonlinear_effects[0] < nonlinear_effects[1] < nonlinear_effects[2]  # strictly increasing
    assert nonlinear_effects[0] < 0 < nonlinear_effects[2]  # even flips sign
