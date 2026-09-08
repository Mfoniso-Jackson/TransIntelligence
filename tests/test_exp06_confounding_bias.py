"""Regression tests for Experiment 6 (docs/research-agenda.md #7c,
experiments/exp06_confounding_bias/RESULTS.md). Small-scale versions of
the real experiment (fewer seeds/samples, for test-suite speed).
"""
import statistics

from experiments.exp06_confounding_bias.run import GRAPH, estimate_x_coefficient, simulate


def test_graph_theoretic_validity_matches_the_experiment_design():
    assert GRAPH.satisfies_backdoor_criterion("X", "Y", set()) is False
    assert GRAPH.satisfies_backdoor_criterion("X", "Y", {"Z"}) is True
    assert GRAPH.satisfies_backdoor_criterion("X", "Y", {"W"}) is False
    assert GRAPH.satisfies_backdoor_criterion("X", "Y", {"Z", "W"}) is False


def test_naive_estimate_is_substantially_biased_by_confounding():
    estimates = [estimate_x_coefficient(simulate(seed, n=200), []) for seed in range(15)]
    assert statistics.mean(estimates) > 0.5  # true effect is 0.0


def test_backdoor_adjusted_estimate_recovers_the_true_null_effect():
    estimates = [estimate_x_coefficient(simulate(seed, n=200), ["Z"]) for seed in range(15)]
    assert abs(statistics.mean(estimates)) < 0.1  # true effect is 0.0


def test_adjusting_for_the_collider_is_worse_than_adjusting_for_the_confounder():
    """The graph correctly flags {W} as backdoor-invalid; this checks that
    invalidity actually shows up as worse bias than the valid {Z}
    adjustment, not just a theoretical label with no empirical
    consequence."""
    z_estimates = [estimate_x_coefficient(simulate(seed, n=200), ["Z"]) for seed in range(15)]
    w_estimates = [estimate_x_coefficient(simulate(seed, n=200), ["W"]) for seed in range(15)]
    z_bias = statistics.mean(abs(e) for e in z_estimates)
    w_bias = statistics.mean(abs(e) for e in w_estimates)
    assert w_bias > z_bias


def test_adding_the_collider_to_a_valid_adjustment_reintroduces_bias():
    """Regression guard for the most important nuance in RESULTS.md:
    {Z} alone is correctly unbiased; {Z, W} together should be MORE
    biased than {Z} alone, because W is a collider whose inclusion opens
    a spurious path even though Z is still present and correctly blocks
    the original confounding path."""
    z_only = [estimate_x_coefficient(simulate(seed, n=200), ["Z"]) for seed in range(15)]
    z_and_w = [estimate_x_coefficient(simulate(seed, n=200), ["Z", "W"]) for seed in range(15)]
    bias_z_only = statistics.mean(abs(e) for e in z_only)
    bias_z_and_w = statistics.mean(abs(e) for e in z_and_w)
    assert bias_z_and_w > bias_z_only
