"""Regression tests for Experiment 9 (docs/research-agenda.md #7f,
experiments/exp09_iv_and_frontdoor/RESULTS.md). Small-scale versions of
the real experiment (fewer seeds/samples, for test-suite speed).
"""
import statistics

from transintelligence.reasoning.causal import (
    front_door_adjustment,
    ordinary_least_squares,
    two_stage_least_squares,
)

from experiments.exp09_iv_and_frontdoor.run import (
    TRUE_EFFECT_XY,
    TRUE_MY,
    TRUE_XM,
    VIOLATION_STRENGTH,
    simulate_frontdoor,
    simulate_iv,
)


def test_2sls_recovers_true_effect_with_a_strong_instrument():
    estimates = []
    for seed in range(15):
        z, x, y = simulate_iv(seed, instrument_strength=0.9, n=200)
        estimates.append(two_stage_least_squares(z, x, y))
    assert abs(statistics.mean(estimates) - TRUE_EFFECT_XY) < 0.15


def test_2sls_degrades_sharply_as_instrument_weakens():
    """Regression guard for the core confound this experiment checks:
    2SLS must not remain reliable regardless of instrument strength --
    variance should grow substantially as the instrument weakens."""
    strong_estimates, weak_estimates = [], []
    for seed in range(15):
        z, x, y = simulate_iv(seed, instrument_strength=0.9, n=200)
        strong_estimates.append(two_stage_least_squares(z, x, y))
        z, x, y = simulate_iv(seed, instrument_strength=0.05, n=200)
        weak_estimates.append(two_stage_least_squares(z, x, y))
    assert statistics.pstdev(weak_estimates) > 5 * statistics.pstdev(strong_estimates)


def test_naive_ols_is_biased_by_the_unobserved_confounder_regardless_of_instrument_strength():
    naive_estimates = []
    for seed in range(15):
        _, x, y = simulate_iv(seed, instrument_strength=0.9, n=200)
        naive_estimates.append(ordinary_least_squares([[1.0, xv] for xv in x], y)[1])
    assert statistics.mean(naive_estimates) - TRUE_EFFECT_XY > 0.2


def test_front_door_recovers_true_effect_when_its_assumption_holds():
    true_total_effect = TRUE_XM * TRUE_MY
    estimates = []
    for seed in range(15):
        x, m, y = simulate_frontdoor(seed, u_to_m_strength=0.0, n=200)
        estimates.append(front_door_adjustment(x, m, y))
    assert abs(statistics.mean(estimates) - true_total_effect) < 0.05


def test_front_door_is_biased_when_the_confounder_also_affects_the_mediator():
    """Regression guard for the confound control this experiment adds:
    front_door_adjustment must not be robust to a violation of its own
    identification assumption (U having no effect on M) -- it should be
    substantially more biased than the valid condition on identical code,
    the same 'adjusting for the wrong thing is worse, not neutral'
    pattern experiment 6 established for the backdoor criterion."""
    true_total_effect = TRUE_XM * TRUE_MY
    valid_estimates, violated_estimates = [], []
    for seed in range(15):
        x, m, y = simulate_frontdoor(seed, u_to_m_strength=0.0, n=200)
        valid_estimates.append(front_door_adjustment(x, m, y))
        x, m, y = simulate_frontdoor(seed, u_to_m_strength=VIOLATION_STRENGTH, n=200)
        violated_estimates.append(front_door_adjustment(x, m, y))
    valid_bias = abs(statistics.mean(valid_estimates) - true_total_effect)
    violated_bias = abs(statistics.mean(violated_estimates) - true_total_effect)
    assert violated_bias > 5 * valid_bias
