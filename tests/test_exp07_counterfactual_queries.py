"""Regression tests for Experiment 7 (docs/research-agenda.md #7d,
experiments/exp07_counterfactual_queries/RESULTS.md). Small-scale versions
of the real experiment (fewer seeds/units, for test-suite speed).
"""
import statistics

from experiments.exp07_counterfactual_queries.run import fit_scm, run_condition, simulate_units, true_scm


def test_abducted_counterfactual_is_essentially_exact_with_true_coefficients():
    scm = true_scm()
    abducted_error, _ = run_condition(seed=0, scm=scm)
    assert abducted_error < 1e-9


def test_naive_counterfactual_has_a_real_systematic_error_with_true_coefficients():
    """Regression guard for the central finding in RESULTS.md: the naive
    plug-in error should be close to the noise sigma's mean absolute
    value (~0.24 for NOISE_SIGMA=0.3), not near zero -- if this ever drops
    close to zero, the confound this experiment controls for (naive being
    "close enough on average") has disappeared and the demonstration no
    longer makes its point."""
    scm = true_scm()
    _, naive_error = run_condition(seed=0, scm=scm)
    assert naive_error > 0.15


def test_abducted_beats_naive_with_estimated_coefficients_too():
    """The realistic end-to-end case: abduction should still decisively
    beat the naive plug-in even when coefficients come from finite-sample
    OLS, not the true values."""
    fit_rows = simulate_units(seed=99, n=500)
    scm_est = fit_scm(fit_rows)
    abducted_error, naive_error = run_condition(seed=0, scm=scm_est)
    assert abducted_error < naive_error / 2


def test_abduction_error_shrinks_as_fitting_sample_size_grows():
    """Estimation error should be the dominant remaining source of
    abducted error once the true-coefficient case is exact -- more
    fitting data should shrink it."""
    small_fit = fit_scm(simulate_units(seed=1, n=100))
    large_fit = fit_scm(simulate_units(seed=1, n=5000))
    small_error, _ = run_condition(seed=0, scm=small_fit)
    large_error, _ = run_condition(seed=0, scm=large_fit)
    assert large_error < small_error
