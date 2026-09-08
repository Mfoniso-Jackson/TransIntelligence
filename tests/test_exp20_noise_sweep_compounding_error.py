"""Regression tests for Experiment 20 (docs/research-agenda.md #7q,
experiments/exp20_noise_sweep_compounding_error/RESULTS.md).
"""
import statistics

from experiments.exp20_noise_sweep_compounding_error.run import CONDITIONS, REGIME_SHIFT_STEP, run_condition

SEEDS = range(5)


def test_gap_is_much_smaller_at_zero_noise_than_at_the_original_noise_level():
    """The core confirmation: experiment 16's persistent post-shift gap
    should shrink dramatically toward zero as environment noise -> 0,
    if it's really driven by compounding estimation error -- a
    deterministic environment gives a correctly-specified model nothing
    to compound."""
    greedy_zero = statistics.mean(run_condition(CONDITIONS["greedy_cusum_adapts"], s, noise_sigma=0.0)[1] for s in SEEDS)
    mpc_zero = statistics.mean(run_condition(CONDITIONS["mpc_beam_cusum_adapts"], s, noise_sigma=0.0)[1] for s in SEEDS)
    gap_zero = greedy_zero - mpc_zero

    greedy_orig = statistics.mean(run_condition(CONDITIONS["greedy_cusum_adapts"], s, noise_sigma=0.1)[1] for s in SEEDS)
    mpc_orig = statistics.mean(run_condition(CONDITIONS["mpc_beam_cusum_adapts"], s, noise_sigma=0.1)[1] for s in SEEDS)
    gap_orig = greedy_orig - mpc_orig

    assert gap_zero < gap_orig / 2


def test_detection_reliability_is_similar_between_agents_at_each_noise_level():
    """Confound control: if greedy detected the shift far more reliably
    than mpc at some noise level, that alone could explain a reward gap
    without any estimation-error compounding at all. It doesn't --
    detection counts should track closely between the two agents."""
    for noise_sigma in (0.05, 0.2):
        greedy_detected = sum(
            1 for s in SEEDS
            if any(t >= REGIME_SHIFT_STEP for t in run_condition(CONDITIONS["greedy_cusum_adapts"], s, noise_sigma)[2])
        )
        mpc_detected = sum(
            1 for s in SEEDS
            if any(t >= REGIME_SHIFT_STEP for t in run_condition(CONDITIONS["mpc_beam_cusum_adapts"], s, noise_sigma)[2])
        )
        assert abs(greedy_detected - mpc_detected) <= 2
