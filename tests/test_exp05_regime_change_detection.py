"""Regression tests for Experiment 5 (docs/research-agenda.md #7b,
experiments/exp05_regime_change_detection/RESULTS.md). Small-scale versions
of the real experiment (fewer seeds, for test-suite speed).
"""
import statistics

from experiments.exp05_regime_change_detection.run import make_history, run_trial
from experiments.exp05_regime_change_detection.sweep_regime_length import run_trial as run_trial_regime_length


def test_low_noise_detection_is_reliable():
    rows = [run_trial(seed, noise_sigma=0.05) for seed in range(10)]
    recalls = [r["recall"] for r in rows]
    assert statistics.mean(recalls) > 0.9


def test_recall_degrades_as_noise_increases():
    """Regression guard for the graceful-degradation finding in RESULTS.md
    (contrasted there with experiment 4's sharp specificity collapse):
    recall at high noise should be meaningfully worse than at low noise,
    not comparable -- if this ever fails because both are equally good OR
    equally bad, the noise sweep needs rerunning, not just this assertion
    adjusting."""
    low_noise = statistics.mean(r["recall"] for r in (run_trial(seed, noise_sigma=0.02) for seed in range(10)))
    high_noise = statistics.mean(r["recall"] for r in (run_trial(seed, noise_sigma=0.5) for seed in range(10)))
    assert high_noise < low_noise - 0.3


def test_regime_shorter_than_burn_in_is_never_detected():
    """Regression guard for the structural finding in RESULTS.md: a regime
    length far below CUSUMTemporalReasoner's default burn_in=30 cannot be
    calibrated on before the next change happens, so recall should be
    exactly (or very close to) zero -- this is a mechanistic property, not
    a probabilistic near-miss, so the bound is tight."""
    rows = [run_trial_regime_length(seed, regime_length=10) for seed in range(10)]
    assert statistics.mean(r["recall"] for r in rows) < 0.05


def test_regime_at_or_above_burn_in_is_reliably_detected():
    rows = [run_trial_regime_length(seed, regime_length=30) for seed in range(10)]
    assert statistics.mean(r["recall"] for r in rows) > 0.9


def test_make_history_produces_expected_change_point_positions():
    _, true_change_steps = make_history(seed=0, noise_sigma=0.05, regime_means=[0.1, 0.2, 0.3], regime_length=25)
    assert true_change_steps == [25, 50]
