"""Regression test for Experiment 13's refit-interval calibration
follow-up (docs/research-agenda.md #7j,
experiments/exp13_regime_shift_world_model/RESULTS.md "Follow-up"
section).
"""
from experiments.exp13_regime_shift_world_model.refit_interval_calibration import run_agent
from experiments.exp13_regime_shift_world_model.run import N_TRIALS, REGIME_SHIFT_TRIAL


def test_more_frequent_refitting_reduces_false_positives_but_trades_off_detection_recall():
    """The core follow-up finding: refit_interval=10 should produce
    fewer false positives than refit_interval=20 under a stationary
    environment, but should NOT be a free lunch -- true-detection recall
    under the severe shift should be no better, and can be worse."""
    seeds = range(4)

    fp_10 = sum(1 for s in seeds if run_agent(s, 10, N_TRIALS + 1, 1.0))
    fp_20 = sum(1 for s in seeds if run_agent(s, 20, N_TRIALS + 1, 1.0))
    assert fp_10 <= fp_20

    detected_10 = sum(
        1 for s in seeds
        if any(t >= REGIME_SHIFT_TRIAL for t in run_agent(s, 10, REGIME_SHIFT_TRIAL, -1.0))
    )
    detected_20 = sum(
        1 for s in seeds
        if any(t >= REGIME_SHIFT_TRIAL for t in run_agent(s, 20, REGIME_SHIFT_TRIAL, -1.0))
    )
    assert detected_10 <= detected_20
