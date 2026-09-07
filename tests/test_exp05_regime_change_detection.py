"""Regression tests for Experiment 5 (docs/research-agenda.md #7b,
experiments/exp05_regime_change_detection/RESULTS.md). Small-scale versions
of the real experiment (fewer seeds, for test-suite speed).
"""
import statistics

from experiments.exp05_regime_change_detection.run import make_history, run_trial
from experiments.exp05_regime_change_detection.sweep_regime_length import run_trial as run_trial_regime_length
from experiments.exp05_regime_change_detection.sweep_segment_accuracy import (
    detected_segmentation_mae,
    ground_truth_segmentation_mae,
    no_segmentation_mae,
)


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


def test_detected_segmentation_beats_no_segmentation_and_trails_ground_truth():
    """Regression guard for the per-segment accuracy finding in RESULTS.md:
    detected segmentation should sit strictly between the no-segmentation
    floor and the ground-truth-segmentation ceiling, not tie either one --
    if detected ever matches no_segmentation, regime_segments() has
    stopped doing anything useful; if it ever beats ground_truth, the
    ground-truth oracle computation has a bug."""
    history, _ = make_history(seed=0, noise_sigma=0.05)
    no_seg = no_segmentation_mae(history)
    detected = detected_segmentation_mae(history)
    gt = ground_truth_segmentation_mae(history)
    assert gt <= detected < no_seg


def test_joint_detection_beats_union_of_independent_detectors_in_weak_signal_regime():
    """Regression guard for the multi-key finding in RESULTS.md: at
    shift=0.04 (weak per-key signal, ~44% single-key recall), joint
    detection should win more often than the union-of-independent-
    detectors confound control across paired seeds -- the full run (50
    seeds) found joint_wins=6 vs union_wins=1; this checks the same
    direction holds at reduced scale, not the exact counts."""
    from experiments.exp05_regime_change_detection.multi_key import run_trial
    rows = [run_trial(seed, shift=0.04) for seed in range(30)]
    joint_wins = sum(1 for r in rows if r["recall_joint"] > r["recall_union"])
    union_wins = sum(1 for r in rows if r["recall_union"] > r["recall_joint"])
    assert joint_wins > union_wins


def test_heavy_tailed_noise_inflates_false_positive_rate():
    """Regression guard for the non-Gaussian finding in RESULTS.md: a
    contaminated-Gaussian (heavy-tailed) noise process should produce a
    meaningfully higher false-positive rate than matched Gaussian noise --
    the full run (200 trials) found 0.08 vs 0.34, a >4x inflation."""
    from experiments.exp05_regime_change_detection.non_gaussian_noise import false_positive_rate
    gaussian_rate = false_positive_rate(heavy_tailed=False)
    heavy_rate = false_positive_rate(heavy_tailed=True)
    assert heavy_rate > gaussian_rate + 0.1


def test_heavy_tailed_noise_does_not_meaningfully_hurt_recall():
    """Companion to the false-positive finding: heavy tails should degrade
    specificity, not detection power on a real shift -- the full run found
    0.967 vs 1.000, not a collapse. If recall ever drops sharply here, the
    story in RESULTS.md ("this is a specificity problem, not a power
    problem") needs revisiting."""
    from experiments.exp05_regime_change_detection.non_gaussian_noise import recall
    assert recall(heavy_tailed=True) > 0.85


def test_gradual_drift_is_reliably_detected_even_when_the_ramp_never_completes():
    """Regression guard for the gradual-drift finding in RESULTS.md, which
    corrected an initial hypothesis (that CUSUM might go "blind" to slow
    drift because it recalibrates mu0/sigma once per detection cycle):
    recall should stay high even for a ramp far longer than the observed
    series, because a fixed calibration reference guarantees any
    persistent drift eventually crosses threshold."""
    from experiments.exp05_regime_change_detection.gradual_drift import run_trial
    hits = [run_trial(seed, ramp_len=300, total_len=400)[0] for seed in range(15)]
    assert sum(hits) / len(hits) > 0.85


def test_detection_delay_grows_with_ramp_length():
    from experiments.exp05_regime_change_detection.gradual_drift import run_trial
    short_delays = [d for _, d in (run_trial(seed, ramp_len=5, total_len=150) for seed in range(15)) if d == d]
    long_delays = [d for _, d in (run_trial(seed, ramp_len=80, total_len=150) for seed in range(15)) if d == d]
    import statistics
    assert statistics.mean(long_delays) > statistics.mean(short_delays)
