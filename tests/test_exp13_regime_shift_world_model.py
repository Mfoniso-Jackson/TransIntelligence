"""Regression tests for Experiment 13 (docs/research-agenda.md #7j,
experiments/exp13_regime_shift_world_model/RESULTS.md). Small-scale
versions of the real experiment (fewer trials/seeds, for test-suite
speed).
"""
import statistics

from experiments.exp13_regime_shift_world_model.run import (
    CONDITIONS,
    REGIME_SHIFT_TRIAL,
    run_condition,
)


def test_never_adapts_collapses_under_a_severe_sign_flip_shift():
    seeds = range(6)
    results = [run_condition(CONDITIONS["never_adapts"], s, post_shift_scale=-1.0) for s in seeds]
    pre_rewards = [r[0] for r in results]
    post_rewards = [r[1] for r in results]
    # post-shift reward should be dramatically worse than pre-shift for never_adapts here
    assert statistics.mean(post_rewards) < statistics.mean(pre_rewards) - 5.0


def test_cusum_detects_and_adapts_beats_sliding_window_baseline_under_sign_flip():
    """The core confound-controlled claim: explicit detection must beat
    a naive always-use-recent-data heuristic, not just beat doing
    nothing."""
    seeds = range(6)
    cusum = statistics.mean(run_condition(CONDITIONS["cusum_detects_and_adapts"], s, post_shift_scale=-1.0)[1] for s in seeds)
    sliding = statistics.mean(run_condition(CONDITIONS["sliding_window_baseline"], s, post_shift_scale=-1.0)[1] for s in seeds)
    never = statistics.mean(run_condition(CONDITIONS["never_adapts"], s, post_shift_scale=-1.0)[1] for s in seeds)
    assert cusum > sliding
    assert cusum > never + 5.0


def test_cusum_reliably_detects_a_severe_sign_flip_shift():
    seeds = range(6)
    detected = 0
    for s in seeds:
        _, _, reset_trials = run_condition(CONDITIONS["cusum_detects_and_adapts"], s, post_shift_scale=-1.0)
        if any(t >= REGIME_SHIFT_TRIAL for t in reset_trials):
            detected += 1
    assert detected >= len(list(seeds)) - 1  # allow at most one miss


def test_mild_attenuation_does_not_show_a_dramatic_adaptation_advantage():
    """Regression guard for the honest boundary finding: a mild shift
    that doesn't flip which action looks best should NOT produce a
    dramatic gap between never_adapts and oracle_adapts, unlike the
    severe sign-flip case."""
    seeds = range(6)
    never = statistics.mean(run_condition(CONDITIONS["never_adapts"], s, post_shift_scale=0.4)[1] for s in seeds)
    oracle = statistics.mean(run_condition(CONDITIONS["oracle_adapts"], s, post_shift_scale=0.4)[1] for s in seeds)
    assert abs(never - oracle) < 1.0
