"""Regression tests for Experiment 16 (docs/research-agenda.md #7m,
experiments/exp16_regime_shift_multistep_planning/RESULTS.md).
"""
import statistics

from experiments.exp16_regime_shift_multistep_planning.run import CONDITIONS, run_condition


def test_exhaustive_multistep_search_reproduces_experiment_14s_oscillation_pathology():
    """Independent replication, in a new environment built for a
    different purpose: exhaustive multi-step search under receding-
    horizon replanning should perform far worse than beam search,
    exactly as experiment 14 found."""
    seeds = range(4)
    exhaustive_post = statistics.mean(run_condition(CONDITIONS["mpc_exhaustive_never_adapts"], s)[1] for s in seeds)
    beam_post = statistics.mean(run_condition(CONDITIONS["mpc_beam_never_adapts"], s)[1] for s in seeds)
    assert beam_post > exhaustive_post


def test_oracle_beam_matches_pre_shift_quality_after_the_shift():
    """Beam search, given the true dynamics exactly, should be immune to
    the oscillation pathology -- post-shift performance should be close
    to pre-shift, not degraded."""
    seeds = range(4)
    results = [run_condition(CONDITIONS["oracle_beam"], s) for s in seeds]
    pre = statistics.mean(r[0] for r in results)
    post = statistics.mean(r[1] for r in results)
    assert abs(post - pre) < 0.5


def test_greedy_cusum_adapts_beats_mpc_beam_cusum_adapts_post_shift():
    """The core, counterintuitive finding: multi-step planning (even
    with beam search, which fixes the pure search-quality pathology)
    persistently underperforms single-step greedy planning once the
    dynamics model must be LEARNED and has just been reset by CUSUM
    detection -- not predictable from experiments 12, 13, or 14 tested
    in isolation."""
    seeds = range(6)
    greedy_post = statistics.mean(run_condition(CONDITIONS["greedy_cusum_adapts"], s)[1] for s in seeds)
    mpc_post = statistics.mean(run_condition(CONDITIONS["mpc_beam_cusum_adapts"], s)[1] for s in seeds)
    assert greedy_post > mpc_post + 5.0
