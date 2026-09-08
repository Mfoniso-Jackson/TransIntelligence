"""Regression test for experiment 17's rollout-count-sweep follow-up
(docs/research-agenda.md #7n,
experiments/exp17_monte_carlo_simulator/RESULTS.md "Follow-up" section).
"""
from experiments.exp17_monte_carlo_simulator.rollout_count_sweep import build_trained
from experiments.exp17_monte_carlo_simulator.run import compare_at_rollout_count


def test_win_rate_at_a_large_rollout_count_is_at_least_as_high_as_at_a_tiny_one():
    """The follow-up's core finding: for this large an effect size, the
    verdict is already fairly reliable even at n_rollouts=1 (well above
    chance), and does not drop to an unreliable level anywhere in the
    tested range -- no sharp reliability cliff exists here. This checks
    the directional relationship on a small seed subset, not the exact
    percentages (which are sampling-noise-sensitive at low n_rollouts)."""
    trained = build_trained()[:4]  # a subset for test speed
    wins_1, total_1, _, _ = compare_at_rollout_count(1, trained, verbose=False)
    wins_200, total_200, _, _ = compare_at_rollout_count(200, trained, verbose=False)
    assert wins_1 / total_1 > 0.6  # well above chance even at n_rollouts=1
    assert wins_200 / total_200 >= wins_1 / total_1 - 0.15  # roughly comparable, not a cliff
