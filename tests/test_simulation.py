"""Tests for MonteCarloSimulator (transintelligence/simulation/model.py,
docs/research-agenda.md #7n) -- the first content in
transintelligence/simulation/, previously an empty package. Verified
against a hand-computed, deterministic (noise-free) case first, so a
directional bug can't hide behind randomness, then a stochastic case
that checks the spread reflects real noise rather than being silently
collapsed to zero.
"""
import random

import pytest

from transintelligence.simulation import MonteCarloSimulator

ACTIONS = ("up", "down")


def deterministic_transition(state, action, rng):
    return state + (1.0 if action == "up" else -1.0)


def identity_score(state):
    return state


def always(action):
    return lambda state, remaining, rng: action


def test_deterministic_rollout_matches_hand_computed_final_position():
    """No noise, 3 steps, always "up" from 0.0: every rollout must land
    on exactly 3.0, with zero spread -- the simplest possible case to
    hand-verify before trusting anything stochastic."""
    simulator = MonteCarloSimulator(n_rollouts=10)
    scores = simulator.simulate(0.0, always("up"), deterministic_transition, identity_score,
                                 horizon=3, rng=random.Random(0))
    assert scores == [3.0] * 10


def test_compare_policies_picks_the_higher_scoring_policy():
    """always-up must beat always-down by exactly 6.0 (3.0 vs -3.0),
    with zero spread on both sides under noise-free dynamics."""
    simulator = MonteCarloSimulator(n_rollouts=10)
    result = simulator.compare_policies(0.0, always("up"), always("down"),
                                         deterministic_transition, identity_score,
                                         horizon=3, rng=random.Random(0))
    assert result["mean_a"] == 3.0
    assert result["mean_b"] == -3.0
    assert result["stdev_a"] == 0.0
    assert result["stdev_b"] == 0.0
    assert result["winner"] == "a"


def test_compare_policies_verdict_is_order_invariant():
    """Swapping which policy is passed as a/b must flip the reported
    winner label, not just silently keep picking "a" regardless of which
    policy is actually better -- rules out a directional bug in
    compare_policies itself."""
    simulator = MonteCarloSimulator(n_rollouts=10)
    forward = simulator.compare_policies(0.0, always("up"), always("down"),
                                          deterministic_transition, identity_score,
                                          horizon=3, rng=random.Random(0))
    reversed_ = simulator.compare_policies(0.0, always("down"), always("up"),
                                            deterministic_transition, identity_score,
                                            horizon=3, rng=random.Random(0))
    assert forward["winner"] == "a"
    assert reversed_["winner"] == "b"


def test_stochastic_rollouts_have_nonzero_spread_reflecting_real_noise():
    """A transition_fn that actually samples noise must produce a
    nonzero stdev across rollouts -- the whole reason simulate() returns
    the full score list rather than only a mean."""
    def noisy_transition(state, action, rng):
        return deterministic_transition(state, action, rng) + rng.gauss(0.0, 1.0)

    simulator = MonteCarloSimulator(n_rollouts=200)
    scores = simulator.simulate(0.0, always("up"), noisy_transition, identity_score,
                                 horizon=3, rng=random.Random(0))
    mean = sum(scores) / len(scores)
    assert 2.5 < mean < 3.5  # true expectation is 3.0
    assert any(score != scores[0] for score in scores)  # not silently collapsed to a point estimate


def test_more_rollouts_gives_a_mean_closer_to_the_true_expectation():
    """A direct check on the Monte Carlo premise itself: more samples
    should reduce estimation error on average, not just cost more
    compute for no benefit. A single small-sample draw is too noisy to
    compare fairly against a single large-sample draw (it can get
    randomly lucky), so this averages the small-sample error over many
    independent draws before comparing."""
    def noisy_transition(state, action, rng):
        return deterministic_transition(state, action, rng) + rng.gauss(0.0, 3.0)

    true_mean = 3.0
    small_errors = []
    for seed in range(20):
        simulator = MonteCarloSimulator(n_rollouts=5)
        scores = simulator.simulate(0.0, always("up"), noisy_transition, identity_score,
                                     horizon=3, rng=random.Random(seed))
        small_errors.append(abs(sum(scores) / len(scores) - true_mean))
    large_simulator = MonteCarloSimulator(n_rollouts=2000)
    large_scores = large_simulator.simulate(0.0, always("up"), noisy_transition, identity_score,
                                             horizon=3, rng=random.Random(999))
    large_error = abs(sum(large_scores) / len(large_scores) - true_mean)
    assert large_error < sum(small_errors) / len(small_errors)


def test_policy_receives_remaining_steps_counting_down_to_one():
    """policy(state, remaining_steps, rng) must see remaining_steps
    count down from horizon to 1, matching Agent.choose_action's own
    convention (experiments 13/16) -- not, say, counting up from 0."""
    seen_remaining = []

    def recording_policy(state, remaining, rng):
        seen_remaining.append(remaining)
        return "up"

    simulator = MonteCarloSimulator(n_rollouts=1)
    simulator.simulate(0.0, recording_policy, deterministic_transition, identity_score,
                        horizon=4, rng=random.Random(0))
    assert seen_remaining == [4, 3, 2, 1]


def test_raises_on_a_non_positive_n_rollouts():
    simulator = MonteCarloSimulator(n_rollouts=0)
    with pytest.raises(ValueError):
        simulator.simulate(0.0, always("up"), deterministic_transition, identity_score,
                            horizon=1, rng=random.Random(0))


def test_horizon_is_clamped_to_at_least_one():
    simulator = MonteCarloSimulator(n_rollouts=1)
    scores = simulator.simulate(0.0, always("up"), deterministic_transition, identity_score,
                                 horizon=0, rng=random.Random(0))
    assert scores == [1.0]
