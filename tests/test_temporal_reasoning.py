"""Tests for CUSUMTemporalReasoner (transintelligence/reasoning/temporal/model.py,
docs/research-agenda.md #7b) -- the first non-trivial content in
transintelligence/reasoning/temporal/, previously just a docstring stub.
"""
from datetime import datetime, timedelta, timezone

import pytest

from transintelligence import State, StateHistory
from transintelligence.reasoning.temporal import CUSUMTemporalReasoner, RegimeSegment, dynamic_time_warp

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _states(values: list[float], key: str = "v", start: datetime = T0, step_minutes: int = 1) -> list[State]:
    return [State("x", {key: v}, start + timedelta(minutes=i * step_minutes)) for i, v in enumerate(values)]


def test_change_points_detects_a_clean_mean_shift():
    values = [0.3] * 30 + [0.9] * 30  # noiseless, unambiguous shift
    history = StateHistory(_states(values))
    reasoner = CUSUMTemporalReasoner(burn_in=10, k_sigma=0.5, h_sigma=5.0, min_sigma=1e-6)
    changes = reasoner.change_points(history, "v")
    assert len(changes) == 1
    detected_index = (changes[0] - T0).total_seconds() / 60
    assert 28 <= detected_index <= 32  # within a few steps of the true shift at index 30


def test_change_points_rarely_fires_on_a_stable_series_at_default_settings():
    # A monotonic drift, even a small one, is a real, persistent level
    # change from CUSUM's perspective and *should* eventually trigger --
    # this tests genuine stationarity (noise around a fixed mean) instead.
    # CUSUM is a statistical test with a nonzero false-alarm rate by design;
    # the class defaults were calibrated to ~6% over 200 trials (see the
    # model.py docstring and experiments/exp05_regime_change_detection/
    # RESULTS.md) -- this checks that calibration holds, using the actual
    # defaults rather than hardcoded parameters that could silently drift
    # out of sync with them.
    import random
    false_positives = 0
    trials = 30
    for seed in range(trials):
        rng = random.Random(seed)
        values = [0.5 + rng.gauss(0, 0.02) for _ in range(100)]
        history = StateHistory(_states(values))
        reasoner = CUSUMTemporalReasoner()  # class defaults, not hardcoded values
        if reasoner.change_points(history, "v"):
            false_positives += 1
    assert false_positives / trials < 0.2


def test_change_points_ignores_non_numeric_and_missing_values():
    states = [
        State("x", {"v": "not_a_number"}, T0),
        State("x", {"other_key": 1.0}, T0 + timedelta(minutes=1)),
        State("x", {"v": True}, T0 + timedelta(minutes=2)),  # bool excluded even though isinstance(bool, int)
    ] + _states([0.3] * 15, start=T0 + timedelta(minutes=3))
    history = StateHistory(states)
    reasoner = CUSUMTemporalReasoner(burn_in=10)
    # Should run without error and only use the trailing numeric-`v` states.
    assert reasoner.change_points(history, "v") == []


def test_change_points_raises_on_too_small_burn_in():
    with pytest.raises(ValueError):
        CUSUMTemporalReasoner(burn_in=1)


def test_regime_segments_partition_the_full_history_with_correct_means():
    values = [0.2] * 20 + [0.8] * 20 + [0.5] * 20
    history = StateHistory(_states(values))
    reasoner = CUSUMTemporalReasoner(burn_in=10, k_sigma=0.5, h_sigma=5.0, min_sigma=1e-6)
    segments = reasoner.regime_segments(history, "v")

    assert len(segments) == 3
    assert sum(len(seg.states) for seg in segments) == 60  # every state accounted for, no overlap
    for seg in segments:
        assert isinstance(seg, RegimeSegment)
    assert segments[0].mean_value == pytest.approx(0.2, abs=0.01)
    assert segments[1].mean_value == pytest.approx(0.8, abs=0.01)
    assert segments[2].mean_value == pytest.approx(0.5, abs=0.01)


def test_regime_segments_empty_history_returns_no_segments():
    reasoner = CUSUMTemporalReasoner(burn_in=10)
    assert reasoner.regime_segments(StateHistory([]), "v") == []


def test_compare_returns_only_differing_keys():
    a = State("x", {"regime": "calm", "volatility": 0.3})
    b = State("x", {"regime": "volatile", "volatility": 0.3, "new_key": 1})
    reasoner = CUSUMTemporalReasoner(burn_in=10)
    diff = reasoner.compare(a, b)
    assert diff == {"regime": ("calm", "volatile"), "new_key": (None, 1)}
    assert "volatility" not in diff


def test_dynamic_time_warp_identical_sequences_have_zero_distance():
    assert dynamic_time_warp([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_dynamic_time_warp_recognizes_same_shape_despite_different_length():
    a = [0.2, 0.2, 0.8, 0.8, 0.8, 0.2, 0.2]
    b = [0.2, 0.8, 0.8, 0.8, 0.2]  # same shape, compressed
    assert dynamic_time_warp(a, b) == 0.0


def test_dynamic_time_warp_beats_naive_pointwise_comparison_under_time_shift():
    """The motivating case for DTW (Sakoe & Chiba 1978, docs/related-work.md
    §9a), constructed directly: a shifted-but-identically-shaped bump is
    ranked as MORE similar to the template by naive pointwise comparison
    than a genuinely different-shaped one at the same position -- backwards
    -- while DTW ranks correctly. See
    experiments/exp05_regime_change_detection/dtw_comparison.py for the
    full shift-magnitude sweep this is drawn from (shift=5 is comfortably
    inside the range where naive fails but DTW doesn't)."""
    template = [0.2] * 10 + [0.8] * 10 + [0.2] * 10
    shifted = [0.2] * 15 + [0.8] * 10 + [0.2] * 5      # same bump, delayed onset
    different = [0.2] * 10 + [0.6] * 10 + [0.2] * 10   # same position, different height

    def naive(x, y):
        return sum(abs(p - q) for p, q in zip(x, y))

    assert dynamic_time_warp(template, shifted) <= dynamic_time_warp(template, different)
    assert naive(template, shifted) > naive(template, different)  # naive gets it backwards


def test_dynamic_time_warp_raises_on_empty_series():
    with pytest.raises(ValueError):
        dynamic_time_warp([], [1.0])


def test_trajectory_distance_matches_dynamic_time_warp_on_the_underlying_series():
    history_a = StateHistory(_states([0.2, 0.2, 0.8, 0.8, 0.8, 0.2, 0.2]))
    history_b = StateHistory(_states([0.2, 0.8, 0.8, 0.8, 0.2]))
    reasoner = CUSUMTemporalReasoner(burn_in=10)
    assert reasoner.trajectory_distance(history_a, history_b, "v") == 0.0
