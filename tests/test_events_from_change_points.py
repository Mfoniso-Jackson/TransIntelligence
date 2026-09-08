"""Tests for events_from_change_points (transintelligence/reasoning/temporal/model.py)
-- the bridge from CUSUMTemporalReasoner's detected change points to
first-class Event records (transintelligence/core/events/).
"""
from datetime import datetime, timedelta, timezone

from transintelligence import State, StateHistory
from transintelligence.reasoning.temporal import CUSUMTemporalReasoner, events_from_change_points

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _states(values: list[float], key: str = "v", start: datetime = T0, step_minutes: int = 1) -> list[State]:
    return [State("x", {key: v}, start + timedelta(minutes=i * step_minutes)) for i, v in enumerate(values)]


def test_events_from_change_points_matches_change_points_exactly():
    """The same clean, unambiguous mean-shift case
    test_temporal_reasoning.py's test_change_points_detects_a_clean_mean_shift
    already verified -- the bridge must produce exactly one Event, at the
    exact same timestamp change_points() itself reports, not an
    approximation."""
    values = [0.3] * 30 + [0.9] * 30
    history = StateHistory(_states(values))
    reasoner = CUSUMTemporalReasoner(burn_in=10, k_sigma=0.5, h_sigma=5.0, min_sigma=1e-6)
    change_ts = reasoner.change_points(history, "v")
    events = events_from_change_points(history, "v", reasoner)
    assert len(events) == len(change_ts) == 1
    assert events[0].timestamp == change_ts[0]


def test_events_carry_subject_id_type_and_key():
    values = [0.3] * 30 + [0.9] * 30
    history = StateHistory(_states(values))
    reasoner = CUSUMTemporalReasoner(burn_in=10, k_sigma=0.5, h_sigma=5.0, min_sigma=1e-6)
    events = events_from_change_points(history, "v", reasoner)
    assert events[0].subject_id == "x"  # inherited from the states' own subject_id
    assert events[0].event_type == "regime_change"
    assert events[0].metadata["key"] == "v"


def test_explicit_subject_id_overrides_the_states_own_subject():
    values = [0.3] * 30 + [0.9] * 30
    history = StateHistory(_states(values))
    reasoner = CUSUMTemporalReasoner(burn_in=10, k_sigma=0.5, h_sigma=5.0, min_sigma=1e-6)
    events = events_from_change_points(history, "v", reasoner, subject_id="sensor_7")
    assert events[0].subject_id == "sensor_7"


def test_no_change_points_gives_no_events():
    values = [0.3] * 30  # stationary -- no shift to detect
    history = StateHistory(_states(values))
    reasoner = CUSUMTemporalReasoner(burn_in=10, k_sigma=0.5, h_sigma=5.0, min_sigma=1e-6)
    assert events_from_change_points(history, "v", reasoner) == []
