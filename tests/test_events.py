"""Tests for Event/EventStore/EventQuery (transintelligence/core/events/)
-- the first content in transintelligence/core/events/, previously
nonexistent (Event was named in the master context's original Phase 1
scope but never built).
"""
from datetime import datetime, timedelta, timezone

from transintelligence import Event
from transintelligence.core.events import EventQuery, EventStore

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _event(subject_id, event_type, offset_seconds=0, **kwargs):
    return Event(subject_id=subject_id, event_type=event_type,
                 timestamp=T0 + timedelta(seconds=offset_seconds), **kwargs)


def test_event_is_a_discrete_happening_not_a_snapshot():
    event = Event(subject_id="sensor_1", event_type="threshold_crossed", description="value exceeded 5.0")
    assert event.subject_id == "sensor_1"
    assert event.event_type == "threshold_crossed"
    assert event.caused_by == ()


def test_caused_by_records_lightweight_provenance():
    event = Event(subject_id="a", event_type="reset", caused_by=("obs_123", "event_456"))
    assert event.caused_by == ("obs_123", "event_456")


def test_query_by_subject_id_and_event_type():
    store = EventStore([
        _event("a", "regime_change", 0), _event("a", "reset", 1), _event("b", "regime_change", 2),
    ])
    results = store.query(EventQuery(subject_id="a", event_type="regime_change"))
    assert len(results) == 1
    assert results[0].subject_id == "a"
    assert results[0].event_type == "regime_change"


def test_subject_ids_and_event_types_list_filters():
    store = EventStore([_event("a", "x", 0), _event("b", "y", 1), _event("c", "z", 2)])
    results = store.query(EventQuery(subject_ids=("a", "c"), event_types=("x", "z")))
    assert {e.subject_id for e in results} == {"a", "c"}


def test_query_time_range_and_limit():
    store = EventStore([_event("a", "x", offset_seconds=i) for i in range(5)])
    results = store.query(EventQuery(start=T0 + timedelta(seconds=1), end=T0 + timedelta(seconds=3)))
    assert len(results) == 3
    limited = store.query(EventQuery(limit=2))
    assert len(limited) == 2


def test_sequence_for_returns_chronological_order_for_one_subject():
    store = EventStore([
        _event("a", "y", offset_seconds=2), _event("b", "x", offset_seconds=0), _event("a", "x", offset_seconds=1),
    ])
    sequence = store.sequence_for("a")
    assert [e.event_type for e in sequence] == ["x", "y"]
