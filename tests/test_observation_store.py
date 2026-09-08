"""Tests for ObservationStore/ObservationQuery
(transintelligence/core/observations/store.py) -- covers both the
original single-value filters and the richer additions (list-valued
filters, max_confidence, limit, latest_per_entity, group_by_entity).
"""
from datetime import datetime, timedelta, timezone

from transintelligence.core.observations import Observation, ObservationQuery, ObservationStore

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _obs(entity_id, property_name, value, confidence=1.0, offset_seconds=0):
    return Observation(entity_id=entity_id, property_name=property_name, value=value,
                        source="test", confidence=confidence, timestamp=T0 + timedelta(seconds=offset_seconds))


def test_entity_ids_filter_matches_any_of_several_entities():
    store = ObservationStore([_obs("a", "x", 1), _obs("b", "x", 2), _obs("c", "x", 3)])
    results = store.query(ObservationQuery(entity_ids=("a", "c")))
    assert {o.entity_id for o in results} == {"a", "c"}


def test_property_names_filter_matches_any_of_several_properties():
    store = ObservationStore([_obs("a", "x", 1), _obs("a", "y", 2), _obs("a", "z", 3)])
    results = store.query(ObservationQuery(property_names=("x", "z")))
    assert {o.property_name for o in results} == {"x", "z"}


def test_entity_id_and_entity_ids_combine_with_and_semantics():
    """A contradictory combination (entity_id not in entity_ids) should
    correctly return nothing, not silently ignore one of the filters."""
    store = ObservationStore([_obs("a", "x", 1), _obs("b", "x", 2)])
    results = store.query(ObservationQuery(entity_id="a", entity_ids=("b",)))
    assert results == []
    results = store.query(ObservationQuery(entity_id="a", entity_ids=("a", "b")))
    assert len(results) == 1


def test_max_confidence_filters_out_high_confidence_observations():
    store = ObservationStore([_obs("a", "x", 1, confidence=0.9), _obs("a", "x", 2, confidence=0.3)])
    results = store.query(ObservationQuery(max_confidence=0.5))
    assert len(results) == 1
    assert results[0].confidence == 0.3


def test_limit_caps_results_after_sorting_by_timestamp():
    store = ObservationStore([_obs("a", "x", i, offset_seconds=i) for i in range(5)])
    results = store.query(ObservationQuery(limit=2))
    assert [o.value for o in results] == [0, 1]  # earliest two, not an arbitrary two


def test_latest_per_entity_returns_the_most_recent_observation_for_each():
    store = ObservationStore([
        _obs("a", "x", "a_old", offset_seconds=0), _obs("a", "x", "a_new", offset_seconds=10),
        _obs("b", "x", "b_only", offset_seconds=5),
    ])
    result = store.latest_per_entity(["a", "b", "c"], "x")
    assert result["a"].value == "a_new"
    assert result["b"].value == "b_only"
    assert result["c"] is None  # no observations at all for c


def test_group_by_entity_groups_and_preserves_timestamp_order_within_each_group():
    store = ObservationStore([
        _obs("a", "x", "a1", offset_seconds=1), _obs("b", "x", "b1", offset_seconds=0),
        _obs("a", "x", "a2", offset_seconds=2),
    ])
    grouped = store.group_by_entity()
    assert set(grouped.keys()) == {"a", "b"}
    assert [o.value for o in grouped["a"]] == ["a1", "a2"]
    assert [o.value for o in grouped["b"]] == ["b1"]
