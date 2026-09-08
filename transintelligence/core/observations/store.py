"""Observation storage and querying primitives."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from transintelligence.core.observations.model import Observation

@dataclass(frozen=True)
class ObservationQuery:
    entity_id: str | None = None
    entity_ids: tuple[str, ...] | None = None
    property_name: str | None = None
    property_names: tuple[str, ...] | None = None
    source: str | None = None
    context_domain: str | None = None
    reference_frame_id: str | None = None
    start: datetime | None = None
    end: datetime | None = None
    min_confidence: float | None = None
    max_confidence: float | None = None
    limit: int | None = None

class ObservationStore:
    """Simple in-memory observation store for deterministic Phase-1 retrieval."""
    def __init__(self, observations: list[Observation] | None = None):
        self._observations: list[Observation] = list(observations or [])

    def add(self, observation: Observation) -> Observation:
        self._observations.append(observation)
        return observation

    def query(self, query: ObservationQuery | None = None) -> list[Observation]:
        q = query or ObservationQuery()
        results = self._observations
        if q.entity_id is not None:
            results = [o for o in results if o.entity_id == q.entity_id]
        if q.entity_ids is not None:
            results = [o for o in results if o.entity_id in q.entity_ids]
        if q.property_name is not None:
            results = [o for o in results if o.property_name == q.property_name]
        if q.property_names is not None:
            results = [o for o in results if o.property_name in q.property_names]
        if q.source is not None:
            results = [o for o in results if o.source == q.source]
        if q.context_domain is not None:
            results = [o for o in results if o.context is not None and o.context.domain == q.context_domain]
        if q.reference_frame_id is not None:
            results = [o for o in results if o.reference_frame is not None and o.reference_frame.id == q.reference_frame_id]
        if q.start is not None:
            results = [o for o in results if o.timestamp >= q.start]
        if q.end is not None:
            results = [o for o in results if o.timestamp <= q.end]
        if q.min_confidence is not None:
            results = [o for o in results if o.confidence >= q.min_confidence]
        if q.max_confidence is not None:
            results = [o for o in results if o.confidence <= q.max_confidence]
        results = sorted(results, key=lambda o: o.timestamp)
        if q.limit is not None:
            results = results[:q.limit]
        return results

    def latest(self, entity_id: str, property_name: str) -> Observation | None:
        matches = self.query(ObservationQuery(entity_id=entity_id, property_name=property_name))
        return matches[-1] if matches else None

    def latest_per_entity(self, entity_ids: list[str], property_name: str) -> dict[str, Observation | None]:
        """The natural multi-entity generalization of `latest()` -- one
        query per entity rather than requiring the caller to loop and
        call `latest()` themselves for every entity they care about."""
        return {entity_id: self.latest(entity_id, property_name) for entity_id in entity_ids}

    def group_by_entity(self, query: ObservationQuery | None = None) -> dict[str, list[Observation]]:
        """Groups a query's results by `entity_id`, each sub-list still
        ordered by timestamp -- the minimal aggregation `query()` alone
        can't express (a single flat list mixes all matching entities
        together)."""
        grouped: dict[str, list[Observation]] = {}
        for observation in self.query(query):
            grouped.setdefault(observation.entity_id, []).append(observation)
        return grouped
