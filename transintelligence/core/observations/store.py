"""Observation storage and querying primitives."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from transintelligence.core.observations.model import Observation

@dataclass(frozen=True)
class ObservationQuery:
    entity_id: str | None = None
    property_name: str | None = None
    source: str | None = None
    context_domain: str | None = None
    reference_frame_id: str | None = None
    start: datetime | None = None
    end: datetime | None = None
    min_confidence: float | None = None

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
        if q.property_name is not None:
            results = [o for o in results if o.property_name == q.property_name]
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
        return sorted(results, key=lambda o: o.timestamp)

    def latest(self, entity_id: str, property_name: str) -> Observation | None:
        matches = self.query(ObservationQuery(entity_id=entity_id, property_name=property_name))
        return matches[-1] if matches else None
