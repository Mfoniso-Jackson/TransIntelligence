"""Event storage and querying -- mirrors `ObservationStore`/`ObservationQuery`
(`transintelligence/core/observations/store.py`) exactly, including its
richer list-valued filters, plus `sequence_for()`: the natural
"what happened to X, in order" query an event log needs that a plain
observation log doesn't (observations aren't inherently sequential in
the same causally-ordered sense events are)."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from transintelligence.core.events.model import Event

@dataclass(frozen=True)
class EventQuery:
    subject_id: str | None = None
    subject_ids: tuple[str, ...] | None = None
    event_type: str | None = None
    event_types: tuple[str, ...] | None = None
    start: datetime | None = None
    end: datetime | None = None
    limit: int | None = None

class EventStore:
    """Simple in-memory event store, the same deterministic-retrieval
    convention `ObservationStore` already uses."""
    def __init__(self, events: list[Event] | None = None):
        self._events: list[Event] = list(events or [])

    def add(self, event: Event) -> Event:
        self._events.append(event)
        return event

    def query(self, query: EventQuery | None = None) -> list[Event]:
        q = query or EventQuery()
        results = self._events
        if q.subject_id is not None:
            results = [e for e in results if e.subject_id == q.subject_id]
        if q.subject_ids is not None:
            results = [e for e in results if e.subject_id in q.subject_ids]
        if q.event_type is not None:
            results = [e for e in results if e.event_type == q.event_type]
        if q.event_types is not None:
            results = [e for e in results if e.event_type in q.event_types]
        if q.start is not None:
            results = [e for e in results if e.timestamp >= q.start]
        if q.end is not None:
            results = [e for e in results if e.timestamp <= q.end]
        results = sorted(results, key=lambda e: e.timestamp)
        if q.limit is not None:
            results = results[:q.limit]
        return results

    def sequence_for(self, subject_id: str) -> list[Event]:
        """Chronologically-ordered events for one subject."""
        return self.query(EventQuery(subject_id=subject_id))
