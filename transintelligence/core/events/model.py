"""Event: a discrete happening, as distinct from `State` (a snapshot of
what something *is* at a point in time). Named in the master context's
own original Phase 1 primitive list (`docs/master-context.md` §19:
"Entity, Relationship, Observation, Event, State, ...") alongside
`State`, but never built -- only `State`/`StateHistory` were, leaving
this a genuine gap rather than new scope.

`CUSUMTemporalReasoner.change_points()` (`transintelligence/reasoning/temporal/`)
already detects moments something changed, but returns bare `datetime`s
-- discarding everything about *what* changed, for *whom*, or why, the
instant the caller looks away. `Event` is the first-class record of that
kind of moment; `events_from_change_points()`
(`transintelligence/reasoning/temporal/model.py`) is the bridge that
turns a detected change point into one.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from transintelligence.core.common import Metadata, new_id, utc_now
from transintelligence.core.contexts import Context

@dataclass(frozen=True)
class Event:
    subject_id: str
    event_type: str
    timestamp: datetime = field(default_factory=utc_now)
    description: str = ""
    caused_by: tuple[str, ...] = ()  # ids of prior events/observations/states that led to this one -- a
                                      # lightweight provenance link, not a causal graph (reasoning/causal/'s job)
    context: Context | None = None
    metadata: Metadata = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("event"))
