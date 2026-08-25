from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from transintelligence.core.common import Metadata, new_id, utc_now

@dataclass(frozen=True)
class State:
    subject_id: str
    values: dict[str, Any]
    timestamp: datetime = field(default_factory=utc_now)
    previous_state_id: str | None = None
    metadata: Metadata = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("state"))

    def transition_to(self, values: dict[str, Any], timestamp: datetime | None = None, **metadata: Any) -> "State":
        return State(self.subject_id, values, timestamp or utc_now(), self.id, metadata)

class StateHistory:
    def __init__(self, states: list[State] | None = None):
        self.states = sorted(states or [], key=lambda s: s.timestamp)
    def add(self, state: State) -> None:
        self.states.append(state); self.states.sort(key=lambda s: s.timestamp)
    def state_at(self, t: datetime) -> State | None:
        candidates = [s for s in self.states if s.timestamp <= t]
        return candidates[-1] if candidates else None
    def trajectory(self, t0: datetime, t1: datetime) -> list[State]:
        return [s for s in self.states if t0 <= s.timestamp <= t1]
