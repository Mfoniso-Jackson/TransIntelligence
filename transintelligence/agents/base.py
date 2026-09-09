from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from transintelligence.memory.base import InMemoryStore
from transintelligence.core.events import Event, EventStore

@dataclass
class AgentStep:
    observation: Any
    representation: Any
    reasoning: Any
    plan: str
    action: str
    outcome: Any

class KernelAgent:
    """`memory` accepts anything satisfying `PersistentStore`'s shape
    (`transintelligence/storage/interfaces.py`) -- `InMemoryStore` (the
    default) or `SQLiteMemoryStore` (`transintelligence/storage/`) both
    conform structurally, so this class doesn't need to know or care
    which one it has. `events`, if given, additionally records a
    `"step_recorded"` `Event` per `run_once()` call through `EventStore`
    (`transintelligence/core/events/`) -- the first place these three
    Later-milestone additions (event reasoning, persistent storage) are
    actually wired into the one place they'd naturally combine, rather
    than sitting as isolated, self-tested modules."""

    def __init__(self, memory: InMemoryStore | None = None, events: EventStore | None = None,
                 agent_id: str = "kernel_agent") -> None:
        self.memory = memory or InMemoryStore()
        self.events = events
        self.agent_id = agent_id

    def run_once(self, observation: Any) -> AgentStep:
        step = AgentStep(observation, observation, "deterministic baseline reasoning", "record observation", "store", {"stored": True})
        self.memory.store(step, tags=("agent_step",))
        if self.events is not None:
            self.events.add(Event(subject_id=self.agent_id, event_type="step_recorded",
                                   description="KernelAgent.run_once stored a new AgentStep",
                                   metadata={"action": step.action}))
        return step
