from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from transintelligence.memory.base import InMemoryStore

@dataclass
class AgentStep:
    observation: Any
    representation: Any
    reasoning: Any
    plan: str
    action: str
    outcome: Any

class KernelAgent:
    def __init__(self, memory: InMemoryStore | None = None): self.memory = memory or InMemoryStore()
    def run_once(self, observation: Any) -> AgentStep:
        step = AgentStep(observation, observation, "deterministic baseline reasoning", "record observation", "store", {"stored": True})
        self.memory.store(step, tags=("agent_step",)); return step
