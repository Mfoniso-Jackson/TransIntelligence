from __future__ import annotations
from dataclasses import dataclass, field, replace
from datetime import datetime
from transintelligence.core.common import Metadata

@dataclass(frozen=True)
class Context:
    domain: str | None = None
    time: datetime | None = None
    location: str | None = None
    observer: str | None = None
    objective: str | None = None
    scale: str | None = None
    assumptions: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    metadata: Metadata = field(default_factory=dict)

    def compose(self, other: "Context") -> "Context":
        return Context(
            domain=other.domain or self.domain, time=other.time or self.time,
            location=other.location or self.location, observer=other.observer or self.observer,
            objective=other.objective or self.objective, scale=other.scale or self.scale,
            assumptions=(*self.assumptions, *other.assumptions),
            constraints=(*self.constraints, *other.constraints),
            metadata={**self.metadata, **other.metadata},
        )
