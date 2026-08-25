from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from transintelligence.core.common import new_id, utc_now
from datetime import datetime
@dataclass(frozen=True)
class MemoryRecord:
    content: Any
    tags: tuple[str, ...] = ()
    timestamp: datetime = field(default_factory=utc_now)
    id: str = field(default_factory=lambda: new_id("mem"))
class InMemoryStore:
    def __init__(self): self.records: list[MemoryRecord] = []
    def store(self, content: Any, tags: tuple[str, ...] = ()) -> MemoryRecord:
        rec = MemoryRecord(content, tags); self.records.append(rec); return rec
    def retrieve(self, tag: str | None = None) -> list[MemoryRecord]:
        return [r for r in self.records if tag is None or tag in r.tags]
