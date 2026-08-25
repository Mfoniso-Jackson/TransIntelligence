from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from transintelligence.core.common import Metadata, new_id, utc_now

@dataclass(frozen=True)
class Entity:
    entity_type: str
    name: str
    id: str = field(default_factory=lambda: new_id("ent"))
    metadata: Metadata = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    embedding_ref: str | None = None

    def with_metadata(self, **metadata: Any) -> "Entity":
        return Entity(self.entity_type, self.name, self.id, {**self.metadata, **metadata}, self.created_at, utc_now(), self.embedding_ref)
