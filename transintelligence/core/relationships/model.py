from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from transintelligence.core.common import Metadata, new_id, utc_now

@dataclass(frozen=True)
class Relationship:
    source_id: str
    target_id: str
    relationship_type: str
    strength: float = 1.0
    confidence: float = 1.0
    directed: bool = True
    timestamp: datetime = field(default_factory=utc_now)
    metadata: Metadata = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("rel"))

    def connects(self, a: str, b: str) -> bool:
        return (self.source_id, self.target_id) == (a, b) or (not self.directed and (self.source_id, self.target_id) == (b, a))
