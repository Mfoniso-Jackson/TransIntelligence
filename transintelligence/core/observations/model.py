from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from transintelligence.core.common import Metadata, new_id, utc_now
from transintelligence.core.contexts import Context
from transintelligence.representation.reference_frames import ReferenceFrame

@dataclass(frozen=True)
class Observation:
    entity_id: str
    property_name: str
    value: Any
    source: str
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=utc_now)
    context: Context | None = None
    reference_frame: ReferenceFrame | None = None
    metadata: Metadata = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("obs"))
