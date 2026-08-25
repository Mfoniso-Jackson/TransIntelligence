from __future__ import annotations
from dataclasses import dataclass, field
from transintelligence.core.common import Metadata, new_id, utc_now
from datetime import datetime

@dataclass(frozen=True)
class Source:
    name: str
    uri: str | None = None
    reliability: float = 0.5
@dataclass(frozen=True)
class Confidence:
    value: float
    rationale: str = ""
@dataclass(frozen=True)
class Evidence:
    content: str
    source: Source
    observation_id: str | None = None
    confidence: Confidence = field(default_factory=lambda: Confidence(0.5))
    id: str = field(default_factory=lambda: new_id("ev"))
@dataclass(frozen=True)
class CounterEvidence(Evidence): pass
@dataclass(frozen=True)
class Claim:
    statement: str
    evidence: tuple[Evidence, ...] = ()
    counter_evidence: tuple[CounterEvidence, ...] = ()
    confidence: Confidence = field(default_factory=lambda: Confidence(0.0, "unsupported until evidence is linked"))
    id: str = field(default_factory=lambda: new_id("claim"))
    def is_supported(self) -> bool: return bool(self.evidence)
@dataclass(frozen=True)
class Hypothesis(Claim):
    status: str = "proposed"
