"""Shared typed primitives for TransIntelligence."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)

Metadata = dict[str, Any]

@dataclass(frozen=True)
class TimeWindow:
    start: datetime | None = None
    end: datetime | None = None
