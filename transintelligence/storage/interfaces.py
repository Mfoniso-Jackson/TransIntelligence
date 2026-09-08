"""The replaceable-interface stub for persistence (docs/roadmap.md's
"Persistent storage adapters, eventually PostgreSQL/pgvector"),
mirroring `transintelligence/reasoning/interfaces.py`'s `Protocol`
pattern -- the only other place in this repo that expresses "here is
the shape a real implementation must have," rather than a concrete
class.

`PersistentStore` matches `InMemoryStore`'s existing shape
(`transintelligence/memory/base.py`) exactly, so any conforming backend
(the stdlib `sqlite3` one in `sqlite_store.py`, or a future
PostgreSQL/pgvector one) is a drop-in replacement for the in-memory
episodic/semantic/strategic namespaces -- callers holding a
`PersistentStore` don't need to know or care which one they have.
"""
from __future__ import annotations
from typing import Any, Protocol
from transintelligence.memory.base import MemoryRecord

class PersistentStore(Protocol):
    def store(self, content: Any, tags: tuple[str, ...] = ()) -> MemoryRecord: ...
    def retrieve(self, tag: str | None = None) -> list[MemoryRecord]: ...
