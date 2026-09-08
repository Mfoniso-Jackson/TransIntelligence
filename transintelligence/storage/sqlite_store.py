"""The first real implementation of `PersistentStore` -- a stdlib-only
(`sqlite3`, no new dependency) drop-in replacement for `InMemoryStore`
(`transintelligence/memory/base.py`): identical `store()`/`retrieve()`
signatures, but records survive process restarts. Deliberately the
stdlib option first, not PostgreSQL/pgvector (docs/roadmap.md's own
"eventually") -- adding a real database driver (psycopg2/asyncpg) would
be this repo's first-ever third-party runtime dependency, a bigger
commitment than this pass makes; a PostgreSQL/pgvector backend would
need its own adapter behind the same `PersistentStore` interface,
not implemented here.

`content` is serialized via `pickle` rather than `json`: `MemoryRecord.content`
is typed `Any` and this repo's own dataclasses (`Observation`, `Event`,
`State`, ...) aren't JSON-serializable out of the box, so `pickle`
(also stdlib) is used to store arbitrary Python objects without
requiring a custom encoder. `tags` are stored as a JSON-encoded list --
plain strings, always JSON-safe, and kept human-inspectable in the
database rather than pickled alongside content.
"""
from __future__ import annotations
import json
import pickle
import sqlite3
from datetime import datetime
from typing import Any
from transintelligence.memory.base import MemoryRecord

class SQLiteMemoryStore:
    def __init__(self, path: str = ":memory:") -> None:
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS memory_records "
            "(id TEXT PRIMARY KEY, content BLOB, tags TEXT, timestamp TEXT)"
        )
        self.conn.commit()

    def store(self, content: Any, tags: tuple[str, ...] = ()) -> MemoryRecord:
        record = MemoryRecord(content, tags)
        self.conn.execute(
            "INSERT INTO memory_records (id, content, tags, timestamp) VALUES (?, ?, ?, ?)",
            (record.id, pickle.dumps(content), json.dumps(list(tags)), record.timestamp.isoformat()),
        )
        self.conn.commit()
        return record

    def retrieve(self, tag: str | None = None) -> list[MemoryRecord]:
        rows = self.conn.execute("SELECT id, content, tags, timestamp FROM memory_records ORDER BY rowid").fetchall()
        records = [
            MemoryRecord(content=pickle.loads(content_blob), tags=tuple(json.loads(tags_json)),
                         timestamp=datetime.fromisoformat(ts), id=record_id)
            for record_id, content_blob, tags_json, ts in rows
        ]
        return [r for r in records if tag is None or tag in r.tags]

    def close(self) -> None:
        self.conn.close()
