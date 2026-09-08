"""Tests for SQLiteMemoryStore (transintelligence/storage/sqlite_store.py)
-- the first real implementation of PersistentStore
(transintelligence/storage/interfaces.py), a stdlib-only drop-in
replacement for InMemoryStore (transintelligence/memory/base.py).
"""
import os
import tempfile

from transintelligence.memory.base import InMemoryStore
from transintelligence.storage import SQLiteMemoryStore


def test_store_and_retrieve_matches_in_memory_store_behavior():
    """Same content, same tags, same operations -- SQLiteMemoryStore
    should behave identically to InMemoryStore for a single session,
    not just "eventually consistent" in some looser sense."""
    in_memory = InMemoryStore()
    sqlite_store = SQLiteMemoryStore()
    for store in (in_memory, sqlite_store):
        store.store({"note": "first"}, tags=("a", "b"))
        store.store({"note": "second"}, tags=("b",))
        store.store({"note": "third"}, tags=("c",))

    assert [r.content for r in in_memory.retrieve("b")] == [r.content for r in sqlite_store.retrieve("b")]
    assert [r.content for r in in_memory.retrieve()] == [r.content for r in sqlite_store.retrieve()]
    assert [r.content for r in in_memory.retrieve("c")] == [r.content for r in sqlite_store.retrieve("c")]


def test_retrieve_preserves_insertion_order():
    store = SQLiteMemoryStore()
    store.store("first")
    store.store("second")
    store.store("third")
    assert [r.content for r in store.retrieve()] == ["first", "second", "third"]


def test_retrieve_with_no_tag_returns_everything():
    store = SQLiteMemoryStore()
    store.store("a", tags=("x",))
    store.store("b", tags=("y",))
    assert len(store.retrieve()) == 2


def test_arbitrary_python_objects_round_trip_via_pickle():
    """MemoryRecord.content is typed Any -- a dict, a list, or a plain
    object should all round-trip exactly, not just JSON-friendly types."""
    store = SQLiteMemoryStore()
    payload = {"nested": [1, 2, {"three": 3.0}], "flag": True, "none": None}
    store.store(payload, tags=("payload",))
    retrieved = store.retrieve("payload")
    assert len(retrieved) == 1
    assert retrieved[0].content == payload


def test_records_survive_across_a_reconnect_to_the_same_file():
    """The actual point of this class: data must survive a process
    restart, not just live for the current connection's lifetime."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "memory.sqlite3")

        store_a = SQLiteMemoryStore(db_path)
        store_a.store({"episode": 1}, tags=("episodic",))
        store_a.close()

        store_b = SQLiteMemoryStore(db_path)
        retrieved = store_b.retrieve("episodic")
        assert len(retrieved) == 1
        assert retrieved[0].content == {"episode": 1}
        store_b.close()


def test_tags_are_stored_as_a_tuple_matching_memory_record():
    store = SQLiteMemoryStore()
    store.store("x", tags=("a", "b", "c"))
    record = store.retrieve("a")[0]
    assert record.tags == ("a", "b", "c")
    assert isinstance(record.tags, tuple)
