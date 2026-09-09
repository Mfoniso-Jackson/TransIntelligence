"""Tests for SemanticMemoryStore (transintelligence/memory/semantic/model.py)
-- the first real content in memory/semantic/ (previously a re-export of
the same generic InMemoryStore episodic/strategic also use) and the
first genuine consumer of EmbeddingProvider (transintelligence/providers/),
which existed only as an interface tested against itself until now.
"""
import pytest

from transintelligence.memory.semantic import SemanticMemoryStore
from transintelligence.providers import BagOfWordsHashEmbeddingProvider, DeterministicHashEmbeddingProvider
from transintelligence.storage import SQLiteMemoryStore


def test_store_and_retrieve_delegate_straight_through_to_the_backend():
    """Nothing about tag-based retrieval should change -- SemanticMemoryStore
    wraps, it doesn't reimplement."""
    store = SemanticMemoryStore()
    store.store("first", tags=("a",))
    store.store("second", tags=("b",))
    assert len(store.retrieve()) == 2
    assert store.retrieve("a")[0].content == "first"


def test_retrieve_similar_ranks_an_exact_match_first_with_the_default_provider():
    """The one thing guaranteed regardless of embedding quality: a query
    that exactly matches a stored record's content has cosine similarity
    1.0 against itself (hashing the same string twice gives identical
    vectors), so it must rank first even among several distractors."""
    store = SemanticMemoryStore()
    store.store("the quick brown fox")
    store.store("something completely unrelated")
    store.store("another unrelated sentence")
    results = store.retrieve_similar("the quick brown fox", k=1)
    assert len(results) == 1
    assert results[0].content == "the quick brown fox"


def test_retrieve_similar_ranks_word_sharing_text_higher_with_bag_of_words_provider():
    """With a provider that actually preserves lexical overlap
    (BagOfWordsHashEmbeddingProvider, unlike the whole-string-hash
    default), text sharing words with the query should rank above text
    sharing none -- the genuine "semantic-ish" retrieval case."""
    store = SemanticMemoryStore(embedding_provider=BagOfWordsHashEmbeddingProvider(dimensions=64))
    store.store("cat food recipes")
    store.store("dog food recipes")
    store.store("stock market report")
    results = store.retrieve_similar("cat food", k=3)
    contents = [r.content for r in results]
    assert contents[0] == "cat food recipes"
    assert contents.index("dog food recipes") < contents.index("stock market report")


def test_retrieve_similar_respects_k():
    store = SemanticMemoryStore(embedding_provider=BagOfWordsHashEmbeddingProvider(dimensions=32))
    for i in range(5):
        store.store(f"document number {i}")
    assert len(store.retrieve_similar("document", k=2)) == 2


def test_retrieve_similar_respects_the_tag_filter():
    store = SemanticMemoryStore(embedding_provider=BagOfWordsHashEmbeddingProvider(dimensions=32))
    store.store("cat food", tags=("kept",))
    store.store("cat food but excluded", tags=("excluded",))
    results = store.retrieve_similar("cat food", k=5, tag="kept")
    assert len(results) == 1
    assert results[0].content == "cat food"


def test_retrieve_similar_with_an_empty_store_returns_an_empty_list():
    store = SemanticMemoryStore()
    assert store.retrieve_similar("anything", k=5) == []


def test_raises_on_non_positive_k():
    store = SemanticMemoryStore()
    store.store("something")
    with pytest.raises(ValueError):
        store.retrieve_similar("something", k=0)


def test_works_with_a_persistent_sqlite_backend():
    """The whole point of wrapping PersistentStore rather than
    InMemoryStore specifically: a SQLiteMemoryStore backend should work
    identically, with no special-casing needed."""
    store = SemanticMemoryStore(backend=SQLiteMemoryStore(), embedding_provider=DeterministicHashEmbeddingProvider())
    store.store("persisted content")
    store.store("other content")
    results = store.retrieve_similar("persisted content", k=1)
    assert results[0].content == "persisted content"
