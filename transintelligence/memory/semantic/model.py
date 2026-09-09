"""Semantic memory: similarity-based retrieval, as distinct from
`episodic`/`strategic` memory's tag-based `retrieve()` -- until now, all
three memory namespaces re-exported the same generic `InMemoryStore`
with no actual difference in behavior. `SemanticMemoryStore` is the
first real content in `memory/semantic/`, and the first genuine consumer
of `EmbeddingProvider` (`transintelligence/providers/`), which existed
only as an interface tested against itself until now.

Wraps any `PersistentStore`-conforming backend (`InMemoryStore` or
`SQLiteMemoryStore`, `transintelligence/storage/`) rather than
reimplementing storage -- `store()`/`retrieve()` delegate straight
through, so nothing about tag-based retrieval changes. `retrieve_similar()`
is the new capability: embeds the query and every candidate record's
`str(content)` fresh on each call (via whichever `EmbeddingProvider` is
given) and ranks by cosine similarity, rather than caching embeddings
alongside a possibly-persistent backend -- avoids a staleness/cache-
invalidation problem a cache would introduce, at the cost of
recomputing embeddings per query. The same "simplicity over efficiency
at this repo's scale" tradeoff `LinearDynamicsModel.fit`'s from-scratch
refitting already makes.

**Ranking quality is entirely a function of the `EmbeddingProvider`
given, not something this class adds.** With the default
`DeterministicHashEmbeddingProvider` (`transintelligence/providers/`),
`retrieve_similar()` reliably ranks an EXACT text match first (hashing
the same string twice always gives cosine similarity 1.0) but does
*not* meaningfully rank genuinely-similar-but-different text nearby --
that provider's own docstring already says textually similar strings
don't hash to nearby vectors. Real semantic search needs a real
embedding model behind the same interface, deliberately not built here
(see `transintelligence/providers/deterministic.py`'s own docstring for
why). Don't read a demo using the default provider as a claim that
semantic search "works" in the ordinary sense -- it demonstrates the
interface and the ranking mechanics, not retrieval quality.
"""
from __future__ import annotations
import math
from typing import Any
from transintelligence.memory.base import InMemoryStore, MemoryRecord
from transintelligence.providers import DeterministicHashEmbeddingProvider
from transintelligence.providers.interfaces import EmbeddingProvider
from transintelligence.storage.interfaces import PersistentStore


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticMemoryStore:
    def __init__(self, backend: PersistentStore | None = None, embedding_provider: EmbeddingProvider | None = None) -> None:
        self.backend = backend if backend is not None else InMemoryStore()
        self.embedding_provider = embedding_provider if embedding_provider is not None else DeterministicHashEmbeddingProvider()

    def store(self, content: Any, tags: tuple[str, ...] = ()) -> MemoryRecord:
        return self.backend.store(content, tags)

    def retrieve(self, tag: str | None = None) -> list[MemoryRecord]:
        return self.backend.retrieve(tag)

    def retrieve_similar(self, query: str, k: int = 5, tag: str | None = None) -> list[MemoryRecord]:
        """The `k` records (optionally restricted to `tag`) whose
        `str(content)` is most similar to `query`, by cosine similarity
        over `embedding_provider`'s vectors -- highest similarity first.
        Empty candidate pool returns an empty list, not an error."""
        if k < 1:
            raise ValueError("k must be at least 1")
        query_vector = self.embedding_provider.embed(query)
        candidates = self.backend.retrieve(tag)
        scored = [
            (_cosine_similarity(query_vector, self.embedding_provider.embed(str(record.content))), record)
            for record in candidates
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [record for _, record in scored[:k]]
