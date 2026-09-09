"""Deterministic, offline reference implementations of
`EmbeddingProvider`/`TextGenerationProvider` (`interfaces.py`) --
stdlib-only, no network call, no API key, no new dependency. The same
"start with the dependency-free option, defer the heavier one" choice
`transintelligence/storage/sqlite_store.py` made for persistence:
this repo has never made a real network call anywhere, and every
experiment here depends on deterministic, offline, seeded
reproducibility -- a real OpenAI/Anthropic-calling implementation would
need API keys, cost money, introduce non-determinism, and require a
mocking strategy this repo has no convention for. Not built here,
left as documented future work behind the same `EmbeddingProvider`/
`TextGenerationProvider` interfaces.

These are reference implementations, not stand-ins for real semantic
capability: `DeterministicHashEmbeddingProvider` does not produce
embeddings where similar text lands near similar vectors (unlike a real
embedding model) -- it exists to make `EmbeddingProvider` concretely
testable and to give `Entity.embedding_ref` a real, reproducible value
to resolve to. `BagOfWordsHashEmbeddingProvider` is a small, honest step
up (see its own docstring): related text genuinely ranks closer via
this one, unlike the whole-string hash. `TemplateTextProvider` does not
generate real language -- it exists so code that depends on a
`TextGenerationProvider` can be written and tested without a live LLM
call.
"""
from __future__ import annotations
import hashlib

class DeterministicHashEmbeddingProvider:
    def __init__(self, dimensions: int = 8) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be at least 1")
        if dimensions > hashlib.sha256().digest_size:
            raise ValueError(f"dimensions must be at most {hashlib.sha256().digest_size} (sha256 digest size)")
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [(byte / 127.5) - 1.0 for byte in digest[:self.dimensions]]

class BagOfWordsHashEmbeddingProvider:
    """Feature-hashed bag-of-words embedding (the "hashing trick":
    Weinberger, Dasgupta, Langford, Smola, Attenberg, *Feature Hashing
    for Large Scale Multitask Learning*, ICML 2009, 1113-1120).
    Unlike `DeterministicHashEmbeddingProvider` (which hashes the WHOLE
    string, destroying any relationship between related-but-different
    text), this hashes each WORD into one of `dimensions` buckets and
    counts occurrences -- text sharing more words produces vectors with
    more overlapping nonzero buckets, giving genuinely higher cosine
    similarity for genuinely related text, not just exact matches.
    Still not a real embedding model -- word order, synonyms, and
    semantics beyond shared vocabulary are all ignored -- a small,
    honest step toward real similarity, not a substitute for one."""

    def __init__(self, dimensions: int = 64) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be at least 1")
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for word in text.lower().split():
            bucket = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16) % self.dimensions
            vector[bucket] += 1.0
        return vector

class TemplateTextProvider:
    def __init__(self, template: str = "[response to: {prompt}]") -> None:
        self.template = template

    def generate(self, prompt: str) -> str:
        return self.template.format(prompt=prompt)
