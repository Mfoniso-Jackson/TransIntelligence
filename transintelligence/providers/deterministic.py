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
to resolve to. `TemplateTextProvider` does not generate real language --
it exists so code that depends on a `TextGenerationProvider` can be
written and tested without a live LLM call.
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

class TemplateTextProvider:
    def __init__(self, template: str = "[response to: {prompt}]") -> None:
        self.template = template

    def generate(self, prompt: str) -> str:
        return self.template.format(prompt=prompt)
