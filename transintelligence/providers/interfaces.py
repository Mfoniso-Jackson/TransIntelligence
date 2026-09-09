"""The replaceable-interface stubs for external model providers
(docs/master-context.md §30: "Keep model providers behind interfaces.
Do not hardcode OpenAI, Anthropic, Gemini, a particular embedding
provider, or a particular vector database into the conceptual core."),
mirroring `transintelligence/reasoning/interfaces.py`'s `Protocol`
pattern and `transintelligence/storage/interfaces.py`'s `PersistentStore`.

Unlike `Event` or `PersistentStore`, nothing in this repo currently
consumes either protocol below -- `Entity.embedding_ref`
(`transintelligence/core/entities/model.py`) is the one placeholder
field that gestures at this need, never read or written anywhere before
now. These interfaces exist so that future code depending on an
embedding or text-generation capability can be written against a stable
shape, without hardcoding a particular vendor -- the principle §30
states, not a claim that a real provider is wired up yet (see
`deterministic.py` for why not).
"""
from __future__ import annotations
from typing import Protocol

class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> list[float]: ...

class TextGenerationProvider(Protocol):
    def generate(self, prompt: str) -> str: ...
