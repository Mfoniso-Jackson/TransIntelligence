"""Tests for the deterministic reference implementations of
EmbeddingProvider/TextGenerationProvider
(transintelligence/providers/deterministic.py) -- the first content in
transintelligence/providers/, previously nonexistent.
"""
import pytest

from transintelligence.providers import DeterministicHashEmbeddingProvider, TemplateTextProvider


def test_embed_is_deterministic_for_the_same_text():
    provider = DeterministicHashEmbeddingProvider()
    assert provider.embed("hello world") == provider.embed("hello world")


def test_embed_differs_for_different_text():
    provider = DeterministicHashEmbeddingProvider()
    assert provider.embed("hello") != provider.embed("world")


def test_embed_respects_the_requested_dimensions():
    provider = DeterministicHashEmbeddingProvider(dimensions=4)
    assert len(provider.embed("anything")) == 4
    provider16 = DeterministicHashEmbeddingProvider(dimensions=16)
    assert len(provider16.embed("anything")) == 16


def test_embed_values_are_within_the_normalized_range():
    provider = DeterministicHashEmbeddingProvider(dimensions=32)
    vector = provider.embed("a reasonably long piece of text to embed")
    assert all(-1.0 <= v <= 1.0 for v in vector)


def test_raises_on_non_positive_dimensions():
    with pytest.raises(ValueError):
        DeterministicHashEmbeddingProvider(dimensions=0)


def test_raises_on_dimensions_larger_than_the_hash_digest():
    with pytest.raises(ValueError):
        DeterministicHashEmbeddingProvider(dimensions=64)  # sha256 digest is 32 bytes


def test_generate_uses_the_default_template():
    provider = TemplateTextProvider()
    assert provider.generate("what is 2+2?") == "[response to: what is 2+2?]"


def test_generate_uses_a_custom_template():
    provider = TemplateTextProvider(template="Q: {prompt}\nA:")
    assert provider.generate("why?") == "Q: why?\nA:"


def test_generate_is_deterministic():
    provider = TemplateTextProvider()
    assert provider.generate("same input") == provider.generate("same input")
