"""Tests for the deterministic reference implementations of
EmbeddingProvider/TextGenerationProvider
(transintelligence/providers/deterministic.py) -- the first content in
transintelligence/providers/, previously nonexistent.
"""
import pytest

from transintelligence.providers import (
    BagOfWordsHashEmbeddingProvider, DeterministicHashEmbeddingProvider, TemplateTextProvider,
)


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


def test_bag_of_words_embed_matches_hand_computed_word_counts():
    """dimensions=8, sha256("a")%8=3 and sha256("b")%8=5 (verified
    directly) -- "a a b" must produce a vector with exactly bucket
    3 = 2.0 (two occurrences of "a"), bucket 5 = 1.0 (one "b"), and
    every other bucket 0.0, not an approximation."""
    provider = BagOfWordsHashEmbeddingProvider(dimensions=8)
    vector = provider.embed("a a b")
    expected = [0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.0]
    assert vector == expected


def test_bag_of_words_embed_is_case_insensitive():
    provider = BagOfWordsHashEmbeddingProvider(dimensions=16)
    assert provider.embed("Cat Food") == provider.embed("cat food")


def test_bag_of_words_shared_words_produce_higher_similarity_than_unrelated_text():
    """The actual point of this provider over the whole-string hash:
    text sharing words should rank as more similar than text sharing
    none, via plain cosine similarity on the embeddings themselves."""
    import math

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    provider = BagOfWordsHashEmbeddingProvider(dimensions=64)
    cat_food = provider.embed("cat food")
    dog_food = provider.embed("dog food")  # shares "food"
    stock_market = provider.embed("stock market report")  # shares nothing

    assert cosine(cat_food, dog_food) > cosine(cat_food, stock_market)


def test_bag_of_words_raises_on_non_positive_dimensions():
    with pytest.raises(ValueError):
        BagOfWordsHashEmbeddingProvider(dimensions=0)


def test_generate_uses_the_default_template():
    provider = TemplateTextProvider()
    assert provider.generate("what is 2+2?") == "[response to: what is 2+2?]"


def test_generate_uses_a_custom_template():
    provider = TemplateTextProvider(template="Q: {prompt}\nA:")
    assert provider.generate("why?") == "Q: why?\nA:"


def test_generate_is_deterministic():
    provider = TemplateTextProvider()
    assert provider.generate("same input") == provider.generate("same input")
