import math

import pytest

from agent_rag.embeddings import Embedder, LocalHashingEmbedder


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))


def test_dimension_is_respected():
    assert len(LocalHashingEmbedder(dim=256).embed_query("anything")) == 256


def test_vectors_are_unit_length():
    vector = LocalHashingEmbedder().embed_query("retry budget exhausted")
    assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-9)


def test_same_text_always_gives_the_same_vector():
    assert LocalHashingEmbedder().embed_query("permanent failure") == \
           LocalHashingEmbedder().embed_query("permanent failure")


def test_documents_and_queries_share_one_space():
    embedder = LocalHashingEmbedder()
    [doc] = embedder.embed_documents(["retry budget exhausted"])
    query = embedder.embed_query("retry budget exhausted")
    assert math.isclose(cosine(doc, query), 1.0, rel_tol=1e-6)


def test_shared_vocabulary_scores_above_unrelated_text():
    embedder = LocalHashingEmbedder()
    query = embedder.embed_query("retry budget exhausted for the collector")
    related = embedder.embed_query("the collector exhausted its retry budget")
    unrelated = embedder.embed_query("hard bounce suppression for mail recipients")
    assert cosine(query, related) > cosine(query, unrelated)


def test_empty_text_returns_a_zero_vector_without_crashing():
    vector = LocalHashingEmbedder().embed_query("   ")
    assert all(v == 0.0 for v in vector)


def test_invalid_dimension_is_rejected():
    with pytest.raises(ValueError):
        LocalHashingEmbedder(dim=0)


def test_the_local_embedder_satisfies_the_protocol():
    assert isinstance(LocalHashingEmbedder(), Embedder)