import pytest

from agent_rag.chunking import chunk_markdown


def test_schema_creation_is_idempotent(store):
    store.create_schema()


def test_new_table_starts_empty(store):
    assert store.count() == 0


def test_upsert_writes_every_chunk(indexed):
    store, _ = indexed
    assert store.count() == 3


def test_upsert_twice_updates_rather_than_duplicates(indexed, doc):
    store, embedder = indexed
    chunks = chunk_markdown("collector.md", doc)
    store.upsert(chunks, embedder.embed_documents([c.embedding_text for c in chunks]))
    assert store.count() == 3


def test_upsert_rejects_mismatched_lengths(store, doc):
    with pytest.raises(ValueError):
        store.upsert(chunk_markdown("d.md", doc), [])


def test_search_finds_the_matching_section(indexed):
    store, embedder = indexed
    results = store.search_vector(embedder.embed_query("duplicate digests dropped"))
    assert results[0].heading_path == "Collector > Suppression"


def test_search_respects_top_k(indexed):
    store, embedder = indexed
    assert len(store.search_vector(embedder.embed_query("retries"), top_k=2)) == 2


def test_scores_descend(indexed):
    store, embedder = indexed
    scores = [r.score for r in store.search_vector(embedder.embed_query("retries"))]
    assert scores == sorted(scores, reverse=True)


def test_citation_names_document_and_section(indexed):
    store, embedder = indexed
    assert store.search_vector(embedder.embed_query("retries"))[0].citation.startswith("collector.md > ")


def test_lexical_search_finds_the_matching_section(indexed):
    store, _ = indexed
    assert store.search_text("duplicate digests dropped")[0].heading_path == "Collector > Suppression"


def test_lexical_search_returns_nothing_when_no_term_matches(indexed):
    store, _ = indexed
    assert store.search_text("zzzqqq wibble") == []


def test_hybrid_ranks_the_agreed_passage_first(indexed):
    store, embedder = indexed
    query = "duplicate digests dropped"
    assert store.search(query, embedder.embed_query(query))[0].heading_path == "Collector > Suppression"


def test_hybrid_score_is_the_sum_of_both_reciprocal_ranks(indexed):
    store, embedder = indexed
    query = "duplicate digests dropped"
    top = store.search(query, embedder.embed_query(query))[0]
    assert top.score == pytest.approx(1 / 61 + 1 / 61)


def test_hybrid_still_works_when_one_arm_finds_nothing(indexed):
    store, embedder = indexed
    results = store.search("zzzqqq wibble", embedder.embed_query("retries backoff"))
    assert results
    assert all(r.score == pytest.approx(1 / (60 + i)) for i, r in enumerate(results, start=1))


def test_hybrid_scores_descend(indexed):
    store, embedder = indexed
    query = "duplicate digests dropped"
    scores = [r.score for r in store.search(query, embedder.embed_query(query))]
    assert scores == sorted(scores, reverse=True)
