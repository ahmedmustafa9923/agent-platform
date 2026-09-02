from agent_rag.retriever import RunbookRetriever


def test_retrieve_finds_the_matching_section(indexed):
    store, embedder = indexed
    results = RunbookRetriever(store, embedder).retrieve("duplicate digests dropped")
    assert results[0].heading_path == "Collector > Suppression"


def test_retrieve_respects_top_k(indexed):
    store, embedder = indexed
    assert len(RunbookRetriever(store, embedder).retrieve("retries", top_k=2)) == 2


def test_empty_query_returns_nothing_without_touching_the_database(indexed):
    store, embedder = indexed
    assert RunbookRetriever(store, embedder).retrieve("   ") == []


def test_context_numbers_and_labels_each_passage(indexed):
    store, embedder = indexed
    retriever = RunbookRetriever(store, embedder)
    results = retriever.retrieve("duplicate digests dropped", top_k=2)
    context = retriever.as_context(results)
    assert context.startswith("[1] collector.md > ")
    assert "[2] " in context


def test_context_of_no_results_is_empty(indexed):
    store, embedder = indexed
    assert RunbookRetriever(store, embedder).as_context([]) == ""