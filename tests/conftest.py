import uuid

import pytest

from agent_rag.chunking import chunk_markdown
from agent_rag.embeddings import LocalHashingEmbedder
from agent_rag.store import PgVectorStore

DOC = """\
# Collector

The collector fetches postings every fifteen minutes.

## Retries

Transient failures are retried with exponential backoff.

## Suppression

Duplicate digests are dropped before sending.
"""


@pytest.fixture
def doc():
    return DOC


@pytest.fixture
def store():
    store = PgVectorStore(table=f"test_chunks_{uuid.uuid4().hex[:8]}")
    store.create_schema()
    yield store
    store.drop_schema()


@pytest.fixture
def indexed(store):
    embedder = LocalHashingEmbedder()
    chunks = chunk_markdown("collector.md", DOC)
    store.upsert(chunks, embedder.embed_documents([c.embedding_text for c in chunks]))
    return store, embedder
