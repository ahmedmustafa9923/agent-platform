import uuid

import pytest

from agent_rag.store import PgVectorStore


@pytest.fixture
def store():
    store = PgVectorStore(table=f"test_chunks_{uuid.uuid4().hex[:8]}")
    store.create_schema()
    yield store
    store.drop_schema()