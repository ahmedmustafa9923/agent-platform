import os
import subprocess

from agent_rag.chunking import chunk_markdown
from agent_rag.embeddings import VertexEmbedder
from agent_rag.store import PgVectorStore

DOC = """\
# Collector

The collector fetches postings every fifteen minutes.

## Retries

Transient failures are retried with exponential backoff, up to five attempts.

## Suppression

Duplicate digests are dropped before sending.
"""

QUERIES = [
    "why didn't it send the same thing twice",
    "what happens when a fetch briefly fails",
]

project = subprocess.check_output(
    ["gcloud", "config", "get-value", "project"], text=True
).strip()

print("host:", os.environ["RAG_DATABASE_URL"].split("@")[-1])

store = PgVectorStore(table="runbook_chunks")
embedder = VertexEmbedder(project=project)

store.drop_schema()
store.create_schema()

chunks = chunk_markdown("collector.md", DOC)
store.upsert(chunks, embedder.embed_documents([c.embedding_text for c in chunks]))
print("rows in AlloyDB:", store.count())

for query in QUERIES:
    top = store.search(query, embedder.embed_query(query), top_k=1)[0]
    print(f"  {query!r}")
    print(f"     -> {top.heading_path}   (rrf {top.score:.5f})")
