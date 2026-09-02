import subprocess

from agent_rag.chunking import chunk_markdown
from agent_rag.embeddings import LocalHashingEmbedder, VertexEmbedder
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
    "duplicate digests dropped",
    "why didn't it send the same thing twice",
    "what happens when a fetch briefly fails",
]

project = subprocess.check_output(
    ["gcloud", "config", "get-value", "project"], text=True
).strip()

embedders = {
    "local": LocalHashingEmbedder(),
    "vertex": VertexEmbedder(project=project),
}

for name, embedder in embedders.items():
    store = PgVectorStore(table=f"cmp_{name}")
    store.drop_schema()
    store.create_schema()

    chunks = chunk_markdown("collector.md", DOC)
    store.upsert(chunks, embedder.embed_documents([c.embedding_text for c in chunks]))

    print(f"=== {name.upper()} ===")
    for query in QUERIES:
        top = store.search_vector(embedder.embed_query(query), top_k=1)[0]
        print(f"  {query!r}")
        print(f"     -> {top.heading_path}   (cos {top.score:+.4f})")
    store.drop_schema()
    print()
