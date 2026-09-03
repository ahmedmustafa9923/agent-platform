import subprocess
import sys

from agent_rag.embeddings import VertexEmbedder
from agent_rag.retriever import RunbookRetriever
from agent_rag.store import PgVectorStore

question = " ".join(sys.argv[1:])
project = subprocess.check_output(
    ["gcloud", "config", "get-value", "project"], text=True
).strip()

retriever = RunbookRetriever(
    PgVectorStore(table="catalog_chunks"),
    VertexEmbedder(project=project),
)

for i, r in enumerate(retriever.retrieve(question, top_k=3), start=1):
    print(f"\n[{i}] {r.citation}   (rrf {r.score:.5f})")
    print(f"    {r.content[:280]}")