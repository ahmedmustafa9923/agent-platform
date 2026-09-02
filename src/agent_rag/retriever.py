from .embeddings import Embedder, LocalHashingEmbedder
from .store import PgVectorStore, SearchResult


class RunbookRetriever:
    def __init__(self, store: PgVectorStore | None = None, embedder: Embedder | None = None):
        self.store = store or PgVectorStore()
        self.embedder = embedder or LocalHashingEmbedder()

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query = (query or "").strip()
        if not query:
            return []
        return self.store.search(query, self.embedder.embed_query(query), top_k=top_k)

    def as_context(self, results: list[SearchResult]) -> str:
        return "\n\n".join(f"[{i}] {r.citation}\n{r.content}" for i, r in enumerate(results, start=1))