import argparse
import os

from .embeddings import LocalHashingEmbedder, VertexEmbedder
from .retriever import RunbookRetriever
from .store import PgVectorStore

TABLE = os.environ.get("RAG_TABLE", "catalog_chunks")


def build_embedder():
    if os.environ.get("RAG_EMBEDDING_BACKEND", "vertex").lower() == "local":
        return LocalHashingEmbedder()
    return VertexEmbedder(
        project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
    )


def build_server(retriever=None):
    from mcp.server.mcpserver import MCPServer

    retriever = retriever or RunbookRetriever(PgVectorStore(table=TABLE), build_embedder())
    server = MCPServer(name="catalog-retrieval")

    @server.tool(
        title="Search the catalog",
        description=(
            "Search the catalog and return the most relevant passages, each with the "
            "document and section it came from. Use natural-language questions."
        ),
    )
    def search_catalog(query: str, top_k: int = 3) -> dict:
        results = retriever.retrieve(query, top_k=top_k)
        return {
            "query": query,
            "match_count": len(results),
            "citations": [r.citation for r in results],
            "context": retriever.as_context(results),
        }

    return server


def main():
    parser = argparse.ArgumentParser(description="Catalog retrieval MCP server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    args = parser.parse_args()

    server = build_server()
    if args.transport == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
