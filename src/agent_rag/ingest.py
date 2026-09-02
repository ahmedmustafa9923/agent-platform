import re
from dataclasses import dataclass
from pathlib import Path

from .chunking import chunk_markdown

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_frontmatter(text: str) -> tuple[dict, str]:
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        return {}, text

    metadata = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        metadata[key.strip()] = value.strip().strip("'\"")
    return metadata, text[match.end():]


def load_documents(directory) -> list[tuple[str, str, dict]]:
    documents = []
    for path in sorted(Path(directory).glob("*.md")):
        metadata, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        metadata.setdefault("source_path", path.name)
        documents.append((path.name, body, metadata))
    return documents


@dataclass
class IngestReport:
    documents: int = 0
    chunks: int = 0

    def summary(self) -> str:
        return f"{self.documents} documents, {self.chunks} chunks"


def ingest_directory(directory, store, embedder) -> IngestReport:
    store.create_schema()
    documents = load_documents(directory)

    chunks = []
    for doc_id, body, metadata in documents:
        chunks.extend(chunk_markdown(doc_id, body, metadata=metadata))

    if chunks:
        store.upsert(chunks, embedder.embed_documents([c.embedding_text for c in chunks]))

    return IngestReport(documents=len(documents), chunks=len(chunks))