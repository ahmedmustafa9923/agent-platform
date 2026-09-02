import pytest

from agent_rag.embeddings import LocalHashingEmbedder
from agent_rag.ingest import ingest_directory, load_documents, parse_frontmatter

DOC_A = """\
---
service: collector
owner: platform
---

# Collector

Fetches postings.

## Retries

Transient failures are retried.
"""

DOC_B = """\
---
service: notifier
---

# Notifier

Sends the digest.

## Suppression

Duplicate digests are dropped.
"""


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "collector.md").write_text(DOC_A)
    (tmp_path / "notifier.md").write_text(DOC_B)
    return tmp_path


def test_frontmatter_is_parsed_into_metadata():
    metadata, _ = parse_frontmatter(DOC_A)
    assert metadata == {"service": "collector", "owner": "platform"}


def test_frontmatter_is_removed_from_the_body():
    _, body = parse_frontmatter(DOC_A)
    assert body.startswith("# Collector")
    assert "service:" not in body


def test_document_without_frontmatter_is_unchanged():
    assert parse_frontmatter("# Title\n\ntext") == ({}, "# Title\n\ntext")


def test_load_documents_reads_every_markdown_file(corpus):
    assert [d[0] for d in load_documents(corpus)] == ["collector.md", "notifier.md"]


def test_load_documents_records_the_source_path(corpus):
    assert load_documents(corpus)[0][2]["source_path"] == "collector.md"


def test_ingest_reports_what_it_did(corpus, store):
    report = ingest_directory(corpus, store, LocalHashingEmbedder())
    assert report.documents == 2
    assert report.chunks == 4


def test_ingest_writes_every_chunk(corpus, store):
    ingest_directory(corpus, store, LocalHashingEmbedder())
    assert store.count() == 4


def test_reingesting_does_not_duplicate(corpus, store):
    embedder = LocalHashingEmbedder()
    ingest_directory(corpus, store, embedder)
    ingest_directory(corpus, store, embedder)
    assert store.count() == 4


def test_frontmatter_metadata_reaches_the_stored_rows(corpus, store):
    ingest_directory(corpus, store, LocalHashingEmbedder())
    hit = store.search_text("duplicate digests dropped")[0]
    assert hit.doc_id == "notifier.md"
    assert hit.metadata["service"] == "notifier"