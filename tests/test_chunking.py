from agent_rag.chunking import Chunk, chunk_markdown

DOC = """\
# Collector

Intro text.

## Retries

Transient failures are retried.

### Backoff

Exponential with jitter.

## Suppression

Duplicates are dropped.
"""

def test_embedding_text_omits_a_blank_heading_path():
    chunk = Chunk(
    doc_id="d.md",
    chunk_index=0,
    heading_path="Collector > Retries",
    content="Retries use exponential backoff.")

    assert chunk.embedding_text == "Collector > Retries\n\nRetries use exponential backoff."
    chunk = Chunk(doc_id="d.md", chunk_index=0, heading_path="", content="body")
    assert chunk.embedding_text == "body"

def test_metadata_is_not_shared_between_chunks():

    a=Chunk(doc_id="d.md", chunk_index=0, heading_path="", content="a")
    b = Chunk(doc_id="d.md", chunk_index=1, heading_path="", content="b")
    a.metadata["service"] = "collector"
    assert b.metadata == {}

def test_each_section_becomes_a_chunk():
    assert len(chunk_markdown("d.md", DOC)) == 4


def test_heading_paths_nest():
    paths = [c.heading_path for c in chunk_markdown("d.md", DOC)]
    assert paths[1] == "Collector > Retries"
    assert paths[2] == "Collector > Retries > Backoff"


def test_heading_path_pops_back_to_sibling_level():
    chunk = next(c for c in chunk_markdown("d.md", DOC) if c.content.startswith("Duplicates"))
    assert chunk.heading_path == "Collector > Suppression"


def test_chunk_indexes_are_contiguous():
    chunks = chunk_markdown("d.md", DOC)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_empty_document_yields_no_chunks():
    assert chunk_markdown("d.md", "") == []


def test_metadata_reaches_every_chunk():
    chunks = chunk_markdown("d.md", DOC, metadata={"service": "collector"})
    assert all(c.metadata["service"] == "collector" for c in chunks)