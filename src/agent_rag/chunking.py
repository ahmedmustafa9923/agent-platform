from dataclasses import dataclass, field

import re

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")

@dataclass
class Chunk:
    doc_id: str
    chunk_index: int
    heading_path: str
    content: str
    metadata: dict = field(default_factory=dict)

    @property
    def embedding_text(self) -> str:
        if self.heading_path:
            return f"{self.heading_path}\n\n{self.content}"
        return self.content

def _split_sections(markdown: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    stack: list[str] = []
    heading_path = ""
    buffer: list[str] = []

    for line in markdown.splitlines():
        match = _HEADING_RE.match(line)
        if match is None:
            buffer.append(line)
            continue

        text = "\n".join(buffer).strip()
        if text:
            sections.append((heading_path, text))
        buffer = []

        level = len(match.group(1))
        del stack[level - 1:]
        stack.append(match.group(2))
        heading_path = " > ".join(stack)

    text = "\n".join(buffer).strip()
    if text:
        sections.append((heading_path, text))
    return sections

def chunk_markdown(doc_id: str, markdown: str, metadata: dict | None = None) -> list[Chunk]:
    base = dict(metadata or {})
    return [
        Chunk(
            doc_id=doc_id,
            chunk_index=i,
            heading_path=heading_path,
            content=text,
            metadata=dict(base),
        )
        for i, (heading_path, text) in enumerate(_split_sections(markdown))
    ]
