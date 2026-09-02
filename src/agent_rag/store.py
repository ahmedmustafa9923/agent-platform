import os
from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg.rows import dict_row

import json
from dataclasses import dataclass
from typing import Sequence

from pgvector import Vector
from .chunking import Chunk

DEFAULT_DSN = "postgresql://postgres:postgres@localhost:5433/runbooks"


def database_url() -> str:
    return os.environ.get("RAG_DATABASE_URL", DEFAULT_DSN)
@dataclass
class SearchResult:
    doc_id: str
    chunk_index: int
    heading_path: str
    content: str
    metadata: dict
    score: float

    @property
    def citation(self) -> str:
        if self.heading_path:
            return f"{self.doc_id} > {self.heading_path}"
        return self.doc_id

class PgVectorStore:
    def __init__(self, dsn: str | None = None, table: str = "runbook_chunks", dim: int = 768):
        self._dsn = dsn or database_url()
        self._table = table
        self._dim = dim

    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")
        if not chunks:
            return 0

        rows = [
            (
                c.doc_id,
                c.chunk_index,
                c.heading_path,
                c.content,
                json.dumps(c.metadata),
                Vector(list(e)),
            )
            for c, e in zip(chunks, embeddings)
        ]

        statement = sql.SQL(
            """
            INSERT INTO {t} (doc_id, chunk_index, heading_path, content, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s::jsonb, %s)
            ON CONFLICT (doc_id, chunk_index) DO UPDATE SET
                heading_path = EXCLUDED.heading_path,
                content      = EXCLUDED.content,
                metadata     = EXCLUDED.metadata,
                embedding    = EXCLUDED.embedding
            """
        ).format(t=sql.Identifier(self._table))

        with self.connect() as conn, conn.cursor() as cur:
            cur.executemany(statement, rows)
            conn.commit()
        return len(rows)

    def search_text(self, query_text: str, top_k: int = 5) -> list[SearchResult]:
        statement = sql.SQL(
            """
            SELECT doc_id, chunk_index, heading_path, content, metadata,
                   ts_rank_cd(tsv, q) AS score
            FROM {t},
                 to_tsquery('english',
                     (SELECT string_agg(quote_literal(lexeme), ' | ')
                        FROM unnest(tsvector_to_array(to_tsvector('english', %(text)s))) AS lexeme)
                 ) AS q
            WHERE tsv @@ q
            ORDER BY ts_rank_cd(tsv, q) DESC, doc_id, chunk_index
            LIMIT %(k)s
            """
        ).format(t=sql.Identifier(self._table))

        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(statement, {"text": query_text, "k": top_k})
            return [
                SearchResult(
                    doc_id=r["doc_id"],
                    chunk_index=r["chunk_index"],
                    heading_path=r["heading_path"],
                    content=r["content"],
                    metadata=r["metadata"],
                    score=float(r["score"]),
                )
                for r in cur.fetchall()
            ]

    def search(
        self,
        query_text: str,
        query_embedding: Sequence[float],
        top_k: int = 5,
        candidate_k: int = 20,
        rrf_k: int = 60,
    ) -> list[SearchResult]:
        dense = self.search_vector(query_embedding, top_k=candidate_k)
        lexical = self.search_text(query_text, top_k=candidate_k)

        scores: dict[tuple[str, int], float] = {}
        best: dict[tuple[str, int], SearchResult] = {}

        for arm in (dense, lexical):
            for rank, result in enumerate(arm, start=1):
                key = (result.doc_id, result.chunk_index)
                scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
                best.setdefault(key, result)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

        results = []
        for key, score in ranked[:top_k]:
            result = best[key]
            result.score = score
            results.append(result)
        return results

    def search_vector(self, query_embedding: Sequence[float], top_k: int = 5) -> list[SearchResult]:
        statement = sql.SQL(
            """
            SELECT doc_id, chunk_index, heading_path, content, metadata,
                   1 - (embedding <=> %(q)s) AS score
            FROM {t}
            ORDER BY embedding <=> %(q)s
            LIMIT %(k)s
            """
        ).format(t=sql.Identifier(self._table))

        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(statement, {"q": Vector(list(query_embedding)), "k": top_k})
            return [
                SearchResult(
                    doc_id=r["doc_id"],
                    chunk_index=r["chunk_index"],
                    heading_path=r["heading_path"],
                    content=r["content"],
                    metadata=r["metadata"],
                    score=float(r["score"]),
                )
                for r in cur.fetchall()
            ]

    @contextmanager
    def connect(self):
        with psycopg.connect(self._dsn, row_factory=dict_row) as conn:
            try:
                register_vector(conn)
            except psycopg.ProgrammingError:
                conn.rollback()
                with conn.cursor() as cur:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                conn.commit()
                register_vector(conn)
            yield conn

    def create_schema(self) -> None:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {table} (
                        id            bigserial PRIMARY KEY,
                        doc_id        text        NOT NULL,
                        chunk_index   integer     NOT NULL,
                        heading_path  text        NOT NULL DEFAULT '',
                        content       text        NOT NULL,
                        metadata      jsonb       NOT NULL DEFAULT '{{}}'::jsonb,
                        embedding     vector({dim}) NOT NULL,
                        tsv tsvector GENERATED ALWAYS AS (
                            setweight(to_tsvector('english', heading_path), 'A')
                            || setweight(to_tsvector('english', content), 'B')
                        ) STORED,
                        UNIQUE (doc_id, chunk_index)
                    )
                    """
                ).format(table=sql.Identifier(self._table), dim=sql.Literal(self._dim))
            )
            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {name} ON {table} "
                    "USING hnsw (embedding vector_cosine_ops)"
                ).format(
                    name=sql.Identifier(f"{self._table}_hnsw"),
                    table=sql.Identifier(self._table),
                )
            )
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {name} ON {table} USING gin (tsv)").format(
                    name=sql.Identifier(f"{self._table}_gin"),
                    table=sql.Identifier(self._table),
                )
            )
            conn.commit()

    def drop_schema(self) -> None:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(sql.SQL("DROP TABLE IF EXISTS {t}").format(t=sql.Identifier(self._table)))
            conn.commit()

    def count(self) -> int:
        with self.connect() as conn, conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT count(*) AS n FROM {t}").format(t=sql.Identifier(self._table)))
            return cur.fetchone()["n"]
