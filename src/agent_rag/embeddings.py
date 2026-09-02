import hashlib
import math
import re
from typing import Protocol, Sequence, runtime_checkable

DEFAULT_DIM = 768

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


@runtime_checkable
class Embedder(Protocol):
    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class LocalHashingEmbedder:
    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        if dim <= 0:
            raise ValueError("embedding dim must be positive")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _features(self, text: str) -> list[str]:
        tokens = _TOKEN_RE.findall(text.lower())
        bigrams = [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        return tokens + bigrams

    def _embed_one(self, text: str) -> list[float]:
        counts: dict[str, int] = {}
        for feature in self._features(text):
            counts[feature] = counts.get(feature, 0) + 1

        vector = [0.0] * self._dim
        for feature, count in counts.items():
            weight = 1.0 + math.log(count)
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self._dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign * weight

        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return vector
        return [v / norm for v in vector]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)

class VertexEmbedder:
    _BATCH = 1

    def __init__(self, model="gemini-embedding-001", dim=768, project=None, location="us-central1"):
        self._model_name = model
        self._dim = dim
        self._project = project
        self._location = location
        self._client = None

    @property
    def dim(self) -> int:
        return self._dim

    def _ensure_client(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(
                vertexai=True, project=self._project, location=self._location
            )
        return self._client

    def _normalize(self, values):
        norm = math.sqrt(sum(v * v for v in values))
        if norm == 0.0:
            return list(values)
        return [v / norm for v in values]

    def _embed(self, texts, task_type):
        from google.genai.types import EmbedContentConfig

        client = self._ensure_client()
        config = EmbedContentConfig(task_type=task_type, output_dimensionality=self._dim)

        out = []
        for start in range(0, len(texts), self._BATCH):
            batch = list(texts[start:start + self._BATCH])
            response = client.models.embed_content(
                model=self._model_name, contents=batch, config=config
            )
            out.extend(self._normalize(e.values) for e in response.embeddings)
        return out

    def embed_documents(self, texts):
        if not texts:
            return []
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text):
        return self._embed([text], "RETRIEVAL_QUERY")[0]
