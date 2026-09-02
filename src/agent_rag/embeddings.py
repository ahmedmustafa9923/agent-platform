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