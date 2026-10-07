from dataclasses import dataclass

import faiss
import numpy as np

from src.ingestion import Chunk


@dataclass
class SearchResult:
    """Represents a retrieved chunk and its similarity score."""

    chunk: Chunk
    score: float


class VectorStore:
    """Stores and searches normalized text embeddings using FAISS."""

    def __init__(self, dimension: int):
        if dimension <= 0:
            raise ValueError("dimension must be greater than zero")

        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: list[Chunk] = []

    def add(
        self,
        embeddings: list[list[float]],
        chunks: list[Chunk],
    ) -> None:
        if len(embeddings) != len(chunks):
            raise ValueError(
                "The number of embeddings must match the number of chunks."
            )

        if not embeddings:
            return

        vectors = np.asarray(embeddings, dtype="float32")

        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embeddings with dimension {self.dimension}, "
                f"but received shape {vectors.shape}."
            )

        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if self.index.ntotal == 0:
            return []

        query = np.asarray(
            [query_embedding],
            dtype="float32",
        )

        if query.shape[1] != self.dimension:
            raise ValueError(
                f"Expected query dimension {self.dimension}, "
                f"but received {query.shape[1]}."
            )

        actual_k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(query, actual_k)

        results = []

        for score, index in zip(scores[0], indices[0]):
            results.append(
                SearchResult(
                    chunk=self.chunks[index],
                    score=float(score),
                )
            )

        return results

    @property
    def size(self) -> int:
        return self.index.ntotal