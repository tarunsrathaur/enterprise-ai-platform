from dataclasses import dataclass
from pathlib import Path

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
        self.embeddings: list[list[float]] = []

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
        self.embeddings.extend(vectors.tolist())

    def remove_document(self, document_id: str) -> int:
        """Remove all chunks and vectors belonging to a document."""

        keep_indices = [
            index
            for index, chunk in enumerate(self.chunks)
            if chunk.document_id != document_id
        ]

        removed_count = len(self.chunks) - len(keep_indices)

        if removed_count == 0:
            return 0

        self.chunks = [
            self.chunks[index]
            for index in keep_indices
        ]

        self.embeddings = [
            self.embeddings[index]
            for index in keep_indices
        ]

        self.index = faiss.IndexFlatIP(self.dimension)

        if self.embeddings:
            vectors = np.asarray(
                self.embeddings,
                dtype="float32",
            )
            self.index.add(vectors)

        return removed_count

    def save(self, directory: str | Path) -> None:
        """Persist the FAISS index and associated metadata."""

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(
            self.index,
            str(directory / "index.faiss"),
        )

        metadata = {
            "dimension": self.dimension,
            "chunks": self.chunks,
            "embeddings": self.embeddings,
        }

        import pickle

        with open(directory / "metadata.pkl", "wb") as file:
            pickle.dump(metadata, file)

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        """Load a persisted vector store."""

        directory = Path(directory)

        index_path = directory / "index.faiss"
        metadata_path = directory / "metadata.pkl"

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}"
            )

        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Vector metadata not found: {metadata_path}"
            )

        import pickle

        with open(metadata_path, "rb") as file:
            metadata = pickle.load(file)

        store = cls(
            dimension=metadata["dimension"],
        )

        store.index = faiss.read_index(
            str(index_path)
        )

        store.chunks = metadata["chunks"]
        store.embeddings = metadata["embeddings"]

        return store

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

        scores, indices = self.index.search(
            query,
            actual_k,
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):
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

    @property
    def is_empty(self) -> bool:
        """Return True when the vector store contains no vectors."""
        return self.index.ntotal == 0