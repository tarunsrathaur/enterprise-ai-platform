from src.ingestion import Chunk
from src.vector_store import VectorStore


def create_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            source="policy.pdf",
            page_number=1,
            chunk_index=0,
            text="AI security requirements.",
        ),
        Chunk(
            chunk_id="chunk-2",
            document_id="doc-1",
            source="policy.pdf",
            page_number=2,
            chunk_index=1,
            text="Access control requirements.",
        ),
        Chunk(
            chunk_id="chunk-3",
            document_id="doc-1",
            source="policy.pdf",
            page_number=3,
            chunk_index=2,
            text="Logging requirements.",
        ),
    ]


def test_add_embeddings():
    store = VectorStore(dimension=3)

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    store.add(embeddings, create_chunks())

    assert store.size == 3


def test_search_returns_most_similar_chunk():
    store = VectorStore(dimension=3)

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]

    store.add(embeddings, create_chunks())

    results = store.search(
        query_embedding=[0.95, 0.05, 0.0],
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "chunk-1"
    assert results[0].score > results[1].score


def test_search_preserves_chunk_metadata():
    store = VectorStore(dimension=3)

    store.add(
        [[1.0, 0.0, 0.0]],
        [create_chunks()[0]],
    )

    results = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=1,
    )

    result = results[0]

    assert result.chunk.document_id == "doc-1"
    assert result.chunk.source == "policy.pdf"
    assert result.chunk.page_number == 1
    assert result.chunk.text == "AI security requirements."


def test_empty_store_returns_no_results():
    store = VectorStore(dimension=3)

    results = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=5,
    )

    assert results == []