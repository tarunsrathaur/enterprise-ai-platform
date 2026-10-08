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

def test_remove_document_removes_only_matching_chunks():
    store = VectorStore(dimension=2)

    chunks = [
        Chunk(
            chunk_id="doc-a-1",
            document_id="doc-a",
            source="a.txt",
            page_number=1,
            chunk_index=0,
            text="Document A content",
        ),
        Chunk(
            chunk_id="doc-b-1",
            document_id="doc-b",
            source="b.txt",
            page_number=1,
            chunk_index=0,
            text="Document B content",
        ),
    ]

    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    store.add(embeddings, chunks)

    removed = store.remove_document("doc-a")

    assert removed == 1
    assert store.size == 1
    assert store.chunks[0].document_id == "doc-b"

def test_remove_document_returns_zero_when_document_does_not_exist():
    store = VectorStore(dimension=2)

    removed = store.remove_document("missing-document")

    assert removed == 0
    assert store.size == 0

def test_vector_store_can_be_saved_and_loaded(tmp_path):
    store = VectorStore(dimension=2)

    chunks = [
        Chunk(
            chunk_id="doc-a-1",
            document_id="doc-a",
            source="a.txt",
            page_number=1,
            chunk_index=0,
            text="Document A content",
        ),
        Chunk(
            chunk_id="doc-b-1",
            document_id="doc-b",
            source="b.txt",
            page_number=1,
            chunk_index=0,
            text="Document B content",
        ),
    ]

    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    store.add(embeddings, chunks)

    store.save(tmp_path)

    loaded_store = VectorStore.load(tmp_path)

    assert loaded_store.size == 2
    assert loaded_store.dimension == 2
    assert len(loaded_store.chunks) == 2
    assert loaded_store.chunks[0].chunk_id == "doc-a-1"
    assert loaded_store.chunks[1].chunk_id == "doc-b-1"

def test_loaded_vector_store_can_search(tmp_path):
    store = VectorStore(dimension=2)

    chunk = Chunk(
        chunk_id="doc-a-1",
        document_id="doc-a",
        source="a.txt",
        page_number=1,
        chunk_index=0,
        text="Document A content",
    )

    store.add(
        [[1.0, 0.0]],
        [chunk],
    )

    store.save(tmp_path)

    loaded_store = VectorStore.load(tmp_path)

    results = loaded_store.search(
        [1.0, 0.0],
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "doc-a-1"
    assert results[0].score > 0.99

def test_vector_store_is_empty():
    store = VectorStore(dimension=2)

    assert store.is_empty

    store.add(
        [[1.0, 0.0]],
        [
            Chunk(
                chunk_id="doc-a-1",
                document_id="doc-a",
                source="a.txt",
                page_number=1,
                chunk_index=0,
                text="Document A",
            )
        ],
    )

    assert not store.is_empty