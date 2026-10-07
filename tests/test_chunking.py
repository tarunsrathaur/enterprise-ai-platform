from src.chunking import TextChunker
from src.ingestion import Document, Page


def create_test_document() -> Document:
    return Document(
        document_id="test-document",
        source="data/raw/test.txt",
        file_type=".txt",
        pages=[
            Page(
                page_number=1,
                content=(
                    "First paragraph contains important information.\n\n"
                    "Second paragraph contains additional information.\n\n"
                    "Third paragraph contains more information."
                ),
            )
        ],
    )


def test_chunk_document_creates_chunks():
    document = create_test_document()

    chunker = TextChunker(
        chunk_size=80,
        chunk_overlap=10,
    )

    chunks = chunker.chunk_document(document)

    assert len(chunks) > 1


def test_chunk_preserves_provenance():
    document = create_test_document()

    chunker = TextChunker(
        chunk_size=80,
        chunk_overlap=10,
    )

    chunks = chunker.chunk_document(document)

    assert all(chunk.document_id == "test-document" for chunk in chunks)
    assert all(chunk.source == "data/raw/test.txt" for chunk in chunks)
    assert all(chunk.page_number == 1 for chunk in chunks)


def test_chunk_ids_are_unique():
    document = create_test_document()

    chunker = TextChunker(
        chunk_size=80,
        chunk_overlap=10,
    )

    chunks = chunker.chunk_document(document)

    chunk_ids = [chunk.chunk_id for chunk in chunks]

    assert len(chunk_ids) == len(set(chunk_ids))


def test_invalid_chunk_configuration():
    try:
        TextChunker(chunk_size=100, chunk_overlap=100)
        assert False
    except ValueError:
        assert True

def test_large_paragraph_is_split():
    large_paragraph = "A" * 250

    document = Document(
        document_id="large-document",
        source="data/raw/large.txt",
        file_type=".txt",
        pages=[
            Page(
                page_number=1,
                content=large_paragraph,
            )
        ],
    )

    chunker = TextChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    chunks = chunker.chunk_document(document)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 100 for chunk in chunks)