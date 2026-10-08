from src.retrieval import RetrievalPipeline


def test_same_document_is_not_indexed_twice(tmp_path):
    pipeline = RetrievalPipeline(
        chunk_size=500,
        chunk_overlap=75,
        index_directory=str(tmp_path / "vector_store"),
        manifest_database=str(tmp_path / "manifest.db"),
    )

    first_count = pipeline.index_document(
        "data/raw/sample-policy.txt"
    )

    second_count = pipeline.index_document(
        "data/raw/sample-policy.txt"
    )

    assert first_count > 0
    assert second_count == 0
    assert pipeline.manifest.size == 1
    
def test_indexed_document_is_registered_in_manifest(tmp_path):
    pipeline = RetrievalPipeline(
        chunk_size=500,
        chunk_overlap=75,
        index_directory=str(tmp_path / "vector_store"),
        manifest_database=str(tmp_path / "manifest.db"),
    )

    chunk_count = pipeline.index_document(
        "data/raw/sample-policy.txt"
    )

    record = pipeline.manifest.get("sample-policy")

    assert record is not None
    assert record.file_name == "sample-policy.txt"
    assert record.content_hash
    assert record.chunk_count == chunk_count

def test_changed_document_replaces_old_vectors(tmp_path):
    document_path = tmp_path / "policy.txt"

    document_path.write_text(
        "Original policy content.",
        encoding="utf-8",
    )

    pipeline = RetrievalPipeline(
        chunk_size=500,
        chunk_overlap=75,
        index_directory=str(tmp_path / "vector_store"),
        manifest_database=str(tmp_path / "manifest.db"),
    )

    first_count = pipeline.index_document(
        str(document_path)
    )

    assert first_count == 1
    assert pipeline.vector_store.size == 1

    document_path.write_text(
        "Updated policy content.",
        encoding="utf-8",
    )

    second_count = pipeline.index_document(
        str(document_path)
    )

    assert second_count == 1
    assert pipeline.vector_store.size == 1

    record = pipeline.manifest.get("policy")

    assert record is not None
    assert record.content_hash
    assert record.chunk_count == 1