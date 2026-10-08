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

def test_persistent_state_is_invalid_when_vector_store_is_empty(
    tmp_path,
):
    pipeline = RetrievalPipeline(
        chunk_size=500,
        chunk_overlap=75,
        index_directory=str(tmp_path / "vector_store"),
        manifest_database=str(tmp_path / "manifest.db"),
    )

    assert not pipeline.validate_persistent_state()

def test_persistent_state_is_valid_after_indexing(
    tmp_path,
):
    pipeline = RetrievalPipeline(
        chunk_size=500,
        chunk_overlap=75,
        index_directory=str(tmp_path / "vector_store"),
        manifest_database=str(tmp_path / "manifest.db"),
    )

    pipeline.index_document(
        "data/raw/sample-policy.txt"
    )

    assert pipeline.validate_persistent_state()

def test_index_directory_processes_supported_documents(
    tmp_path,
):
    corpus = tmp_path / "corpus"
    corpus.mkdir()

    (corpus / "policy.txt").write_text(
        "Enterprise AI policy content.",
        encoding="utf-8",
    )

    (corpus / "procedure.txt").write_text(
        "Enterprise procedure content.",
        encoding="utf-8",
    )

    (corpus / "ignored.md").write_text(
        "This should not be indexed.",
        encoding="utf-8",
    )

    pipeline = RetrievalPipeline(
        chunk_size=500,
        chunk_overlap=75,
        index_directory=str(
            tmp_path / "vector_store"
        ),
        manifest_database=str(
            tmp_path / "manifest.db"
        ),
    )

    total_chunks = pipeline.index_corpus(
        str(corpus)
    )

    assert total_chunks == 2
    assert pipeline.manifest.size == 2
    assert pipeline.vector_store.size == 2

def test_index_directory_requires_existing_directory(
    tmp_path,
):
    pipeline = RetrievalPipeline(
        chunk_size=500,
        chunk_overlap=75,
        index_directory=str(
            tmp_path / "vector_store"
        ),
        manifest_database=str(
            tmp_path / "manifest.db"
        ),
    )

    missing_directory = tmp_path / "missing"

    try:
        pipeline.index_corpus(
            str(missing_directory)
        )
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass