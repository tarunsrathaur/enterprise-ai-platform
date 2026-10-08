from src.manifest import DocumentManifest


def test_new_document_requires_ingestion(tmp_path):
    manifest = DocumentManifest(
        database_path=str(tmp_path / "manifest.db")
    )

    assert manifest.needs_ingestion(
        document_id="policy",
        content_hash="abc123",
    )


def test_same_document_hash_does_not_require_ingestion(tmp_path):
    manifest = DocumentManifest(
        database_path=str(tmp_path / "manifest.db")
    )

    manifest.register(
        document_id="policy",
        file_name="policy.txt",
        content_hash="abc123",
        chunk_count=4,
    )

    assert not manifest.needs_ingestion(
        document_id="policy",
        content_hash="abc123",
    )


def test_changed_document_hash_requires_ingestion(tmp_path):
    manifest = DocumentManifest(
        database_path=str(tmp_path / "manifest.db")
    )

    manifest.register(
        document_id="policy",
        file_name="policy.txt",
        content_hash="abc123",
        chunk_count=4,
    )

    assert manifest.needs_ingestion(
        document_id="policy",
        content_hash="changed456",
    )


def test_manifest_stores_chunk_count(tmp_path):
    manifest = DocumentManifest(
        database_path=str(tmp_path / "manifest.db")
    )

    manifest.register(
        document_id="policy",
        file_name="policy.txt",
        content_hash="abc123",
        chunk_count=4,
    )

    record = manifest.get("policy")

    assert record is not None
    assert record.chunk_count == 4
    assert record.file_name == "policy.txt"