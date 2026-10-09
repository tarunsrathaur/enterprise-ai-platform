
import hashlib
import json

import pytest

from src.coursework_manifest import validate_coursework_manifest


def create_manifest(tmp_path, count=8):
    root = tmp_path / "coursework"
    root.mkdir()

    documents = []

    for i in range(count):
        filename = f"document_{i}.pdf"
        content = f"private test PDF placeholder {i}".encode()
        (root / filename).write_bytes(content)

        digest = hashlib.sha256(content).hexdigest()
        documents.append({
            "document_id": digest[:16],
            "file_name": filename,
            "sha256": digest,
            "page_count": 1,
            "extracted_characters": len(content),
            "approved": True,
        })

    manifest = {
        "schema_version": 1,
        "source_file_count": count,
        "documents": documents,
    }

    manifest_path = root / "approved_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    return root, manifest_path, manifest


def test_valid_manifest_passes(tmp_path):
    root, manifest_path, _ = create_manifest(tmp_path)

    result = validate_coursework_manifest(manifest_path, root)

    assert len(result) == 8


def test_changed_pdf_fails_hash_validation(tmp_path):
    root, manifest_path, manifest = create_manifest(tmp_path)
    (root / manifest["documents"][0]["file_name"]).write_bytes(b"changed")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_coursework_manifest(manifest_path, root)


def test_missing_pdf_is_rejected(tmp_path):
    root, manifest_path, manifest = create_manifest(tmp_path)
    (root / manifest["documents"][0]["file_name"]).unlink()

    with pytest.raises(FileNotFoundError):
        validate_coursework_manifest(manifest_path, root)


def test_wrong_document_count_is_rejected(tmp_path):
    root, manifest_path, _ = create_manifest(tmp_path, count=7)

    with pytest.raises(ValueError, match="Expected 8 documents"):
        validate_coursework_manifest(manifest_path, root)


def test_path_traversal_is_rejected(tmp_path):
    root, manifest_path, manifest = create_manifest(tmp_path)
    manifest["documents"][0]["file_name"] = "../outside.pdf"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid PDF filename"):
        validate_coursework_manifest(manifest_path, root)


def test_unapproved_document_is_rejected(tmp_path):
    root, manifest_path, manifest = create_manifest(tmp_path)
    manifest["documents"][0]["approved"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="not approved"):
        validate_coursework_manifest(manifest_path, root)
