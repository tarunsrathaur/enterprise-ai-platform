
import hashlib
import json
import re
from pathlib import Path


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_coursework_manifest(
    manifest_path: str | Path,
    corpus_root: str | Path,
    expected_count: int = 8,
) -> list[dict]:
    manifest_path = Path(manifest_path)
    corpus_root = Path(corpus_root).resolve(strict=True)

    if not corpus_root.is_dir():
        raise ValueError("Coursework corpus root must be a directory.")

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema version.")

    documents = manifest.get("documents")

    if not isinstance(documents, list):
        raise ValueError("Manifest documents must be a list.")

    if len(documents) != expected_count:
        raise ValueError(
            f"Expected {expected_count} documents; found {len(documents)}."
        )

    if manifest.get("source_file_count") != len(documents):
        raise ValueError("Manifest source_file_count does not match documents.")

    seen_ids = set()
    seen_hashes = set()

    for document in documents:
        if not isinstance(document, dict):
            raise ValueError("Each document entry must be an object.")

        filename = document.get("file_name")

        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
            or "/" in filename
            or "\\" in filename
            or not filename.lower().endswith(".pdf")
        ):
            raise ValueError(f"Invalid PDF filename: {filename!r}")

        path = corpus_root / filename

        if not path.is_file():
            raise FileNotFoundError(f"Approved PDF not found: {filename}")

        # Resolve paths to prevent symlinks escaping the approved corpus.
        if path.resolve(strict=True).parent != corpus_root:
            raise ValueError(f"PDF resolves outside corpus root: {filename}")

        content_hash = document.get("sha256")

        if (
            not isinstance(content_hash, str)
            or not _SHA256_PATTERN.fullmatch(content_hash)
        ):
            raise ValueError(f"Invalid SHA-256 for {filename}")

        if content_hash in seen_hashes:
            raise ValueError(f"Duplicate content hash: {filename}")

        if sha256_file(path) != content_hash:
            raise ValueError(f"SHA-256 mismatch: {filename}")

        document_id = document.get("document_id")

        if document_id != content_hash[:16]:
            raise ValueError(f"Document ID mismatch: {filename}")

        if document_id in seen_ids:
            raise ValueError(f"Duplicate document ID: {document_id}")

        if document.get("approved") is not True:
            raise ValueError(f"Document is not approved: {filename}")

        for field in ("page_count", "extracted_characters"):
            value = document.get(field)
            if type(value) is not int or value < 0:
                raise ValueError(f"Invalid {field} for {filename}")

        if document["page_count"] < 1:
            raise ValueError(f"PDF has no pages: {filename}")

        seen_ids.add(document_id)
        seen_hashes.add(content_hash)

    return documents
