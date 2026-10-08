from pathlib import Path

import pytest

from src.ingestion import DocumentLoader


SAMPLE_DOCUMENT = Path("data/raw/sample-policy.txt")


def test_load_text_document():
    loader = DocumentLoader()

    document = loader.load(SAMPLE_DOCUMENT)

    assert document.document_id == "sample-policy"
    assert document.file_type == ".txt"

    assert len(document.pages) == 1
    assert document.pages[0].page_number == 1

    assert "Enterprise AI Access Policy" in document.pages[0].content


def test_document_content_property():
    loader = DocumentLoader()

    document = loader.load(SAMPLE_DOCUMENT)

    assert "Enterprise AI Access Policy" in document.content


def test_missing_document_raises_error():
    loader = DocumentLoader()

    with pytest.raises(FileNotFoundError):
        loader.load("data/raw/does-not-exist.txt")


def test_unsupported_file_type_raises_error(tmp_path):
    unsupported_file = tmp_path / "sample.docx"
    unsupported_file.write_text("test content", encoding="utf-8")

    loader = DocumentLoader()

    with pytest.raises(ValueError):
        loader.load(unsupported_file)

def test_load_pdf_preserves_page_numbers():
    loader = DocumentLoader()

    document = loader.load("data/raw/FAQ.pdf")

    assert document.file_type == ".pdf"
    assert len(document.pages) > 0

    page_numbers = [page.page_number for page in document.pages]

    assert page_numbers == list(range(1, len(document.pages) + 1))

    assert all(page.content.strip() for page in document.pages)

def test_document_contains_metadata():
    loader = DocumentLoader()

    document = loader.load("data/raw/sample-policy.txt")

    assert document.file_name == "sample-policy.txt"
    assert document.file_type == ".txt"
    assert document.content_hash
    assert len(document.content_hash) == 64
    assert document.ingestion_timestamp
    assert document.page_count == 1