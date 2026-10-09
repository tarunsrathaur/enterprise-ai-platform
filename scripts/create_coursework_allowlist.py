
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pymupdf


ROOT = Path("data/coursework")
OUTPUT = ROOT / "approved_manifest.json"

APPROVED_FILES = [
    "Cheat_Sheet_ML.pdf",
    "Cheat_Sheet_MFML.pdf",
    "Cheat_Sheet_ISM.pdf",
    "Lecture_1_Companion.pdf",
    "Lecture_2_Companion.pdf",
    "MFML Mid-Semester Study Guide.pdf",
    "ML Mid-Semester Study Guide.pdf",
    "Mathematical Foundations for Machine Learning (S2-25_AIMLCZC416) COURSE HANDOUT.pdf",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not ROOT.is_dir():
        raise FileNotFoundError(f"Coursework directory not found: {ROOT}")

    documents = []

    for filename in APPROVED_FILES:
        path = ROOT / filename

        if not path.is_file():
            raise FileNotFoundError(f"Approved PDF not found: {path}")

        with pymupdf.open(path) as pdf:
            page_count = len(pdf)
            extracted_characters = sum(
                len(page.get_text().strip()) for page in pdf
            )

        content_hash = sha256_file(path)

        documents.append({
            "document_id": content_hash[:16],
            "file_name": filename,
            "sha256": content_hash,
            "page_count": page_count,
            "extracted_characters": extracted_characters,
            "approved": True,
        })

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_file_count": len(APPROVED_FILES),
        "documents": documents,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Manifest created: {OUTPUT}")
    print(f"Approved documents: {len(documents)}")
    for document in documents:
        print(
            f'{document["document_id"]} | '
            f'{document["page_count"]} pages | '
            f'{document["file_name"]}'
        )


if __name__ == "__main__":
    main()
