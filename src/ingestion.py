from dataclasses import dataclass
from pathlib import Path


@dataclass
class Page:
    """Represents a single page extracted from a document."""

    page_number: int
    content: str

@dataclass
class Chunk:
    """Represents a retrievable unit of document content."""

    chunk_id: str
    document_id: str
    source: str
    page_number: int
    chunk_index: int
    text: str
    
@dataclass
class Document:
    """Represents a source document loaded into the ingestion pipeline."""

    document_id: str
    source: str
    file_type: str
    pages: list[Page]

    @property
    def content(self) -> str:
        """Return all page content as a single string."""
        return "\n".join(page.content for page in self.pages)


class DocumentLoader:
    """Loads supported documents from the local filesystem."""

    SUPPORTED_EXTENSIONS = {".txt", ".pdf"}

    def load(self, file_path: str | Path) -> Document:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        extension = path.suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported types: {self.SUPPORTED_EXTENSIONS}"
            )

        if extension == ".txt":
            pages = self._load_text(path)
        else:
            pages = self._load_pdf(path)

        return Document(
            document_id=path.stem,
            source=str(path),
            file_type=extension,
            pages=pages,
        )

    def _load_text(self, path: Path) -> list[Page]:
        content = path.read_text(encoding="utf-8")

        return [
            Page(
                page_number=1,
                content=content,
            )
        ]

    def _load_pdf(self, path: Path) -> list[Page]:
        import pymupdf

        pages = []

        with pymupdf.open(path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                pages.append(
                    Page(
                        page_number=page_number,
                        content=page.get_text(),
                    )
                )

        return pages