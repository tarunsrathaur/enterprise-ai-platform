import re

from src.ingestion import Chunk, Document


class TextChunker:
    """Splits document pages into deterministic, size-bounded chunks."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 75,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, document: Document) -> list[Chunk]:
        chunks = []
        global_chunk_index = 0

        for page in document.pages:
            paragraphs = self._split_into_paragraphs(page.content)
            page_chunks = self._chunk_paragraphs(paragraphs)

            for text in page_chunks:
                chunks.append(
                    Chunk(
                        chunk_id=(
                            f"{document.document_id}-"
                            f"p{page.page_number}-"
                            f"c{global_chunk_index}"
                        ),
                        document_id=document.document_id,
                        source=document.source,
                        page_number=page.page_number,
                        chunk_index=global_chunk_index,
                        text=text,
                    )
                )

                global_chunk_index += 1

        return chunks

    def _split_into_paragraphs(self, text: str) -> list[str]:
        paragraphs = re.split(r"\n\s*\n", text)

        return [
            paragraph.strip()
            for paragraph in paragraphs
            if paragraph.strip()
        ]

    def _chunk_paragraphs(self, paragraphs: list[str]) -> list[str]:
        chunks = []
        current = ""

        for paragraph in paragraphs:
            # Normal case: paragraph fits into current chunk.
            if len(current) + len(paragraph) + 2 <= self.chunk_size:
                current = (
                    paragraph
                    if not current
                    else f"{current}\n\n{paragraph}"
                )
                continue

            # Save the current chunk before starting another.
            if current:
                chunks.append(current)

                overlap = current[-self.chunk_overlap:]
            else:
                overlap = ""

            # If the paragraph itself is too large, split it.
            if len(paragraph) > self.chunk_size:
                paragraph_chunks = self._split_large_text(paragraph)

                if overlap:
                    paragraph_chunks[0] = (
                        f"{overlap}\n\n{paragraph_chunks[0]}"
                    )

                chunks.extend(paragraph_chunks[:-1])

                current = paragraph_chunks[-1]
            else:
                current = (
                    f"{overlap}\n\n{paragraph}"
                    if overlap
                    else paragraph
                )

        if current:
            chunks.append(current)

        return chunks

    def _split_large_text(self, text: str) -> list[str]:
        chunks = []

        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            chunks.append(text[start:end])

            if end >= text_length:
                break

            start = end - self.chunk_overlap

        return chunks