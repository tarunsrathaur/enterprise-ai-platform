from pathlib import Path

from src.chunking import TextChunker
from src.embeddings import EmbeddingService
from src.ingestion import DocumentLoader
from src.manifest import DocumentManifest
from src.vector_store import SearchResult, VectorStore


class RetrievalPipeline:
    """End-to-end document ingestion and semantic retrieval pipeline."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 75,
        index_directory: str = "data/processed/vector_store",
        manifest_database: str = "data/processed/manifest.db",
    ):
        self.loader = DocumentLoader()
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.embedding_service = EmbeddingService()

        self.index_directory = index_directory

        index_path = Path(index_directory)

        if (
            index_path.exists()
            and (index_path / "index.faiss").exists()
            and (index_path / "metadata.pkl").exists()
        ):
            self.vector_store = VectorStore.load(
                index_directory
            )
        else:
            self.vector_store = VectorStore(
                dimension=self.embedding_service.dimension
            )

        self.manifest = DocumentManifest(
            database_path=manifest_database
        )

    def index_document(self, file_path: str) -> int:
        """Load, chunk and index a new or changed document."""

        document = self.loader.load(file_path)

        existing_record = self.manifest.get(
            document.document_id
        )

        # Document already exists and has not changed.
        if existing_record is not None:
            if existing_record.content_hash == document.content_hash:
                return 0

            # Document has changed.
            self.vector_store.remove_document(
                document.document_id
            )

        chunks = self.chunker.chunk_document(document)

        if not chunks:
            return 0

        embeddings = self.embedding_service.embed_texts(
            [chunk.text for chunk in chunks]
        )

        self.vector_store.add(
            embeddings,
            chunks,
        )

        self.vector_store.save(
            self.index_directory
        )

        self.manifest.register(
            document_id=document.document_id,
            file_name=document.file_name,
            content_hash=document.content_hash,
            chunk_count=len(chunks),
        )

        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Retrieve the most relevant chunks for a query."""

        query_embedding = self.embedding_service.embed_text(
            query
        )

        return self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )