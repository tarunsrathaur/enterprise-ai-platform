from src.chunking import TextChunker
from src.embeddings import EmbeddingService
from src.ingestion import DocumentLoader
from src.vector_store import SearchResult, VectorStore


class RetrievalPipeline:
    """End-to-end document ingestion and semantic retrieval pipeline."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 75,
    ):
        self.loader = DocumentLoader()
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.embedding_service = EmbeddingService()

        self.vector_store = VectorStore(
            dimension=self.embedding_service.dimension
        )

    def index_document(self, file_path: str) -> int:
        """Load, chunk and index a document."""

        document = self.loader.load(file_path)

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