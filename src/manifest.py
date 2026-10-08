import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DocumentRecord:
    """Tracks the indexed state of a document."""

    document_id: str
    file_name: str
    content_hash: str
    chunk_count: int


class DocumentManifest:
    """Persistent document manifest backed by SQLite."""

    def __init__(
        self,
        database_path: str = "data/processed/manifest.db",
    ) -> None:
        self.database_path = Path(database_path)

        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path)

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL
                )
                """
            )

    def get(
        self,
        document_id: str,
    ) -> DocumentRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    document_id,
                    file_name,
                    content_hash,
                    chunk_count
                FROM documents
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()

        if row is None:
            return None

        return DocumentRecord(
            document_id=row[0],
            file_name=row[1],
            content_hash=row[2],
            chunk_count=row[3],
        )

    def needs_ingestion(
        self,
        document_id: str,
        content_hash: str,
    ) -> bool:
        record = self.get(document_id)

        if record is None:
            return True

        return record.content_hash != content_hash

    def register(
        self,
        document_id: str,
        file_name: str,
        content_hash: str,
        chunk_count: int,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    document_id,
                    file_name,
                    content_hash,
                    chunk_count
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(document_id)
                DO UPDATE SET
                    file_name = excluded.file_name,
                    content_hash = excluded.content_hash,
                    chunk_count = excluded.chunk_count
                """,
                (
                    document_id,
                    file_name,
                    content_hash,
                    chunk_count,
                ),
            )

    @property
    def size(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM documents"
            ).fetchone()

        return int(row[0])