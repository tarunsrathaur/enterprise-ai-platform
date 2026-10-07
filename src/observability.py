import logging
import time
from dataclasses import dataclass, field
from uuid import uuid4


logger = logging.getLogger("enterprise_ai")


@dataclass
class RequestMetrics:
    """Captures operational metrics for one RAG request."""

    request_id: str = field(default_factory=lambda: str(uuid4()))
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    candidates_retrieved: int = 0
    evidence_selected: int = 0
    top_retrieval_score: float | None = None
    abstained: bool = False
    source_pages: list[int] = field(default_factory=list)

    def start_timer(self) -> float:
        return time.perf_counter()

    def elapsed_ms(self, start_time: float) -> float:
        return (time.perf_counter() - start_time) * 1000

    def log(self) -> None:
        logger.info(
            "RAG request completed | "
            "request_id=%s "
            "retrieval_ms=%.2f "
            "generation_ms=%.2f "
            "total_ms=%.2f "
            "candidates=%d "
            "evidence=%d "
            "top_score=%s "
            "abstained=%s "
            "pages=%s",
            self.request_id,
            self.retrieval_latency_ms,
            self.generation_latency_ms,
            self.total_latency_ms,
            self.candidates_retrieved,
            self.evidence_selected,
            (
                f"{self.top_retrieval_score:.4f}"
                if self.top_retrieval_score is not None
                else "None"
            ),
            self.abstained,
            self.source_pages,
        )