import logging
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.evidence import EvidenceSelector
from src.generation import AnswerGenerator
from src.observability import RequestMetrics
from src.retrieval import RetrievalPipeline


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


app = FastAPI(
    title="Enterprise Knowledge Intelligence Platform",
    description="Production-oriented RAG API",
    version="0.1.0",
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


class SourceResponse(BaseModel):
    page: int
    score: float
    chunk_id: str


class MetricsResponse(BaseModel):
    request_id: str
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    candidates_retrieved: int
    evidence_selected: int
    top_retrieval_score: float | None
    abstained: bool


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    metrics: MetricsResponse


pipeline = RetrievalPipeline(
    chunk_size=500,
    chunk_overlap=75,
)

evidence_selector = EvidenceSelector(
    minimum_score=0.35,
    relative_score_threshold=0.65,
    max_evidence=3,
)

generator = AnswerGenerator()


CORPUS_PATH = Path("data/raw")

if CORPUS_PATH.exists():
    pipeline.index_corpus(str(CORPUS_PATH))


@app.get("/health")
def health() -> dict:
    return {
        "status": "healthy",
        "service": "enterprise-knowledge-intelligence-platform",
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:

    metrics = RequestMetrics()

    total_start = metrics.start_timer()

    # -----------------------------
    # Retrieval
    # -----------------------------

    retrieval_start = metrics.start_timer()

    results = pipeline.search(
        request.question,
        top_k=request.top_k,
    )

    metrics.retrieval_latency_ms = metrics.elapsed_ms(
        retrieval_start
    )

    metrics.candidates_retrieved = len(results)

    if results:
        metrics.top_retrieval_score = results[0].score

    # -----------------------------
    # Evidence selection
    # -----------------------------

    evidence = evidence_selector.select(results)

    metrics.evidence_selected = len(evidence)

    # -----------------------------
    # Generation
    # -----------------------------

    generation_start = metrics.start_timer()

    generated = generator.generate(
        query=request.question,
        results=evidence,
    )

    metrics.generation_latency_ms = metrics.elapsed_ms(
        generation_start
    )

    metrics.source_pages = sorted(
        {
            result.chunk.page_number
            for result in generated.sources
        }
    )

    metrics.abstained = len(generated.sources) == 0

    metrics.total_latency_ms = metrics.elapsed_ms(
        total_start
    )

    metrics.log()

    sources = [
        SourceResponse(
            page=result.chunk.page_number,
            score=result.score,
            chunk_id=result.chunk.chunk_id,
        )
        for result in generated.sources
    ]

    return AskResponse(
        answer=generated.answer,
        sources=sources,
        metrics=MetricsResponse(
            request_id=metrics.request_id,
            retrieval_latency_ms=metrics.retrieval_latency_ms,
            generation_latency_ms=metrics.generation_latency_ms,
            total_latency_ms=metrics.total_latency_ms,
            candidates_retrieved=metrics.candidates_retrieved,
            evidence_selected=metrics.evidence_selected,
            top_retrieval_score=metrics.top_retrieval_score,
            abstained=metrics.abstained,
        ),
    )