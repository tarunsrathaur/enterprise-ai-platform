from src.evidence import EvidenceSelector
from src.ingestion import Chunk
from src.vector_store import SearchResult


def make_result(page: int, score: float) -> SearchResult:
    chunk = Chunk(
        chunk_id=f"chunk-{page}-{score}",
        document_id="test-document",
        source="test.pdf",
        page_number=page,
        chunk_index=0,
        text=f"Test content for page {page}",
    )

    return SearchResult(
        chunk=chunk,
        score=score,
    )


def test_selects_high_confidence_evidence():
    selector = EvidenceSelector(
        minimum_score=0.35,
        relative_score_threshold=0.65,
    )

    results = [
        make_result(5, 0.68),
        make_result(1, 0.40),
        make_result(4, 0.38),
    ]

    selected = selector.select(results)

    assert len(selected) == 1
    assert selected[0].chunk.page_number == 5


def test_preserves_multiple_relevant_chunks():
    selector = EvidenceSelector(
        minimum_score=0.35,
        relative_score_threshold=0.65,
    )

    results = [
        make_result(5, 0.64),
        make_result(5, 0.48),
        make_result(1, 0.20),
    ]

    selected = selector.select(results)

    assert len(selected) == 2
    assert all(result.chunk.page_number == 5 for result in selected)


def test_rejects_low_confidence_retrieval():
    selector = EvidenceSelector(
        minimum_score=0.35,
        relative_score_threshold=0.65,
    )

    results = [
        make_result(1, 0.2463),
        make_result(5, 0.2351),
        make_result(1, 0.2213),
    ]

    selected = selector.select(results)

    assert selected == []


def test_empty_results_return_empty_evidence():
    selector = EvidenceSelector()

    selected = selector.select([])

    assert selected == []