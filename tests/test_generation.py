from src.generation import AnswerGenerator
from src.ingestion import Chunk
from src.vector_store import SearchResult


def make_result(page: int, score: float = 0.8) -> SearchResult:
    chunk = Chunk(
        chunk_id=f"chunk-{page}",
        document_id="test-document",
        source="test.pdf",
        page_number=page,
        chunk_index=0,
        text="Test content.",
    )

    return SearchResult(
        chunk=chunk,
        score=score,
    )


def test_no_evidence_returns_abstention_message():
    generator = AnswerGenerator()

    result = generator.generate(
        query="What is the answer?",
        results=[],
    )

    assert (
        result.answer
        == "I could not find enough information in the provided documents."
    )

    assert result.sources == []


def test_generated_answer_preserves_source_pages(monkeypatch):
    class FakeResponse:
        def __getitem__(self, key):
            if key == "message":
                return {
                    "content": "The answer is available in the document."
                }
            raise KeyError(key)

    monkeypatch.setattr(
        "src.generation.ollama.chat",
        lambda **kwargs: FakeResponse(),
    )

    generator = AnswerGenerator()

    result = generator.generate(
        query="What is the answer?",
        results=[
            make_result(5),
            make_result(4),
        ],
    )

    assert "[Page 4]" in result.answer
    assert "[Page 5]" in result.answer

    assert len(result.sources) == 2