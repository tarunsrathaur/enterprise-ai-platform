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
                    "content": "The answer is available in the document. [SOURCE 1] [SOURCE 2]"
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

    assert "[test.pdf, Page 4]" in result.answer
    assert "[test.pdf, Page 5]" in result.answer

    assert len(result.sources) == 2


def test_citations_identify_source_document_and_page(monkeypatch):
    class FakeResponse:
        def __getitem__(self, key):
            if key == "message":
                return {
                    "content": (
                        "Gradient descent updates model parameters. "
                        "[SOURCE 1] [SOURCE 2]"
                    )
                }
            raise KeyError(key)

    monkeypatch.setattr(
        "src.generation.ollama.chat",
        lambda **kwargs: FakeResponse(),
    )

    generator = AnswerGenerator()

    first = make_result(3)
    first.chunk.source = "data/coursework/ML Study Guide.pdf"

    second = make_result(3)
    second.chunk.source = "data/coursework/ML Cheat Sheet.pdf"

    result = generator.generate(
        query="How are model parameters updated?",
        results=[first, second],
    )

    assert "[ML Study Guide.pdf, Page 3]" in result.answer
    assert "[ML Cheat Sheet.pdf, Page 3]" in result.answer


def test_explicit_abstention_returns_no_sources_or_citations(monkeypatch):
    class FakeResponse:
        def __getitem__(self, key):
            if key == "message":
                return {
                    "content": (
                        "The disciplinary action is not explicitly stated "
                        "in the provided document."
                    )
                }
            raise KeyError(key)

    monkeypatch.setattr(
        "src.generation.ollama.chat",
        lambda **kwargs: FakeResponse(),
    )

    generator = AnswerGenerator()
    result = generator.generate(
        query=(
            "What disciplinary action applies if confidential information "
            "is entered into a public AI service?"
        ),
        results=[make_result(1)],
    )

    assert result.answer == (
        "I could not find enough information in the provided documents."
    )
    assert result.sources == []


def test_unrelated_answer_is_not_treated_as_abstention(monkeypatch):
    class FakeResponse:
        def __getitem__(self, key):
            if key == "message":
                return {
                    "content": (
                        "All production AI systems must maintain logging "
                        "and monitoring. [SOURCE 1]"
                    )
                }
            raise KeyError(key)

    monkeypatch.setattr(
        "src.generation.ollama.chat",
        lambda **kwargs: FakeResponse(),
    )

    generator = AnswerGenerator()
    result = generator.generate(
        query="Which cloud provider is mandated?",
        results=[make_result(1)],
    )

    assert "logging and monitoring" in result.answer
    assert "[test.pdf, Page 1]" in result.answer
    assert len(result.sources) == 1

def test_generated_answer_cites_only_model_identified_supporting_source(
    monkeypatch,
):
    class FakeResponse:
        def __getitem__(self, key):
            if key == "message":
                return {
                    "content": (
                        "The answer is stated in the first document. "
                        "[SOURCE 1]"
                    )
                }
            raise KeyError(key)

    monkeypatch.setattr(
        "src.generation.ollama.chat",
        lambda **kwargs: FakeResponse(),
    )

    generator = AnswerGenerator()

    supporting_result = make_result(5)
    supporting_result.chunk.text = (
        "MS Teams is not an approved platform for academic queries."
    )

    unrelated_result = make_result(4)
    unrelated_result.chunk.text = (
        "The examination marking scheme is available on the LMS."
    )

    result = generator.generate(
        query="Is MS Teams approved for academic queries?",
        results=[supporting_result, unrelated_result],
    )

    assert "[test.pdf, Page 5]" in result.answer
    assert "[test.pdf, Page 4]" not in result.answer
    assert len(result.sources) == 1
    assert result.sources[0].chunk.page_number == 5


def test_supported_answer_is_preserved_after_except_for_abstention(
    monkeypatch,
):
    response_text = (
        "I could not find enough information in the provided documents "
        "to determine the answer, except for the fact that there are "
        "two quizzes and two assignments [SOURCE 1]."
    )

    monkeypatch.setattr(
        "src.generation.ollama.chat",
        lambda **kwargs: {
            "message": {"content": response_text}
        },
    )

    generator = AnswerGenerator()
    result = generator.generate(
        query="How many quizzes and assignments are conducted?",
        results=[make_result(2)],
    )

    assert "two quizzes and two assignments" in result.answer
    assert "could not find enough information" not in result.answer.lower()
    assert "[test.pdf, Page 2]" in result.answer
    assert len(result.sources) == 1


def test_supported_answer_is_preserved_after_other_than_abstention(
    monkeypatch,
):
    response_text = (
        "I could not find enough information in the provided documents "
        "regarding the slide timing, other than that slides are uploaded "
        "at least five days prior to the lecture [SOURCE 1]."
    )

    monkeypatch.setattr(
        "src.generation.ollama.chat",
        lambda **kwargs: {
            "message": {"content": response_text}
        },
    )

    generator = AnswerGenerator()
    result = generator.generate(
        query="When are slides uploaded before a lecture?",
        results=[make_result(1)],
    )

    assert "at least five days prior" in result.answer
    assert "could not find enough information" not in result.answer.lower()
    assert "[test.pdf, Page 1]" in result.answer
    assert len(result.sources) == 1


def test_pure_abstention_returns_fallback_without_sources(
    monkeypatch,
):
    response_text = (
        "I could not find enough information in the provided documents."
    )

    monkeypatch.setattr(
        "src.generation.ollama.chat",
        lambda **kwargs: {
            "message": {"content": response_text}
        },
    )

    generator = AnswerGenerator()
    result = generator.generate(
        query="What is not documented?",
        results=[make_result(1)],
    )

    assert (
        result.answer
        == "I could not find enough information in the provided documents."
    )
    assert result.sources == []


def test_mixed_abstention_without_source_marker_returns_fallback(
    monkeypatch,
):
    response_text = (
        "I could not find enough information in the provided documents, "
        "except for the fact that there are two quizzes and two assignments."
    )

    monkeypatch.setattr(
        "src.generation.ollama.chat",
        lambda **kwargs: {
            "message": {"content": response_text}
        },
    )

    generator = AnswerGenerator()
    result = generator.generate(
        query="How many quizzes and assignments are conducted?",
        results=[make_result(2)],
    )

    assert (
        result.answer
        == "I could not find enough information in the provided documents."
    )
    assert result.sources == []
