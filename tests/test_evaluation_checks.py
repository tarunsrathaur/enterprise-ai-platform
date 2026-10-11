from src.evaluation_checks import has_expected_citation
from src.evaluation_checks import is_abstention


EXPECTED_DOCUMENTS = {"faq"}
EXPECTED_PAGES = {4, 5}

SOURCES = [
    {"source": "FAQ.pdf", "page": 4},
    {"source": "FAQ.pdf", "page": 5},
]


def test_accepts_expected_document_and_page_citation():
    answer = (
        "Students can use the discussion forum. "
        "[FAQ.pdf, Page 4]"
    )

    assert has_expected_citation(
        answer, SOURCES, EXPECTED_DOCUMENTS, EXPECTED_PAGES
    )


def test_rejects_missing_citation():
    answer = "Students can use the discussion forum."

    assert not has_expected_citation(
        answer, SOURCES, EXPECTED_DOCUMENTS, EXPECTED_PAGES
    )


def test_rejects_wrong_page():
    answer = "Students can use the discussion forum. [FAQ.pdf, Page 2]"

    assert not has_expected_citation(
        answer, SOURCES, EXPECTED_DOCUMENTS, EXPECTED_PAGES
    )


def test_rejects_wrong_document():
    answer = "Students can use the discussion forum. [Other.pdf, Page 4]"
    sources = [{"source": "Other.pdf", "page": 4}]

    assert not has_expected_citation(
        answer, sources, EXPECTED_DOCUMENTS, EXPECTED_PAGES
    )


def test_rejects_empty_sources():
    answer = "Students can use the discussion forum. [FAQ.pdf, Page 4]"

    assert not has_expected_citation(
        answer, [], EXPECTED_DOCUMENTS, EXPECTED_PAGES
    )


def test_accepts_matching_source_when_other_sources_are_irrelevant():
    answer = "Clarification is available. [FAQ.pdf, Page 5]"
    sources = [
        {"source": "Other.pdf", "page": 5},
        {"source": "FAQ.pdf", "page": 5},
    ]

    assert has_expected_citation(
        answer, sources, EXPECTED_DOCUMENTS, EXPECTED_PAGES
    )

def test_recognizes_standard_insufficient_information_response():
    answer = (
        "I could not find enough information "
        "in the provided documents."
    )

    assert is_abstention(answer)


def test_recognizes_explicitly_undocumented_detail():
    answer = (
        "Disciplinary action is not explicitly stated "
        "in the provided document. [sample-policy.txt, Page 1]"
    )

    assert is_abstention(answer)


def test_does_not_classify_a_normal_answer_as_abstention():
    answer = (
        "All production AI systems must maintain appropriate "
        "logging and monitoring. [sample-policy.txt, Page 1]"
    )

    assert not is_abstention(answer)


def test_recognizes_standard_response_with_surrounding_whitespace():
    answer = (
        "  I could not find enough information "
        "in the provided documents.  "
    )

    assert is_abstention(answer)


def test_recognizes_specific_detail_missing_abstention():
    answer = (
        "I could not find enough information in the provided documents "
        "to determine the specific disciplinary action for entering "
        "confidential customer information into a public AI service."
    )

    assert is_abstention(answer)
