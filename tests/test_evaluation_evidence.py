
from types import SimpleNamespace

from scripts.validate_evaluation_evidence import validate_scenarios


def make_document(document_id, pages):
    page_objects = [
        SimpleNamespace(page_number=number, content=text)
        for number, text in pages.items()
    ]
    content = "\n".join(page.content for page in page_objects)

    return SimpleNamespace(
        document_id=document_id,
        pages=page_objects,
        content=content,
    )


def make_scenario(**overrides):
    scenario = {
        "scenario_id": "DIRECT-001",
        "category": "direct_retrieval",
        "question": "Which platform is not approved?",
        "answerable": True,
        "expected_answer": "MS Teams is not an approved platform for academic queries.",
        "source_document_ids": ["FAQ"],
        "expected_pages": [5],
    }
    scenario.update(overrides)
    return scenario


def test_accepts_answer_with_matching_page_evidence():
    scenario = make_scenario()
    documents = {
        "FAQ": make_document(
            "FAQ",
            {5: "MS Teams is not an approved platform for academic queries."},
        )
    }

    errors, warnings = validate_scenarios([scenario], documents)

    assert errors == []
    assert warnings == []


def test_rejects_missing_source_document():
    scenario = make_scenario()

    errors, _ = validate_scenarios([scenario], {})

    assert len(errors) == 1
    assert "source document 'FAQ' not found" in errors[0]


def test_rejects_missing_page():
    scenario = make_scenario()
    documents = {
        "FAQ": make_document("FAQ", {4: "Some other content."})
    }

    errors, _ = validate_scenarios([scenario], documents)

    assert len(errors) == 1
    assert "page 5 does not exist" in errors[0]


def test_warns_when_expected_answer_has_weak_lexical_support():
    scenario = make_scenario(
        expected_answer="The quarterly reimbursement limit is 9000 credits."
    )
    documents = {
        "FAQ": make_document("FAQ", {5: "MS Teams is not approved for queries."})
    }

    errors, warnings = validate_scenarios([scenario], documents)

    assert errors == []
    assert len(warnings) == 1
    assert "weak lexical overlap" in warnings[0]


def test_warns_when_unanswerable_question_overlaps_corpus():
    scenario = make_scenario(
        scenario_id="UNANSWERABLE-001",
        category="unanswerable",
        question="What penalty applies to academic queries on MS Teams?",
        answerable=False,
        expected_answer=None,
        source_document_ids=[],
        expected_pages=[],
    )
    documents = {
        "FAQ": make_document(
            "FAQ",
            {5: "MS Teams is not an approved platform for academic queries."},
        )
    }

    errors, warnings = validate_scenarios([scenario], documents)

    assert errors == []
    assert len(warnings) == 1
    assert "manually verify unanswerability" in warnings[0]
