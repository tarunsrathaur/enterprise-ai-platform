import re
import sys
from pathlib import Path

from fastapi.testclient import TestClient
from src.evaluation_checks import has_expected_citation, is_abstention

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation_dataset import load_scenarios
from src.api import app


DATASET_PATH = PROJECT_ROOT / "data" / "evaluation" / "scenarios.json"
client = TestClient(app)

ABSTENTION_MESSAGE = (
    "I could not find enough information in the provided documents."
)

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can",
    "for", "from", "in", "is", "it", "of", "on", "or", "that",
    "the", "their", "this", "to", "what", "which", "with",
    "not", "must", "use", "using",
}


def meaningful_terms(text: str) -> set[str]:
    """Extract normalized terms for a basic answer-content check."""
    return {
        term
        for term in re.findall(r"\b[a-z0-9]+\b", text.casefold())
        if len(term) > 2 and term not in STOP_WORDS
    }


def evaluate_scenario(scenario: dict) -> dict:
    """Evaluate one scenario against the local FastAPI application."""
    response = client.post(
        "/ask",
        json={
            "question": scenario["question"],
            "top_k": 3,
        },
    )

    if response.status_code != 200:
        return {
            "scenario_id": scenario["scenario_id"],
            "passed": False,
            "checks": {
                "http_success": False,
                "answer_content": False,
                "source_evidence": False,
                "abstention": False,
                "citation_correctness": False,
            },
            "details": f"API returned HTTP {response.status_code}: "
            f"{response.text}",
        }

    data = response.json()
    answer = data["answer"]
    sources = data["sources"]

    checks = {
        "http_success": True,
        "answer_content": False,
        "source_evidence": False,
        "abstention": False,
    }
    details = []

    if scenario["answerable"]:
        expected_terms = meaningful_terms(scenario["expected_answer"])
        answer_terms = meaningful_terms(answer)

        coverage = (
            len(expected_terms & answer_terms) / len(expected_terms)
            if expected_terms
            else 0.0
        )

        # A basic lexical check, not a semantic correctness guarantee.


        if scenario["scenario_id"] in ("DIRECT-001", "PARA-001"):
            normalized_answer = answer.casefold()

            mentions_platform = (
                "ms teams" in normalized_answer
                or "teams" in normalized_answer
            )
            mentions_academic_queries = any(
                phrase in normalized_answer
                for phrase in (
                    "academic quer",
                    "academic doubt",
                    "faculty",
                )
            )
            expresses_prohibition = any(
                phrase in normalized_answer
                for phrase in (
                    "not approved",
                    "must not use",
                    "cannot use",
                    "can't use",
                    "cannot be used",
                    "not permitted",
                    "prohibited",
                    "do not use",
                    "not allowed",
                    "cannot contact",
                )
            )

            checks["answer_content"] = (
                mentions_platform
                and mentions_academic_queries
                and expresses_prohibition
            )
        else:
            checks["answer_content"] = coverage >= 0.50


        details.append(f"answer term coverage={coverage:.2f}")

        expected_documents = {
            document_id.casefold()
            for document_id in scenario["source_document_ids"]
        }
        expected_pages = set(scenario["expected_pages"])

        checks["source_evidence"] = any(
            Path(source["source"]).stem.casefold() in expected_documents
            and source["page"] in expected_pages
            for source in sources
        )

        checks["citation_correctness"] = has_expected_citation(
            answer=answer,
            sources=sources,
            expected_documents=expected_documents,
            expected_pages=expected_pages,
        )

        details.append(
            f"returned sources={len(sources)}, "
            f"expected documents={sorted(expected_documents)}, "
            f"expected pages={sorted(expected_pages)}"
        )

        checks["abstention"] = not data["metrics"]["abstained"]
    else:
        checks["answer_content"] = is_abstention(answer)
        checks["source_evidence"] = len(sources) == 0
        checks["abstention"] = (
            data["metrics"]["abstained"] is True
            and is_abstention(answer)
        )


    passed = all(checks.values())

    return {
        "scenario_id": scenario["scenario_id"],
        "passed": passed,
        "checks": checks,
        "details": "; ".join(details),
    }


def main() -> int:
    scenarios = load_scenarios(DATASET_PATH)
    results = []

    print("Enterprise RAG evaluation")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Scenarios: {len(scenarios)}")

    for scenario in scenarios:
        result = evaluate_scenario(scenario)
        results.append(result)

        print("\n" + "-" * 72)
        print(f"Scenario: {result['scenario_id']}")
        print(f"Status: {'PASS' if result['passed'] else 'FAIL'}")

        for check, passed in result["checks"].items():
            print(f"  {check}: {'PASS' if passed else 'FAIL'}")

        if result["details"]:
            print(f"  Details: {result['details']}")

    passed_count = sum(result["passed"] for result in results)

    print("\n" + "=" * 72)
    print(f"Scenarios passed: {passed_count}/{len(results)}")
    print(f"Scenarios failed: {len(results) - passed_count}")

    # A failed evaluation should be visible to scripts and CI.
    return 0 if passed_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
