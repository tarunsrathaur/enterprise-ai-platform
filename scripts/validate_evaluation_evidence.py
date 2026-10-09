
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation_dataset import load_scenarios
from src.ingestion import DocumentLoader


DATASET_PATH = PROJECT_ROOT / "data" / "evaluation" / "scenarios.json"
CORPUS_PATH = PROJECT_ROOT / "data" / "raw"


def normalize(text: str) -> str:
    """Normalize extracted text, including words split across PDF lines."""
    text = re.sub(r"-\s*\n\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.casefold().strip()


def meaningful_terms(text: str) -> set[str]:
    """Return distinctive terms for a basic lexical evidence check."""
    stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "by", "can",
        "for", "from", "in", "is", "it", "of", "on", "or", "that",
        "the", "their", "this", "to", "what", "which", "with",
    }
    return {
        word
        for word in re.findall(r"\b[a-z0-9]+\b", normalize(text))
        if len(word) > 2 and word not in stop_words
    }



def validate_scenarios(scenarios, documents):
    """Return validation errors and warnings for evaluation scenarios."""
    errors = []
    warnings = []

    for scenario in scenarios:
        scenario_id = scenario["scenario_id"]

        if scenario["answerable"]:
            for document_id in scenario["source_document_ids"]:
                document = documents.get(document_id)

                if document is None:
                    errors.append(
                        f"{scenario_id}: source document "
                        f"'{document_id}' not found."
                    )
                    continue

                for page_number in scenario["expected_pages"]:
                    page = next(
                        (
                            page for page in document.pages
                            if page.page_number == page_number
                        ),
                        None,
                    )

                    if page is None:
                        errors.append(
                            f"{scenario_id}: page {page_number} does not "
                            f"exist in document '{document_id}'."
                        )
                        continue

                    expected_terms = meaningful_terms(
                        scenario["expected_answer"]
                    )
                    page_terms = meaningful_terms(page.content)
                    overlap = expected_terms & page_terms

                    if (
                        expected_terms
                        and len(overlap) / len(expected_terms) < 0.5
                    ):
                        warnings.append(
                            f"{scenario_id}: weak lexical overlap on "
                            f"{document_id}, page {page_number} "
                            f"({len(overlap)}/{len(expected_terms)} terms)."
                        )

        else:
            question_terms = meaningful_terms(scenario["question"])
            matched_documents = []

            for document_id, document in documents.items():
                document_text = normalize(document.content)
                document_terms = meaningful_terms(document_text)
                overlap = question_terms & document_terms

                if (
                    question_terms
                    and len(overlap) / len(question_terms) >= 0.5
                ):
                    matched_documents.append(document_id)

            if matched_documents:
                warnings.append(
                    f"{scenario_id}: question terms overlap with documents "
                    f"{matched_documents}; manually verify unanswerability."
                )

    return errors, warnings


def main() -> int:
    scenarios = load_scenarios(DATASET_PATH)
    documents = {}

    for file_path in sorted(CORPUS_PATH.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in {".txt", ".pdf"}:
            document = DocumentLoader().load(file_path)
            documents[document.document_id] = document

    errors, warnings = validate_scenarios(scenarios, documents)

    for warning in warnings:
        print(f"WARNING: {warning}")

    for error in errors:
        print(f"ERROR: {error}")

    print(f"\nScenarios checked: {len(scenarios)}")
    print(f"Documents loaded: {len(documents)}")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
