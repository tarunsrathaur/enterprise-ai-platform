from pathlib import Path


def has_expected_citation(
    answer: str,
    sources: list[dict],
    expected_documents: set[str],
    expected_pages: set[int],
) -> bool:
    """Check whether the answer cites an expected document and page."""
    for source in sources:
        source_name = source.get("source", "")
        page = source.get("page")

        if (
            Path(source_name).stem.casefold() in expected_documents
            and page in expected_pages
        ):
            citation = f"[{source_name}, Page {page}]"
            if citation in answer:
                return True

    return False


INSUFFICIENT_INFORMATION_ANSWER = (
    "I could not find enough information in the provided documents."
)



def is_abstention(answer: str) -> bool:
    """Identify explicit insufficient-evidence responses."""
    normalized_answer = answer.strip().casefold()

    if normalized_answer.startswith(
        INSUFFICIENT_INFORMATION_ANSWER.casefold()
    ):
        return True

    return (
        "not explicitly stated in the provided document"
        in normalized_answer
        or "could not find enough information in the provided documents"
        in normalized_answer
    )
