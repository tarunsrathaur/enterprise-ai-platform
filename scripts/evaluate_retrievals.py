from pathlib import Path
import sys

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval import RetrievalPipeline


EVALUATION_CASES = [
    {
        "query": "Can LFs help students via MS Teams?",
        "expected_page": 5,
    },
    {
        "query": "How can students seek clarification or resolve doubts?",
        "expected_page": 4,
    },
    {
        "query": "When should experiential learning exercises be completed?",
        "expected_page": 5,
    },
    {
        "query": "What programming language should be used for experiential learning?",
        "expected_page": 5,
    },
    {
        "query": "Where can students find faculty notes?",
        "expected_page": 1,
    },
]


def main() -> None:
    pipeline = RetrievalPipeline(
        chunk_size=500,
        chunk_overlap=75,
    )

    count = pipeline.index_document("data/raw/FAQ.pdf")
    print(f"Indexed chunks: {count}\n")

    hits = 0

    for case in EVALUATION_CASES:
        results = pipeline.search(case["query"], top_k=3)

        top_page = results[0].chunk.page_number if results else None
        success = top_page == case["expected_page"]

        if success:
            hits += 1

        status = "PASS" if success else "FAIL"

        print(f"[{status}] {case['query']}")
        print(f"      Expected page : {case['expected_page']}")
        print(f"      Retrieved page: {top_page}")

        if results:
            print(f"      Top score     : {results[0].score:.4f}")

        print()

    total = len(EVALUATION_CASES)
    hit_rate = hits / total

    print("=" * 50)
    print(f"Retrieval Hit@1: {hits}/{total} ({hit_rate:.1%})")
    print("=" * 50)


if __name__ == "__main__":
    main()