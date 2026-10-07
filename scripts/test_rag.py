from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evidence import EvidenceSelector
from src.generation import AnswerGenerator
from src.retrieval import RetrievalPipeline


EVALUATION_CASES = [
    {
    "query": "Can LFs help students via MS Teams?",
    "expected_page": 5,
    "expected_keyword": "not an approved",
    "answerable": True,
},
{
    "query": "How can students seek clarification or resolve doubts?",
    "expected_page": 4,
    "expected_keyword": "discussion forum",
    "answerable": True,
},
{
    "query": "When should experiential learning exercises be completed?",
    "expected_page": 5,
    "expected_keyword": "lecture",
    "answerable": True,
},
{
    "query": "What programming language should be used for experiential learning?",
    "expected_page": 5,
    "expected_keyword": "python",
    "answerable": True,
},
{
    "query": "Where can students find faculty notes?",
    "expected_page": 1,
    "expected_keyword": "MS Teams",
    "answerable": True,
},
{
    "query": "What is the professor's personal phone number?",
    "expected_page": None,
    "expected_keyword": None,
    "answerable": False,
},
]


def main() -> None:
    pipeline = RetrievalPipeline(
        chunk_size=500,
        chunk_overlap=75,
    )

    count = pipeline.index_document("data/raw/FAQ.pdf")

    print(f"Indexed chunks: {count}")
    print()

    evidence_selector = EvidenceSelector(
        relative_score_threshold=0.65,
        max_evidence=3,
    )

    generator = AnswerGenerator(
        model_name="llama3.2:3b"
    )

    passed = 0

    for index, case in enumerate(EVALUATION_CASES, start=1):

        query = case["query"]

        print("=" * 70)
        print(f"TEST {index}: {query}")
        print("=" * 70)

        retrieved = pipeline.search(
            query,
            top_k=3,
        )

        if not retrieved:
            print("No retrieval results.")
            print("RESULT: FAIL")
            continue

        print("Retrieved candidates:")

        for result in retrieved:
            print(
                f"  Page {result.chunk.page_number} "
                f"| Score: {result.score:.4f}"
            )

        evidence = evidence_selector.select(
            retrieved
        )

        print()
        print("Selected evidence:")

        for result in evidence:
            print(
                f"  Page {result.chunk.page_number} "
                f"| Score: {result.score:.4f}"
            )

        generated = generator.generate(
            query=query,
            results=evidence,
        )

        print()
        print("ANSWER:")
        print(generated.answer)
        print()

        if case["answerable"]:

            answer_ok = (
            case["expected_keyword"].lower()
                in generated.answer.lower()
            )

            retrieval_ok = (
                retrieved[0].chunk.page_number
                == case["expected_page"]
            )

            evidence_ok = any(
                result.chunk.page_number
                == case["expected_page"]
                for result in evidence
            )

            citation_ok = (
                f"[Page {case['expected_page']}]"
                in generated.answer
            )

            if retrieval_ok and evidence_ok and answer_ok and citation_ok:
                print("RESULT: PASS")
                passed += 1
            else:
                print("RESULT: FAIL")
                print(f"Answer keyword found: {answer_ok}")

        else:

            refusal_phrases = [
                "could not find",
                "not available",
                "not provided",
                "do not contain",
                "does not contain",
                "cannot find",
                "no information",
            ]

            answer_lower = generated.answer.lower()

            refused = any(
                phrase in answer_lower
                for phrase in refusal_phrases
            )

            # An unsupported answer must not attach source citations.
            has_citation = "[Page " in generated.answer

            if refused and not has_citation:
                print("RESULT: PASS")
                passed += 1
            else:
                print("RESULT: FAIL")

        print()

    total = len(EVALUATION_CASES)

    print("=" * 70)
    print(f"RAG Evaluation: {passed}/{total} passed")
    print(f"Pass rate: {passed / total:.1%}")
    print("=" * 70)


if __name__ == "__main__":
    main()