from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evidence import EvidenceSelector
from src.generation import AnswerGenerator
from src.retrieval import RetrievalPipeline


DOCUMENT = "data/raw/FAQ.pdf"

MODELS = [
    "llama3.2:3b",
    "gemma2:9b",
]

TEST_CASES = [
    {
        "query": "Can LFs help students via MS Teams?",
        "expected_keyword": "not an approved",
        "answerable": True,
    },
    {
        "query": "How can students seek clarification or resolve doubts?",
        "expected_keyword": "discussion forum",
        "answerable": True,
    },
    {
        "query": "When should experiential learning exercises be completed?",
        "expected_keyword": "lecture",
        "answerable": True,
    },
    {
        "query": "What programming language should be used for experiential learning?",
        "expected_keyword": "python",
        "answerable": True,
    },
    {
        "query": "Where can students find faculty notes?",
        "expected_keyword": "MS Teams",
        "answerable": True,
    },
    {
        "query": "What is the professor's personal phone number?",
        "expected_keyword": None,
        "answerable": False,
    },
]


def evaluate_model(model_name: str):
    print("\n" + "=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)

    pipeline = RetrievalPipeline(
        chunk_size=500,
        chunk_overlap=75,
    )

    evidence_selector = EvidenceSelector(
        minimum_score=0.35,
        relative_score_threshold=0.65,
        max_evidence=3,
    )

    generator = AnswerGenerator(model_name=model_name)

    start_index = time.perf_counter()

    indexed_count = pipeline.index_document(DOCUMENT)

    index_time_ms = (
        time.perf_counter() - start_index
    ) * 1000

    print(
        f"Indexed chunks: {indexed_count} | "
        f"Indexing time: {index_time_ms:.2f} ms"
    )

    results_summary = []

    for number, case in enumerate(TEST_CASES, start=1):

        print("\n" + "-" * 70)
        print(f"TEST {number}: {case['query']}")
        print("-" * 70)

        request_start = time.perf_counter()

        retrieval_start = time.perf_counter()

        results = pipeline.search(
            case["query"],
            top_k=3,
        )

        retrieval_time_ms = (
            time.perf_counter() - retrieval_start
        ) * 1000

        evidence = evidence_selector.select(results)

        generation_start = time.perf_counter()

        generated = generator.generate(
            query=case["query"],
            results=evidence,
        )

        generation_time_ms = (
            time.perf_counter() - generation_start
        ) * 1000

        total_time_ms = (
            time.perf_counter() - request_start
        ) * 1000

        answer = generated.answer

        if case["answerable"]:
            keyword_pass = (
                case["expected_keyword"].lower()
                in answer.lower()
            )

            citation_pass = "[Page " in answer

            pass_test = (
                len(evidence) > 0
                and keyword_pass
                and citation_pass
            )

        else:
            abstention_pass = (
                len(evidence) == 0
                and "could not find enough information"
                in answer.lower()
                and "[Page " not in answer
            )

            pass_test = abstention_pass

        print(f"Retrieval latency : {retrieval_time_ms:.2f} ms")
        print(f"Generation latency: {generation_time_ms:.2f} ms")
        print(f"Total latency     : {total_time_ms:.2f} ms")

        if results:
            print(
                f"Top retrieval score: "
                f"{results[0].score:.4f}"
            )

        print(
            f"Evidence selected : {len(evidence)}"
        )

        print(f"Answer:\n{answer}")

        print(
            f"RESULT: {'PASS' if pass_test else 'FAIL'}"
        )

        results_summary.append(
            {
                "passed": pass_test,
                "retrieval_ms": retrieval_time_ms,
                "generation_ms": generation_time_ms,
                "total_ms": total_time_ms,
            }
        )

    passed = sum(
        result["passed"]
        for result in results_summary
    )

    answerable_cases = [
        result
        for result, case in zip(
            results_summary,
            TEST_CASES,
        )
        if case["answerable"]
    ]

    abstention_cases = [
        result
        for result, case in zip(
            results_summary,
            TEST_CASES,
        )
        if not case["answerable"]
    ]

    avg_retrieval = sum(
        result["retrieval_ms"]
        for result in results_summary
    ) / len(results_summary)

    avg_generation = sum(
        result["generation_ms"]
        for result in results_summary
    ) / len(results_summary)

    avg_total = sum(
        result["total_ms"]
        for result in results_summary
    ) / len(results_summary)

    print("\n" + "=" * 70)
    print(f"SUMMARY: {model_name}")
    print("=" * 70)

    print(
        f"Overall pass rate     : "
        f"{passed}/{len(TEST_CASES)} "
        f"({passed / len(TEST_CASES):.1%})"
    )

    print(
        f"Average retrieval     : "
        f"{avg_retrieval:.2f} ms"
    )

    print(
        f"Average generation    : "
        f"{avg_generation:.2f} ms"
    )

    print(
        f"Average total         : "
        f"{avg_total:.2f} ms"
    )

    if answerable_cases:
        answerable_passed = sum(
            result["passed"]
            for result in answerable_cases
        )

        print(
            f"Answerable accuracy  : "
            f"{answerable_passed}/{len(answerable_cases)} "
            f"({answerable_passed / len(answerable_cases):.1%})"
        )

    if abstention_cases:
        abstention_passed = sum(
            result["passed"]
            for result in abstention_cases
        )

        print(
            f"Abstention accuracy  : "
            f"{abstention_passed}/{len(abstention_cases)} "
            f"({abstention_passed / len(abstention_cases):.1%})"
        )

    return {
        "model": model_name,
        "passed": passed,
        "total": len(TEST_CASES),
        "avg_retrieval_ms": avg_retrieval,
        "avg_generation_ms": avg_generation,
        "avg_total_ms": avg_total,
    }


def main():
    all_results = []

    for model_name in MODELS:
        all_results.append(
            evaluate_model(model_name)
        )

    print("\n\n")
    print("=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    print(
        f"{'Model':<18}"
        f"{'Pass Rate':<14}"
        f"{'Retrieval ms':<16}"
        f"{'Generation ms':<18}"
        f"{'Total ms':<14}"
    )

    print("-" * 80)

    for result in all_results:
        print(
            f"{result['model']:<18}"
            f"{result['passed']}/{result['total']:<12}"
            f"{result['avg_retrieval_ms']:<16.2f}"
            f"{result['avg_generation_ms']:<18.2f}"
            f"{result['avg_total_ms']:<14.2f}"
        )


if __name__ == "__main__":
    main()