from dataclasses import dataclass

import ollama

from src.config import RAG_MODEL
from src.vector_store import SearchResult


@dataclass
class GeneratedAnswer:
    """Represents an answer generated from selected evidence."""
    answer: str
    sources: list[SearchResult]


class AnswerGenerator:
    """Generates grounded answers using a local Ollama model."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or RAG_MODEL

    def generate(
        self,
        query: str,
        results: list[SearchResult],
    ) -> GeneratedAnswer:

        if not results:
            return GeneratedAnswer(
                answer=(
                    "I could not find enough information "
                    "in the provided documents."
                ),
                sources=[],
            )

        context_parts = []

        for index, result in enumerate(results, start=1):
            context_parts.append(
                f"""SOURCE {index}
PAGE: {result.chunk.page_number}
CONTENT:
{result.chunk.text}
"""
            )

        context = "\n\n".join(context_parts)

        prompt = f"""
You are answering a question using an enterprise document.

IMPORTANT:
The answer MUST be based only on the provided source content.

If the source contains the answer, answer the question directly.
Do NOT say that information is missing when the answer is explicitly
present in the source.

Only say:
"I could not find enough information in the provided documents."
when the provided source genuinely does not contain enough information
to answer the question.

Do not use outside knowledge.
Do not invent facts.
Do not create citations.
Do not mention source numbers or page numbers.

QUESTION:
{query}

SOURCE CONTENT:
{context}

TASK:
Read the source content carefully.
Identify the specific information that answers the question.
Then provide a concise direct answer.

ANSWER:
""".strip()

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            options={
                "temperature": 0,
            },
        )

        answer = response["message"]["content"].strip()

        source_pages = sorted(
            {
                result.chunk.page_number
                for result in results
            }
        )

        citations = " ".join(
            f"[Page {page}]"
            for page in source_pages
        )

        answer_with_citations = (
            f"{answer} {citations}"
        ).strip()

        return GeneratedAnswer(
            answer=answer_with_citations,
            sources=results,
        )