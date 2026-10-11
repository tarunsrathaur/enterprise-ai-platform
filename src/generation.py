from dataclasses import dataclass
import re

import ollama

from src.config import RAG_MODEL
from src.evaluation_checks import is_abstention
from src.vector_store import SearchResult


@dataclass
class GeneratedAnswer:
    """Represents an answer generated from selected evidence."""

    answer: str
    sources: list[SearchResult]


class AnswerGenerator:
    """Generates grounded answers using a local Ollama model."""

    SOURCE_MARKER_PATTERN = re.compile(r"\[SOURCE\s+(\d+)\]", re.IGNORECASE)

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or RAG_MODEL

    def generate(
        self,
        query: str,
        results: list[SearchResult],
    ) -> GeneratedAnswer:
        fallback = (
            "I could not find enough information "
            "in the provided documents."
        )

        if not results:
            return GeneratedAnswer(answer=fallback, sources=[])

        context_parts = []

        for index, result in enumerate(results, start=1):
            context_parts.append(
                f"""[SOURCE {index}]
PAGE: {result.chunk.page_number}
CONTENT:
{result.chunk.text}
"""
            )

        context = "\n\n".join(context_parts)

        prompt = f"""
Answer the question using only the supplied source content.

Rules:
- If the source contains the answer, state it directly and concisely.
- Preserve important names, numbers, dates, and requirements.
- Do not refuse when the source directly answers the question.
- After each factual claim, include the source marker(s) that support it,
  using the exact format [SOURCE 1], [SOURCE 2], and so on.
- Only cite sources that directly support the claim.
- Do not invent source markers.
- If the requested information is absent, respond exactly:
  "{fallback}"

QUESTION:
{query}

SOURCE CONTENT:
{context}

ANSWER:
""".strip()

        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
        )

        answer = response["message"]["content"].strip()

        # Remove any trailing explanation that contradicts an already
        # provided, source-supported answer.
        abstention_phrases = [
            "I could not find enough information",
            "I cannot find enough information",
            "the provided documents do not contain",
        ]

        for phrase in abstention_phrases:
            position = answer.lower().find(phrase.lower())
            if position > 0:
                answer = answer[:position].strip()
                break

        if is_abstention(answer):
            return GeneratedAnswer(answer=fallback, sources=[])

        source_numbers = []
        for match in self.SOURCE_MARKER_PATTERN.finditer(answer):
            number = int(match.group(1))
            if 1 <= number <= len(results) and number not in source_numbers:
                source_numbers.append(number)

        if not source_numbers:
            return GeneratedAnswer(answer=fallback, sources=[])

        selected_sources = [results[number - 1] for number in source_numbers]

        clean_answer = self.SOURCE_MARKER_PATTERN.sub("", answer)
        clean_answer = re.sub(r"[ \t]+", " ", clean_answer)
        clean_answer = re.sub(r" *\n *", "\n", clean_answer).strip()

        citations = []
        seen_citations = set()

        for result in selected_sources:
            source_name = (
                result.chunk.source.replace("\\", "/").rsplit("/", 1)[-1]
            )
            citation = (source_name, result.chunk.page_number)

            if citation not in seen_citations:
                seen_citations.add(citation)
                citations.append(
                    f"[{source_name}, Page {result.chunk.page_number}]"
                )

        answer_with_citations = (
            f"{clean_answer} {' '.join(citations)}"
        ).strip()

        return GeneratedAnswer(
            answer=answer_with_citations,
            sources=selected_sources,
        )