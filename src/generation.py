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
- Use only the supplied source content as factual evidence.
- Answer the exact question asked, not a related or broader question.
- If a source directly answers the question, prioritize that direct evidence.
- If the source explicitly prohibits an action, state the prohibition clearly.
  Do not begin by saying the action is allowed when the source says it is not.
- Do not combine facts about alternative permitted channels with a prohibition
  in a way that creates a contradictory answer.
- Never invent or infer numbers, durations, dates, thresholds, penalties,
  approval status, or mandatory requirements.
- A related policy statement does not establish a detail it does not specify.
  For example, a requirement to review an output does not specify how long
  that review must take.
- If the requested detail is absent, respond exactly:
  "I could not find enough information in the provided documents."
- Preserve important names, numbers, dates, and requirements exactly as
  supported by the source.
- After each factual claim, include the source marker(s) that support it,
  using the exact format [SOURCE 1], [SOURCE 2], and so on.
- Only cite sources that directly support the claim.
- Do not invent source markers.

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

        # Remove a trailing abstention explanation when a factual answer
        # has already been provided.
        abstention_phrases = [
            "I could not find enough information",
            "I cannot find enough information",
            "the provided documents do not contain",
        ]

        for phrase in abstention_phrases:
            position = answer.casefold().find(phrase.casefold())
            if position > 0:
                answer = answer[:position].strip()
                break

        # Some model responses begin with an abstention but then state
        # a fact that the supplied evidence actually supports.
        if is_abstention(answer):
            contrast_patterns = [
                r"\bexcept for (?:the fact that )?",
                r"\bother than (?:the fact that )?",
                r"\bapart from (?:the fact that )?",
            ]

            lowered_answer = answer.casefold()
            abstention_positions = [
                lowered_answer.find(phrase.casefold())
                for phrase in abstention_phrases
                if lowered_answer.find(phrase.casefold()) >= 0
            ]

            abstention_position = (
                min(abstention_positions)
                if abstention_positions
                else -1
            )

            contrast_matches = []
            for pattern in contrast_patterns:
                match = re.search(pattern, answer, re.IGNORECASE)
                if (
                    match is not None
                    and abstention_position >= 0
                    and match.start() > abstention_position
                    and match.end() < len(answer)
                ):
                    contrast_matches.append(match)

            if contrast_matches:
                contrast = min(
                    contrast_matches,
                    key=lambda match: match.start(),
                )
                cleaned_answer = answer[contrast.end():].strip(
                    " \t\r\n,;:-"
                )

                # Keep the extracted claim only if it has a source marker.
                if self.SOURCE_MARKER_PATTERN.search(cleaned_answer):
                    answer = cleaned_answer

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