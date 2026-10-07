from dataclasses import dataclass

from src.vector_store import SearchResult


@dataclass
class EvidenceSelector:
    """
    Selects retrieved chunks using both absolute and relative
    relevance thresholds.
    """

    minimum_score: float = 0.35
    relative_score_threshold: float = 0.65
    max_evidence: int = 3

    def select(
        self,
        results: list[SearchResult],
    ) -> list[SearchResult]:

        if not results:
            return []

        top_score = results[0].score

        # Reject the entire retrieval result when the strongest
        # candidate is below the minimum confidence threshold.
        if top_score < self.minimum_score:
            return []

        selected = []

        for result in results:

            relative_score = result.score / top_score

            if (
                result.score >= self.minimum_score
                and relative_score >= self.relative_score_threshold
            ):
                selected.append(result)

            if len(selected) >= self.max_evidence:
                break

        return selected