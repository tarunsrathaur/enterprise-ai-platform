from src.embeddings import EmbeddingService


def test_embedding_dimension():
    service = EmbeddingService()

    embedding = service.embed_text(
        "Enterprise AI applications require security controls."
    )

    assert len(embedding) == service.dimension


def test_similar_sentences_have_high_similarity():
    service = EmbeddingService()

    embeddings = service.embed_texts(
        [
            "AI applications require strong security controls.",
            "Artificial intelligence systems need robust security measures.",
        ]
    )

    similarity = sum(
        a * b for a, b in zip(embeddings[0], embeddings[1])
    )

    assert similarity > 0.5


def test_unrelated_sentences_have_lower_similarity():
    service = EmbeddingService()

    embeddings = service.embed_texts(
        [
            "AI applications require strong security controls.",
            "The restaurant serves spicy vegetarian food.",
        ]
    )

    similarity = sum(
        a * b for a, b in zip(embeddings[0], embeddings[1])
    )

    assert similarity < 0.5