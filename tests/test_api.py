from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


def test_ask_returns_grounded_answer():
    response = client.post(
        "/ask",
        json={
            "question": "Can LFs help students via MS Teams?",
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "answer" in data
    assert "sources" in data

    assert "MS Teams" in data["answer"]
    assert "[Page 5]" in data["answer"]

    assert len(data["sources"]) >= 1
    assert data["sources"][0]["page"] == 5


def test_ask_abstains_when_information_is_missing():
    response = client.post(
        "/ask",
        json={
            "question": "What is the professor's personal phone number?",
            "top_k": 3,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "could not find enough information" in data["answer"].lower()

    assert data["sources"] == []


def test_ask_rejects_invalid_top_k():
    response = client.post(
        "/ask",
        json={
            "question": "What are the academic query channels?",
            "top_k": 0,
        },
    )

    assert response.status_code == 422