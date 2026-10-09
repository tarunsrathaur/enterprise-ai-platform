
import json

import pytest

from src.evaluation_dataset import load_scenarios


def write_dataset(tmp_path, scenarios):
    dataset = {
        "dataset_version": "0.1.0",
        "scenarios": scenarios,
    }
    path = tmp_path / "scenarios.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")
    return path


def valid_scenario():
    return {
        "scenario_id": "DIRECT-001",
        "category": "direct_retrieval",
        "question": "What is the policy?",
        "answerable": True,
        "expected_answer": "The documented policy.",
        "source_document_ids": ["policy"],
        "expected_pages": [1],
    }


def test_loads_valid_scenario_dataset(tmp_path):
    path = write_dataset(tmp_path, [valid_scenario()])

    scenarios = load_scenarios(path)

    assert len(scenarios) == 1
    assert scenarios[0]["scenario_id"] == "DIRECT-001"


def test_rejects_duplicate_scenario_ids(tmp_path):
    scenario = valid_scenario()
    duplicate = {**scenario, "question": "A different question?"}
    path = write_dataset(tmp_path, [scenario, duplicate])

    with pytest.raises(ValueError, match="Duplicate scenario_id"):
        load_scenarios(path)


def test_rejects_answerable_scenario_without_answer(tmp_path):
    scenario = {**valid_scenario(), "expected_answer": ""}
    path = write_dataset(tmp_path, [scenario])

    with pytest.raises(ValueError, match="need an expected answer"):
        load_scenarios(path)


def test_rejects_unanswerable_scenario_with_answer(tmp_path):
    scenario = {
        **valid_scenario(),
        "scenario_id": "UNANSWERABLE-001",
        "category": "unanswerable",
        "answerable": False,
        "expected_answer": "An invented answer",
        "source_document_ids": [],
        "expected_pages": [],
    }
    path = write_dataset(tmp_path, [scenario])

    with pytest.raises(ValueError, match="must have expected_answer set to null"):
        load_scenarios(path)


def test_rejects_missing_dataset_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_scenarios(tmp_path / "missing.json")
