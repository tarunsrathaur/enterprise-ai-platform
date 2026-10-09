
import json
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "scenario_id",
    "category",
    "question",
    "answerable",
    "expected_answer",
    "source_document_ids",
    "expected_pages",
}


def load_scenarios(file_path: str | Path) -> list[dict[str, Any]]:
    """Load and validate an evaluation scenario dataset."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, dict):
        raise ValueError("Dataset root must be a JSON object.")

    if not isinstance(dataset.get("dataset_version"), str):
        raise ValueError("dataset_version must be a string.")

    scenarios = dataset.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError("scenarios must be a list.")

    seen_ids: set[str] = set()

    for index, scenario in enumerate(scenarios):
        location = f"Scenario at index {index}"

        if not isinstance(scenario, dict):
            raise ValueError(f"{location} must be a JSON object.")

        missing_fields = REQUIRED_FIELDS - scenario.keys()
        if missing_fields:
            raise ValueError(
                f"{location} is missing fields: {sorted(missing_fields)}"
            )

        scenario_id = scenario["scenario_id"]
        if not isinstance(scenario_id, str) or not scenario_id.strip():
            raise ValueError(f"{location} has an invalid scenario_id.")

        if scenario_id in seen_ids:
            raise ValueError(f"Duplicate scenario_id: {scenario_id}")
        seen_ids.add(scenario_id)

        if not isinstance(scenario["category"], str) or not scenario["category"].strip():
            raise ValueError(f"{scenario_id}: category must be a non-empty string.")

        if not isinstance(scenario["question"], str) or not scenario["question"].strip():
            raise ValueError(f"{scenario_id}: question must be a non-empty string.")

        if not isinstance(scenario["answerable"], bool):
            raise ValueError(f"{scenario_id}: answerable must be a boolean.")

        expected_answer = scenario["expected_answer"]
        if scenario["answerable"]:
            if not isinstance(expected_answer, str) or not expected_answer.strip():
                raise ValueError(
                    f"{scenario_id}: answerable scenarios need an expected answer."
                )
            if not scenario["source_document_ids"]:
                raise ValueError(
                    f"{scenario_id}: answerable scenarios need source document IDs."
                )
            if not scenario["expected_pages"]:
                raise ValueError(
                    f"{scenario_id}: answerable scenarios need expected pages."
                )
        else:
            if expected_answer is not None:
                raise ValueError(
                    f"{scenario_id}: unanswerable scenarios must have "
                    "expected_answer set to null."
                )
            if scenario["source_document_ids"] or scenario["expected_pages"]:
                raise ValueError(
                    f"{scenario_id}: unanswerable scenarios must not declare "
                    "supporting documents or pages."
                )

        if not isinstance(scenario["source_document_ids"], list):
            raise ValueError(
                f"{scenario_id}: source_document_ids must be a list."
            )

        if not isinstance(scenario["expected_pages"], list):
            raise ValueError(f"{scenario_id}: expected_pages must be a list.")

        if any(not isinstance(page, int) or page <= 0 for page in scenario["expected_pages"]):
            raise ValueError(
                f"{scenario_id}: expected_pages must contain positive integers."
            )

    return scenarios