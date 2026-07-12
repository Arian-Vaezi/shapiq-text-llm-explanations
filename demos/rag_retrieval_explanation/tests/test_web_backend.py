"""Tests for the demo's framework-independent web backend."""

from __future__ import annotations

from demos.rag_retrieval_explanation.backend.schemas import RunResponse
from demos.rag_retrieval_explanation.backend.services import (
    default_settings,
    run_sample,
)


def test_sample_run_returns_dashboard_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        "demos.rag_retrieval_explanation.backend.services.generate_answer_from_chunks",
        lambda **_kwargs: "The model says Curie's awards were Physics and Chemistry.",
    )
    settings = default_settings()
    settings.budget = 32

    result = run_sample("Marie Curie Nobel categories", settings)

    assert result["question"].startswith("Which two Nobel")
    assert result["answer_target_type"] == "generated_answer"
    assert result["retrieval_method"] == "Preselected controlled chunks"
    RunResponse.model_validate(result)
    assert [target["target_type"] for target in result["targets"]] == [
        "reference_answer",
        "generated_answer",
    ]
    assert result["targets"][0]["answer"].startswith("Marie Curie won")
    assert result["targets"][1]["answer"].startswith("The model says")
    assert result["targets"][0]["results"] is not result["targets"][1]["results"]
    assert len(result["chunks"]) == 4
    assert result["results"][0]["value_name"] == "Lexical grounding"
    assert len(result["results"][0]["attributions"]) == 4
    assert len(result["results"][0]["deletion_curve"]) == 5


def test_sample_run_rejects_unknown_scenario() -> None:
    settings = default_settings()

    try:
        run_sample("missing", settings)
    except ValueError as error:
        assert "Unknown sample scenario" in str(error)
    else:
        raise AssertionError("Expected ValueError")
