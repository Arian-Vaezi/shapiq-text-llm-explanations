"""Tests for the demo's QASPER ingestion and external evaluation."""

from __future__ import annotations

from demos.rag_retrieval_explanation.evals.experiments.qasper import (
    _map_evidence,
    _paper_passages,
    qasper_candidates,
    select_qasper_cases,
)


def _row() -> dict[str, object]:
    return {
        "id": "paper-1",
        "title": "A Test Paper",
        "abstract": ["This paper studies retrieval explanations."],
        "full_text": {
            "section_name": ["Methods", "Results"],
            "paragraphs": [
                ["We use a dense retriever to select relevant paragraphs."],
                ["The dense method retrieves the annotated evidence first."],
            ],
        },
        "qas": {
            "question": ["What does the dense method retrieve first?"],
            "question_id": ["question-1"],
            "answers": [
                {
                    "answer": [
                        {
                            "unanswerable": False,
                            "extractive_spans": ["the annotated evidence"],
                            "yes_no": None,
                            "free_form_answer": "",
                            "evidence": [
                                "The dense method retrieves the annotated evidence first."
                            ],
                        }
                    ],
                    "annotation_id": ["annotation-1"],
                    "worker_id": ["worker-1"],
                }
            ],
        },
    }


def test_qasper_selection_maps_answer_and_evidence_to_paragraph() -> None:
    cases = select_qasper_cases({"rows": [_row()]}, limit=1)

    assert len(cases) == 1
    assert cases[0].gold_answer == "the annotated evidence"
    assert cases[0].gold_evidence_ids == ("paper-1:paragraph:0001",)
    assert len(qasper_candidates(cases[0])) == 3


def test_qasper_evidence_mapping_accepts_long_contained_annotation() -> None:
    row = _row()
    row["full_text"]["paragraphs"][1][0] = (  # type: ignore[index]
        "The dense method retrieves the annotated evidence first. "
        "This additional sentence appears in the paper paragraph."
    )
    passages = _paper_passages(row)

    evidence_ids = _map_evidence(
        [
            "The dense method retrieves the annotated evidence first. "
            "This additional sentence appears"
        ],
        passages,
    )

    assert evidence_ids == ("paper-1:paragraph:0001",)


def test_qasper_selection_skips_yes_no_annotations() -> None:
    row = _row()
    row["qas"]["answers"][0]["answer"][0]["yes_no"] = True  # type: ignore[index]

    try:
        select_qasper_cases({"rows": [row]}, limit=1)
    except RuntimeError as error:
        assert "Only 0 eligible" in str(error)
    else:
        msg = "Expected yes/no-only row to be excluded"
        raise AssertionError(msg)
