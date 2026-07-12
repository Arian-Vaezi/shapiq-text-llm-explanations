"""Boundary validation tests for PDF API requests."""

from __future__ import annotations

from demos.rag_retrieval_explanation.backend.api import MAX_PDF_BYTES, app
from fastapi.testclient import TestClient

CLIENT = TestClient(app)
BASE_FORM = {
    "question": "What is the main result?",
    "settings_json": '{"value_modes":["Lexical grounding"]}',
}


def test_pdf_endpoint_rejects_non_pdf_content() -> None:
    """A filename or MIME type alone must not bypass PDF content validation."""
    response = CLIENT.post(
        "/api/runs/pdf",
        data=BASE_FORM,
        files={"file": ("fake.pdf", b"not a pdf", "application/pdf")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "The uploaded file is not a valid PDF."


def test_pdf_endpoint_rejects_oversized_upload() -> None:
    """Uploads larger than the configured bound must fail with HTTP 413."""
    response = CLIENT.post(
        "/api/runs/pdf",
        data=BASE_FORM,
        files={"file": ("large.pdf", b"%PDF-" + b"x" * MAX_PDF_BYTES, "application/pdf")},
    )

    assert response.status_code == 413


def test_pdf_endpoint_rejects_invalid_chunk_overlap() -> None:
    """Chunk overlap must stay below the selected chunk size."""
    response = CLIENT.post(
        "/api/runs/pdf",
        data={**BASE_FORM, "chunk_words": "100", "chunk_overlap": "100"},
        files={"file": ("valid.pdf", b"%PDF-1.7\n", "application/pdf")},
    )

    assert response.status_code == 400
    assert "chunk_overlap" in response.json()["detail"]


def test_compare_endpoint_requires_model_id_list() -> None:
    """Comparison model IDs must be a bounded JSON list of strings."""
    response = CLIENT.post(
        "/api/runs/compare/pdf",
        data={**BASE_FORM, "model_ids_json": '"gemma4-e4b"'},
        files={"file": ("valid.pdf", b"%PDF-1.7\n", "application/pdf")},
    )

    assert response.status_code == 400
    assert "list of strings" in response.json()["detail"]
