"""Integrity tests for published result loading and report generation."""

from __future__ import annotations

import json

import pytest

from demos.rag_retrieval_explanation.evals.reporting.build_eval_report import (
    _hardware_description,
    _load_artifacts,
)


def test_report_loader_rejects_summary_artifact_mismatch(tmp_path) -> None:
    """Published summaries and per-case artifacts must have identical case IDs."""
    run_dir = tmp_path / "result"
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)
    (run_dir / "summary.csv").write_text("case_id\nexpected\n", encoding="utf-8")
    (artifacts_dir / "extra.json").write_text(
        json.dumps({"case": {"case_id": "extra"}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Summary/artifact mismatch"):
        _load_artifacts(run_dir)


def test_hardware_description_uses_manifest_values() -> None:
    """Report hardware wording must come from the recorded run manifest."""
    manifest = {
        "embedding_device_resolved": "cuda",
        "n_gpu_layers": 4,
    }

    assert _hardware_description(manifest) == (
        "BGE embeddings on CUDA; 4 llama.cpp layers offloaded to CUDA"
    )
