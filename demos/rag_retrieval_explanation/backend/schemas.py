"""HTTP schemas for the local RAG explanation dashboard."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RuntimeSettings(BaseModel):
    """Model and explanation settings shared by sample and PDF runs."""

    value_modes: list[str] = Field(default_factory=lambda: ["Lexical grounding"])
    interaction_index: Literal["SV", "k-SII", "STII", "FSII"] = "k-SII"
    max_order: int = Field(default=2, ge=1, le=3)
    budget: int = Field(default=64, ge=16, le=512)
    model_path: str = ""
    model_id: str | None = None
    n_ctx: int = Field(default=4096, ge=512, le=65536)
    n_gpu_layers: int = Field(default=-1, ge=-1)
    n_threads: int = Field(default=0, ge=0, le=64)
    max_new_tokens: int = Field(default=96, ge=16, le=512)


class SampleRunRequest(BaseModel):
    """Request to explain one bundled sample trace."""

    scenario_name: str
    settings: RuntimeSettings


class ComparisonSampleRequest(BaseModel):
    """Request to compare multiple models on a bundled sample trace."""

    scenario_name: str
    settings: RuntimeSettings
    model_ids: list[str]


class ModelRunResultResponse(BaseModel):
    """Single-model result within a comparison run."""

    model_id: str
    model_display_name: str
    status: Literal["completed", "missing_model", "failed"]
    error: str | None = None
    expected_path: str | None = None
    result: RunResponse | None = None


class ComparisonResponse(BaseModel):
    """Response containing per-model results for a comparison run."""

    question: str
    source: str
    results: list[ModelRunResultResponse]


class AttributionPoint(BaseModel):
    """One first-order chunk attribution."""

    player: int
    chunk: str
    attribution: float
    loo: float | None = None


class CurvePoint(BaseModel):
    """One point on a deletion or insertion curve."""

    step: int
    score: float
    label: str


class FaithfulnessResponse(BaseModel):
    """Compact faithfulness diagnostics for an explanation."""

    full_score: float
    score_without_top: float
    score_with_top_only: float
    deletion_drop: float
    sufficiency_gap: float


class ExplanationResultResponse(BaseModel):
    """Attribution and faithfulness results for one value function."""

    value_name: str
    value_description: str
    value_formula: str
    full_score: float
    empty_score: float
    top_chunk: str
    top_score: float
    attributions: list[AttributionPoint]
    pairwise_matrix: list[list[float]]
    strongest_pair: str
    strongest_pair_value: float
    faithfulness: FaithfulnessResponse
    deletion_curve: list[CurvePoint]
    insertion_curve: list[CurvePoint]
    random_deletion_baseline: list[float]
    random_insertion_baseline: list[float]
    retrieval_deletion_baseline: list[float] | None = None


class ChunkResponse(BaseModel):
    """Retrieved chunk returned to the dashboard."""

    title: str
    text: str
    page_number: int | None = None
    chunk_type: str = "body_text"
    section_title: str = ""
    retrieval_score: float | None = None
    dense_score: float | None = None
    keyword_score: float | None = None
    rerank_score: float | None = None
    flags: list[str] = Field(default_factory=list)


class AnswerTargetResponse(BaseModel):
    """One answer target and its independently computed explanations."""

    target_type: Literal["reference_answer", "generated_answer"]
    label: str
    answer: str
    results: list[ExplanationResultResponse]


class RunResponse(BaseModel):
    """Complete result of one retrieval explanation run."""

    run_name: str
    source: str
    answer_target_type: Literal["reference_answer", "generated_answer"]
    retrieval_method: str
    retrieval_top_k: int
    question: str
    answer: str
    takeaway: str
    chunks: list[ChunkResponse]
    retrieval_debug: dict[str, object] | None = None
    results: list[ExplanationResultResponse]
    targets: list[AnswerTargetResponse]
