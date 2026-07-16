"""Local FastAPI server for the RAG retrieval explanation dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from demos.rag_retrieval_explanation.core.sample_data import SAMPLE_TRACES

from .model_registry import DEFAULT_MODEL_ID, MODEL_REGISTRY, get_model_status
from .schemas import (
    ComparisonResponse,
    ComparisonSampleRequest,
    RunResponse,
    RuntimeSettings,
    SampleRunRequest,
)
from .services import (
    BGE_BASE_MODEL_ID,
    DEFAULT_GGUF_PATH,
    VALUE_FUNCTIONS,
    run_comparison,
    run_pdf,
    run_sample,
)

MAX_PDF_BYTES = 25 * 1024 * 1024
MAX_COMPARISON_MODELS = 4

app = FastAPI(
    title="RAG Retrieval Explanation API",
    version="1.0.0",
    description="Local API for retrieval attribution with shapiq and llama.cpp.",
)
app.add_middleware(
    CORSMiddleware,  # ty: ignore[invalid-argument-type]
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _read_pdf_upload(file: UploadFile) -> bytes:
    """Read and validate one bounded PDF upload."""
    pdf_bytes = await file.read(MAX_PDF_BYTES + 1)
    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="PDF files must be 25 MB or smaller.")
    if b"%PDF-" not in pdf_bytes[:1024]:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid PDF.")
    return pdf_bytes


def _validate_pdf_options(
    *,
    question: str,
    retrieval_top_k: int,
    chunk_words: int,
    chunk_overlap: int,
) -> str:
    """Validate cross-field PDF retrieval options and normalize the question."""
    normalized_question = question.strip()
    if not normalized_question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    if not 1 <= retrieval_top_k <= 20:
        raise HTTPException(status_code=400, detail="retrieval_top_k must be between 1 and 20.")
    if not 50 <= chunk_words <= 1000:
        raise HTTPException(status_code=400, detail="chunk_words must be between 50 and 1000.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_words:
        raise HTTPException(
            status_code=400,
            detail="chunk_overlap must be non-negative and smaller than chunk_words.",
        )
    return normalized_question


def _parse_model_ids(value: str) -> list[str]:
    """Parse a bounded JSON list of model IDs for comparison runs."""
    try:
        model_ids = json.loads(value)
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=400, detail="model_ids_json must be valid JSON.") from error
    if not isinstance(model_ids, list) or not all(isinstance(item, str) for item in model_ids):
        raise HTTPException(status_code=400, detail="model_ids_json must be a list of strings.")
    model_ids = list(dict.fromkeys(model_ids))
    if not 1 <= len(model_ids) <= MAX_COMPARISON_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Choose between 1 and {MAX_COMPARISON_MODELS} distinct models.",
        )
    return model_ids


@app.get("/api/health")
def health() -> dict[str, str]:
    """Return a lightweight process health response."""
    return {"status": "ok"}


@app.get("/api/meta")
def metadata() -> dict[str, object]:
    """Return dashboard scenarios, defaults, and available local models."""
    return {
        "scenarios": [
            {
                "name": name,
                "question": trace["question"],
                "takeaway": trace["takeaway"],
            }
            for name, trace in SAMPLE_TRACES.items()
        ],
        "value_functions": [{"name": name, **details} for name, details in VALUE_FUNCTIONS.items()],
        "defaults": {
            "model_path": DEFAULT_GGUF_PATH,
            "model_id": DEFAULT_MODEL_ID,
            "value_modes": ["Local contrastive likelihood"],
            "interaction_index": "k-SII",
            "max_order": 2,
            "budget": 64,
            "n_ctx": 4096,
            "n_gpu_layers": 20,
            "n_threads": 0,
            "max_new_tokens": 96,
            "retrieval_method": "Dense embeddings",
            "retrieval_top_k": 6,
            "chunk_words": 180,
            "chunk_overlap": 40,
            "embedding_model_id": BGE_BASE_MODEL_ID,
            "embedding_device": "auto",
        },
        "models": [get_model_status(mid) for mid in MODEL_REGISTRY],
        "default_model_id": DEFAULT_MODEL_ID,
    }


@app.post("/api/runs/sample", response_model=RunResponse)
def sample_run(request: SampleRunRequest) -> dict:
    """Run one bundled demonstration scenario."""
    try:
        return run_sample(request.scenario_name, request.settings)
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/runs/pdf", response_model=RunResponse)
async def pdf_run(
    file: Annotated[UploadFile, File()],
    question: Annotated[str, Form(max_length=2000)],
    settings_json: Annotated[str, Form(max_length=50_000)],
    reference_answer: Annotated[str, Form(max_length=10_000)] = "",
    retrieval_method: Annotated[str, Form()] = "Dense embeddings",
    retrieval_top_k: Annotated[int, Form()] = 6,
    chunk_words: Annotated[int, Form()] = 180,
    chunk_overlap: Annotated[int, Form()] = 40,
    embedding_model_id: Annotated[str, Form(max_length=500)] = BGE_BASE_MODEL_ID,
    embedding_device: Annotated[str, Form(max_length=50)] = "auto",
) -> dict:
    """Run retrieval and attribution for one uploaded PDF."""
    question = _validate_pdf_options(
        question=question,
        retrieval_top_k=retrieval_top_k,
        chunk_words=chunk_words,
        chunk_overlap=chunk_overlap,
    )
    pdf_bytes = await _read_pdf_upload(file)
    try:
        settings = RuntimeSettings.model_validate_json(settings_json)
        return run_pdf(
            filename=file.filename or "document.pdf",
            pdf_bytes=pdf_bytes,
            question=question,
            reference_answer=reference_answer,
            settings=settings,
            retrieval_method=retrieval_method,
            retrieval_top_k=retrieval_top_k,
            chunk_words=chunk_words,
            chunk_overlap=chunk_overlap,
            embedding_model_id=embedding_model_id,
            embedding_device=embedding_device,
        )
    except (ValueError, RuntimeError, OSError, ImportError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/runs/compare/sample", response_model=ComparisonResponse)
def compare_sample_run(request: ComparisonSampleRequest) -> dict:
    """Compare configured models on one bundled scenario."""
    try:
        return run_comparison(
            source="sample",
            scenario_name=request.scenario_name,
            base_settings=request.settings,
            model_ids=request.model_ids,
        )
    except (ValueError, RuntimeError, OSError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/runs/compare/pdf", response_model=ComparisonResponse)
async def compare_pdf_run(
    file: Annotated[UploadFile, File()],
    question: Annotated[str, Form(max_length=2000)],
    settings_json: Annotated[str, Form(max_length=50_000)],
    model_ids_json: Annotated[str, Form(max_length=10_000)],
    reference_answer: Annotated[str, Form(max_length=10_000)] = "",
    retrieval_method: Annotated[str, Form()] = "Dense embeddings",
    retrieval_top_k: Annotated[int, Form()] = 6,
    chunk_words: Annotated[int, Form()] = 180,
    chunk_overlap: Annotated[int, Form()] = 40,
    embedding_model_id: Annotated[str, Form(max_length=500)] = BGE_BASE_MODEL_ID,
    embedding_device: Annotated[str, Form(max_length=50)] = "auto",
) -> dict:
    """Compare configured models on one uploaded PDF."""
    question = _validate_pdf_options(
        question=question,
        retrieval_top_k=retrieval_top_k,
        chunk_words=chunk_words,
        chunk_overlap=chunk_overlap,
    )
    pdf_bytes = await _read_pdf_upload(file)
    try:
        settings = RuntimeSettings.model_validate_json(settings_json)
        model_ids = _parse_model_ids(model_ids_json)
        return run_comparison(
            source="pdf",
            pdf_filename=file.filename or "document.pdf",
            pdf_bytes=pdf_bytes,
            question=question,
            reference_answer=reference_answer,
            base_settings=settings,
            model_ids=model_ids,
            retrieval_method=retrieval_method,
            retrieval_top_k=retrieval_top_k,
            chunk_words=chunk_words,
            chunk_overlap=chunk_overlap,
            embedding_model_id=embedding_model_id,
            embedding_device=embedding_device,
        )
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if FRONTEND_DIST.is_dir():
    assets = FRONTEND_DIST / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def frontend(full_path: str) -> FileResponse:
        """Serve a built asset or the single-page application shell."""
        candidate = (FRONTEND_DIST / full_path).resolve()
        if candidate.is_file() and FRONTEND_DIST.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")
