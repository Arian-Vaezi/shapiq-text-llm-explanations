"""Local FastAPI server for the RAG retrieval explanation dashboard."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..core.sample_data import SAMPLE_TRACES
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

app = FastAPI(
    title="RAG Retrieval Explanation API",
    version="1.0.0",
    description="Local API for retrieval attribution with shapiq and llama.cpp.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/meta")
def metadata() -> dict[str, object]:
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
    try:
        return run_sample(request.scenario_name, request.settings)
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/runs/pdf", response_model=RunResponse)
async def pdf_run(
    file: UploadFile = File(...),
    question: str = Form(...),
    reference_answer: str = Form(""),
    settings_json: str = Form(...),
    retrieval_method: str = Form("Dense embeddings"),
    retrieval_top_k: int = Form(6),
    chunk_words: int = Form(180),
    chunk_overlap: int = Form(40),
    embedding_model_id: str = Form(BGE_BASE_MODEL_ID),
    embedding_device: str = Form("auto"),
) -> dict:
    if file.content_type != "application/pdf" and not (file.filename or "").lower().endswith(
        ".pdf"
    ):
        raise HTTPException(status_code=400, detail="Upload a PDF file.")
    try:
        settings = RuntimeSettings.model_validate_json(settings_json)
        return run_pdf(
            filename=file.filename or "document.pdf",
            pdf_bytes=await file.read(),
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
    file: UploadFile = File(...),
    question: str = Form(...),
    reference_answer: str = Form(""),
    settings_json: str = Form(...),
    model_ids_json: str = Form(...),
    retrieval_method: str = Form("Dense embeddings"),
    retrieval_top_k: int = Form(6),
    chunk_words: int = Form(180),
    chunk_overlap: int = Form(40),
    embedding_model_id: str = Form(BGE_BASE_MODEL_ID),
    embedding_device: str = Form("auto"),
) -> dict:
    if file.content_type != "application/pdf" and not (file.filename or "").lower().endswith(
        ".pdf"
    ):
        raise HTTPException(status_code=400, detail="Upload a PDF file.")
    try:
        settings = RuntimeSettings.model_validate_json(settings_json)
        model_ids = json.loads(model_ids_json)
        return run_comparison(
            source="pdf",
            pdf_filename=file.filename or "document.pdf",
            pdf_bytes=await file.read(),
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
        candidate = (FRONTEND_DIST / full_path).resolve()
        if candidate.is_file() and FRONTEND_DIST.resolve() in candidate.parents:
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")
