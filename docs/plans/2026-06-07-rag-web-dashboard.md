# RAG Web Dashboard Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the Streamlit RAG explanation demo with a local React research dashboard backed by FastAPI.

**Architecture:** Keep the existing Python `core/` modules as the computation layer. Add a small FastAPI adapter that accepts sample or PDF inputs and serializes attribution results as JSON. Add a Vite/React/TypeScript client that renders the workspace, evidence, interactions, audit curves, and settings.

**Tech Stack:** Python, FastAPI, Pydantic, shapiq, llama.cpp, React, TypeScript, Vite, Recharts.

---

### Task 1: Extract the run service

**Files:**
- Create: `demos/rag_retrieval_explanation/backend/services.py`
- Create: `demos/rag_retrieval_explanation/backend/schemas.py`
- Test: `tests/test_rag_web_backend.py`

Implement sample trace loading, PDF retrieval/generation, Shapley execution, audit metrics, and JSON-safe serialization without importing Streamlit.

### Task 2: Add the HTTP API

**Files:**
- Create: `demos/rag_retrieval_explanation/backend/api.py`
- Create: `demos/rag_retrieval_explanation/backend/__init__.py`

Expose health, metadata, sample-run, and PDF-run endpoints. Enable localhost Vite CORS and serve the built frontend when present.

### Task 3: Build the research dashboard

**Files:**
- Create: `demos/rag_retrieval_explanation/frontend/package.json`
- Create: `demos/rag_retrieval_explanation/frontend/src/*`

Build a restrained black-and-white workspace with settings, evidence attribution bars, interaction matrix, audit curves, retrieval details, and responsive layout.

### Task 4: Replace commands and dependencies

**Files:**
- Modify: `Makefile`
- Modify: `pyproject.toml`
- Modify: `demos/rag_retrieval_explanation/requirements.txt`
- Modify: `demos/rag_retrieval_explanation/README.md`
- Delete: `demos/rag_retrieval_explanation/app.py`

Remove Streamlit and Altair, add FastAPI/Uvicorn, and document one-command local startup.

### Task 5: Verify end to end

Run:

```bash
uv run pytest tests/test_rag_web_backend.py tests/test_rag_retrieval_policy.py tests/test_rag_llama_cpp_backend.py
npm --prefix demos/rag_retrieval_explanation/frontend run build
uv run pre-commit run --all-files
```

Start the API and verify `/api/health`, `/api/meta`, a sample explanation request, and the built dashboard.
