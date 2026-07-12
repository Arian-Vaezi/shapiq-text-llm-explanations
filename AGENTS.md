# AGENTS.md

This role of this file is to describe common mistakes and confusion points that agents might encounter as they work in this project. If you ever encounter something in the project that surprises you, please alert the developer working with you and indicate that this is the case in the Agent.md file to help prevent future agents from having the same issue.

## Commands to interact with the codebase which you should run:

### Build Docs (only use this command verbatim from the project root)

```bash
rm -rf docs/source/generated docs/source/auto_examples && uv run sphinx-build -b html docs/source docs/build/html
```

### Run Pre-commit (takes only 3s)

```bash
uv run pre-commit run --all-files
```

## demos/rag_retrieval_explanation structure

The demo was refactored (June 2026). Key structural facts:

- **`core/`** is the only implementation directory for the Python RAG pipeline.
  Import directly from its focused modules; the former demo-root compatibility
  shims were removed because there are no external callers to preserve.
- **`core/schemas.py`** is the single source of truth for all domain dataclasses
  (`RetrievedChunk`, `CandidateChunk`, `RankedChunk`, `PDFPage`, `RetrievalDebugInfo`).
- Import pipeline operations directly from `core/chunking.py`, `core/retrieval.py`,
  and `core/generation.py`; there is no aggregate `rag_pipeline.py` module.

## Project-specific surprises to remember

- In `src/shapiq/game.py`, the base `Game` stores `player_name_lookup` but does
  not expose a `player_names` list. Demo code that needs display names should
  keep its own labels or use the demo's retrieved chunk titles.
- Generation and model-backed value functions use `llama-cpp-python` with a
  local GGUF file. The default path is
  `models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf`; model files are gitignored.
- Dense retrieval remains optional and still uses `transformers`. The default
  PDF path uses TF-IDF and does not require Hugging Face.
- `demos/rag_retrieval_explanation/eval_report.html` is a static report with a
  generated detail block. Rebuild the per-question interaction panels with
  `uv run python -m demos.rag_retrieval_explanation.evals.reporting.build_eval_report`
  instead of hand-editing the generated block.
