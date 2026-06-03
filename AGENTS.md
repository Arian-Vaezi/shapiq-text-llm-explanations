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

## demos/rag_retrieval_explanation structure (post-refactor)

The demo was refactored (June 2026). Key structural facts:

- **`core/`** is the canonical module directory. `evaluation.py`, `rag_game.py`,
  `rag_pipeline.py`, `value_functions.py`, `model_backends.py`, `sample_data.py`
  at the demo root are **backward-compat shims** that re-export from `core/`.
  Do not edit them directly — edit `core/` instead.
- **`core/schemas.py`** is the single source of truth for all domain dataclasses
  (`RetrievedChunk`, `CandidateChunk`, `RankedChunk`, `PDFPage`, `RetrievalDebugInfo`).
- **`core/rag_pipeline.py`** is itself a shim re-exporting from `core/chunking.py`,
  `core/retrieval.py`, and `core/generation.py`. Edit those three files, not
  `core/rag_pipeline.py` directly.
- **`evals/runners/`** contains the real runner logic. `evals/run_controlled_eval.py`
  and `evals/run_pdf_smoke_eval.py` are shims that delegate to `evals/runners/`.
- **`evals/configs/`** holds the eval config YAMLs. The grid runner reads these;
  do not inline config values into runner scripts.
- **`evals/runs/`** is gitignored. Run directories follow the canonical schema:
  `manifest.json`, `summary.csv`, `report.md`, `plots/`, `artifacts/`.
- **PyYAML** is needed for the grid runner (`run_grid_eval.py`). If missing:
  `uv add pyyaml`.

## Project-specific surprises to remember

- In `src/shapiq/game.py`, the base `Game` stores `player_name_lookup` but does
  not expose a `player_names` list. Demo code that needs display names should
  keep its own labels or use the demo's retrieved chunk titles.
- `demos/rag_retrieval_explanation` may use Hugging Face gated models. Having a
  Hugging Face account is not enough: the user may also need to accept access on
  the specific model page and run `uv run huggingface-cli login` once.
- Gemma 4 model repos use `model_type: gemma4`. If local `transformers` does not
  recognize that architecture, the fix is a newer `transformers` build with
  Gemma 4 support; this is separate from Hugging Face authentication.
