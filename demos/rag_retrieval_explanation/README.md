# RAG Evidence Lab

A local research dashboard for explaining which retrieved chunks support a RAG
answer. The interface is React; FastAPI runs the existing Python retrieval,
shapiq attribution, audit, and llama.cpp generation code.

## Architecture

```text
React + TypeScript dashboard
            |
       FastAPI JSON API
            |
  core/ retrieval + shapiq + llama.cpp
```

The application is designed for a single user on one machine. It has no login,
database, cloud inference, or Hugging Face authentication requirement.

## Features

- Bundled research scenarios or uploaded PDF input.
- End-to-end controlled evaluation over a shared 108-passage corpus with 27
  questions.
- Interaction labels for 30 complementary, redundant, and conflicting passage
  pairs.
- TF-IDF retrieval and optional dense retrieval.
- Local GGUF answer generation through llama.cpp and Metal on Apple Silicon.
- Shapley value and interaction attribution over retrieved chunks.
- Evidence ranking, pairwise interaction matrix, and LOO marginals.
- Deletion curves with random and retrieval-score baselines.
- Configurable value function, interaction index, budget, and model runtime.

## Setup

From the repository root:

```bash
make setup-rag-local
```

This installs the Python and frontend dependencies.

## Models

GGUF model files are stored in `models/llm/` (gitignored). They are **not**
downloaded automatically. Use the download script to fetch them.

### List registered models and local availability

```bash
uv run python scripts/download_models.py --list
```

### Download the recommended default (Qwen3 8B, ~5 GB)

```bash
uv run python scripts/download_models.py --model qwen3-8b
```

### Download the full Qwen3 family

```bash
uv run python scripts/download_models.py --family qwen3
```

### Download the Gemma 4 comparison family (E2B + E4B + 26B-A4B)

```bash
uv run python scripts/download_models.py --family gemma4
```

### Download everything (check sizes with --list first)

```bash
uv run python scripts/download_models.py --all
```

### Force re-download

```bash
uv run python scripts/download_models.py --model qwen3-8b --force
```

### Registered model families

| Family   | Models                              | Sizes        |
|----------|-------------------------------------|--------------|
| Qwen2.5  | qwen2.5-1.5b                        | ~1 GB        |
| Qwen3    | qwen3-1.5b, qwen3-4b, qwen3-8b      | 1–5 GB       |
| Gemma 4  | gemma4-e2b, gemma4-e4b, gemma4-26b-a4b | 3–17 GB   |

Large models (gemma4-26b-a4b, ~17 GB) are registered but not downloaded by
default. Check availability first with `--list`.

### Missing models in the UI

When a selected model is not found locally, the sidebar shows the expected path
and the exact `uv run python scripts/download_models.py --model <id>` command.
In comparison mode, missing models appear as failed cards while other available
models still complete normally.

## Run

Build the frontend and start the local application:

```bash
make app
```

Open <http://127.0.0.1:8000>.

For frontend development, use two terminals:

```bash
make api
make frontend
```

Vite runs at <http://127.0.0.1:5173> and proxies `/api` to FastAPI.

## API

- `GET /api/health`
- `GET /api/meta`
- `POST /api/runs/sample`
- `POST /api/runs/pdf`
- Interactive documentation: <http://127.0.0.1:8000/docs>

## Tests

```bash
make test
make eval-controlled-bge-lexical
make eval-interactions-controlled
make eval-controlled
make eval-stability
make eval-qasper-smoke
make verify-report-results
make frontend-build
uv run pre-commit run --all-files
```

`make eval-controlled` runs the real retriever over 27 labelled questions and
then evaluates gold-answer attribution. `make eval-controlled-model` additionally
runs closed-book, gold-context, and retrieved-context generation plus
generated-answer attribution with the local GGUF model.
`make eval-stability` reads the recorded controlled artifacts and compares
attribution rankings across value functions and retrieval settings.
The controlled benchmark is maintained directly as
`evals/cases/controlled_benchmark.json`; it contains 10 complementary, 10
redundant, and 10 conflicting labelled interaction pairs.
`make eval-controlled-bge-lexical` runs BGE-base retrieval plus lexical
support on this controlled set. `make eval-interactions-controlled` recomputes the
RQ2 table with interaction coverage and conditional sign recovery separated.
`make eval-qasper-smoke` runs the first five frozen QASPER validation cases;
`make eval-qasper` runs the full 50-case external validation.
`make verify-report-results` recomputes the paper tables from saved run
artifacts and warns when older artifacts lack newly added raw faithfulness
curves.

Model-backed controlled interaction runs use the same runner and benchmark path,
but are slower because each coalition requires local likelihood scoring:

```bash
uv run python -m demos.rag_retrieval_explanation.evals.experiments.run_controlled_eval \
  --method "Dense embeddings" \
  --embedding-model models/embedding/bge-base-en-v1.5 \
  --value-function target_likelihood \
  --model-path models/llm/gemma-4-E4B-it-Q4_K_M.gguf \
  --n-gpu-layers -1 \
  --output-dir demos/rag_retrieval_explanation/evals/runs/YYYYMMDD_controlled_interactions_bge_target_ll_e4b
```

Replace `target_likelihood` with `contrastive_likelihood` for contrastive LL.
Use `--method "TF-IDF"` for the TF-IDF + contrastive LL setting. If native
llama.cpp inference fails or is interrupted, keep the run directory out of the
reported tables until `summary.csv` and all per-case artifacts are complete.

## Directory Structure

```text
demos/rag_retrieval_explanation/
├── backend/       # FastAPI schemas, routes, and orchestration
├── core/          # Retrieval, generation, value functions, and shapiq game
├── frontend/      # React + Vite research dashboard
├── data/          # Local demo documents
└── README.md
```
