# RAG Retrieval Explanation

This demo explains which retrieved passages support a RAG answer. It combines
document retrieval, local llama.cpp generation, and Shapley attribution in a
FastAPI and React application.

## Start here

The main research output is [`eval_report.html`](eval_report.html). It contains
the reported controlled, QASPER, and Artemis results together with per-case
attributions and interaction details.

The report is built from the versioned artifacts in `evals/results/`; no model
inference is required to open, rebuild, or verify it.

## Project structure

```text
rag_retrieval_explanation/
├── backend/          FastAPI routes and application services
├── core/             Retrieval, generation, Shapley games, and evaluation
├── data/             Documents bundled with the demo
├── evals/
│   ├── cases/        Frozen evaluation inputs
│   ├── experiments/  Controlled and QASPER experiment runners
│   ├── reporting/    Result aggregation, verification, and HTML generation
│   ├── results/      Versioned results used by the final report
│   ├── runs/         Ignored workspace for new experiment outputs
│   └── slurm/        Slurm jobs used for the reported GPU experiments
├── frontend/         React and Vite interface
├── tests/            Demo-specific tests
└── eval_report.html  Static evaluation report
```

The data flow is:

```text
cases → experiments → runs → reviewed results → reporting → eval_report.html
```

## Evaluation scope

- Controlled benchmark: 50 questions over 200 passages, with 60 labelled
  complementary, redundant, or conflicting evidence pairs.
- QASPER: a frozen 50-case scientific question-answering subset.
- Artemis: a prior-knowledge control using the bundled document in `data/`.

See [`evals/README.md`](evals/README.md) for the experiment and report commands.

## Setup

Run commands from the repository root:

```bash
uv sync --group rag_demo --group test
npm --prefix demos/rag_retrieval_explanation/frontend install
```

Model files are stored under `models/` and are not committed. To inspect or
download the registered GGUF models:

```bash
uv run python scripts/download_models.py --list
uv run python scripts/download_models.py --model qwen3-8b
```

## Run the application

```bash
make app
```

Open <http://127.0.0.1:8000>. For frontend development, run `make api` and
`make frontend` in separate terminals.

## Verify the submitted results

```bash
make test
make verify-report-results
make eval-report-html
```

`make verify-report-results` recomputes the reported metrics from
`evals/results/`. `make eval-report-html` rebuilds the generated sections of the
static report from the same artifacts.

## Run new experiments

The local evaluation targets write new outputs to the ignored `evals/runs/`
workspace:

```bash
make eval-controlled
make eval-controlled-model
make eval-controlled-bge-lexical
make eval-qasper-smoke
make eval-qasper
```

The production GPU commands used for the submitted results are recorded in
`evals/slurm/`. New runs should only be moved into `evals/results/` after their
summaries and per-case artifacts have been checked.
