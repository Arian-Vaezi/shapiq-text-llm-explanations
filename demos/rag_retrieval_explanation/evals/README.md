# Evaluation pipeline

This directory contains the inputs, runners, published results, and reporting
tools for `../eval_report.html`.

## Data flow

```text
cases/ → experiments/ → runs/ → results/ → reporting/ → eval_report.html
```

- `cases/` contains the frozen controlled, QASPER, and Artemis inputs.
- `experiments/` contains the programs that produce per-case artifacts.
- `runs/` is an ignored workspace for new or incomplete runs.
- `results/` contains the reviewed artifacts used by the submitted report and
  is tracked by Git.
- `reporting/` derives interaction and stability summaries, verifies reported
  metrics, and rebuilds the HTML report.
- `slurm/` contains the five production jobs used for the 50-case GPU runs.

## Published results

```text
results/
├── controlled/
│   ├── bge_lexical/
│   ├── bge_target_likelihood/
│   ├── bge_contrastive_likelihood/
│   └── tfidf_contrastive_likelihood/
├── qasper/bge_contrastive_likelihood/
├── artemis/bge_contrastive_likelihood/
├── derived/interactions/
├── derived/stability/
└── report_verification.json
```

Each primary result directory contains a manifest, CSV and JSON summaries, a
short Markdown report, and the per-case artifacts used by the HTML builder.

## Rebuild and verify the report

Run from the repository root:

```bash
make verify-report-results
make eval-report-html
```

The equivalent module commands are:

```bash
uv run python -m demos.rag_retrieval_explanation.evals.reporting.verify_report_results
uv run python -m demos.rag_retrieval_explanation.evals.reporting.build_eval_report
```

Rebuild the derived controlled summaries with:

```bash
make eval-interactions-controlled
make eval-stability
```

## Run experiments

Local experiment targets write to `runs/` by default:

```bash
make eval-controlled
make eval-controlled-model
make eval-controlled-bge-lexical
make eval-qasper-smoke
make eval-qasper
```

The Slurm files in `slurm/` record the model, retrieval, and GPU settings used
for the submitted model-backed results. A new run should not replace a published
result until its manifest, summary, and full set of case artifacts have been
checked.

Create the log directory from the repository root before submitting one of these
jobs, because Slurm opens the configured output files before the job starts:

```bash
mkdir -p logs
```
