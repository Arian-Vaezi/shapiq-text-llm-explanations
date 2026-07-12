# RAG evaluation pipeline

This directory contains the complete reproduction chain for the static
`eval_report.html` report.

- `cases/`: frozen controlled, QASPER, and Artemis evaluation manifests.
- `experiments/`: benchmark loaders, metrics, and runners that produce run artifacts.
- `reporting/`: aggregation, stability analysis, HTML generation, and verification.
- `slurm/`: the five production Slurm jobs used for the reported 50-case runs.
- `runs/`: generated evaluation artifacts consumed by the reporting tools.

From the repository root, rebuild and verify the report with:

```bash
uv run python -m demos.rag_retrieval_explanation.evals.reporting.build_eval_report
uv run python -m demos.rag_retrieval_explanation.evals.reporting.verify_report_results
```
