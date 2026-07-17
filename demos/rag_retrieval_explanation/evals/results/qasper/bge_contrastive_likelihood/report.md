# QASPER External Evaluation

- Cases: 50
- Mean retrieval Recall@6: 0.549
- Mean retrieval MRR: 0.437
- Gold-answer top attribution is evidence: 0.560
- Gold-answer attribution mass on evidence: 0.379
- Generated-answer top attribution is evidence: 0.420
- Generated-answer attribution mass on evidence: 0.361
- Gold-answer deletion AUC: 0.031
- Random deletion AUC: 0.070
- Closed-book token F1: 0.019
- Gold-context token F1: 0.296
- Retrieved-context token F1: 0.183
- Knowledge-source diagnoses: {"generation_or_model_failure": 49, "retrieval_failure": 1}
- QASPER failure categories: {"generation_failure_with_gold_context": 34, "retrieval_miss": 15, "retrieved_context_failure": 1}

Configuration: QASPER validation, paragraph candidates, BGE-base top-6,
Gemma E4B, and local contrastive likelihood with exact 64-coalition games.

See `summary.csv`, `manifest.json`, and `artifacts/` for case-level results.
