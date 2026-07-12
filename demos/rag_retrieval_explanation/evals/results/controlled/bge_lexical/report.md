# Controlled Retrieval and Attribution Evaluation

- Cases: 50
- Mean retrieval Recall@k: 0.947
- Mean retrieval MRR: 0.945
- Top attribution is gold: 0.951
- Mean gold attribution mass: 0.739
- Mean gold deletion AUC: 0.236
- Mean leave-one-out deletion AUC: 0.261
- Mean random deletion AUC: 0.500
- Interaction sign recovery: 0.906
- Interaction pairs: 53 evaluable / 60 labelled; 7 skipped

The controlled corpus is searched by the same retrieval code as the PDF path.
Gold-answer attribution and generated-answer attribution are stored separately.
Missing-evidence cases have no gold evidence and are excluded from evidence recall.
Interaction accuracy excludes labelled pairs where either passage was not retrieved.

See `summary.csv` and `artifacts/` for per-case results.
