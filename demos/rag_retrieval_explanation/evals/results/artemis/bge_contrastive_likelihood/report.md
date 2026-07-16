# Controlled Retrieval and Attribution Evaluation

- Cases: 10
- Mean retrieval Recall@k: 1.000
- Mean retrieval MRR: 0.950
- Top attribution is gold: 0.900
- Mean gold attribution mass: 0.834
- Mean gold deletion AUC: 0.044
- Mean leave-one-out deletion AUC: 0.044
- Mean random deletion AUC: 0.093
- Interaction sign recovery: n/a
- Interaction pairs: 0 evaluable / 0 labelled; 0 skipped

The controlled corpus is searched by the same retrieval code as the PDF path.
Gold-answer attribution and generated-answer attribution are stored separately.
Missing-evidence cases have no gold evidence and are excluded from evidence recall.
Interaction accuracy excludes labelled pairs where either passage was not retrieved.

See `summary.csv` and `artifacts/` for per-case results.
