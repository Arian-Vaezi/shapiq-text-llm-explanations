# Controlled Retrieval and Attribution Evaluation

- Cases: 50
- Mean retrieval Recall@k: 0.882
- Mean retrieval MRR: 0.899
- Top attribution is gold: 0.854
- Mean gold attribution mass: 0.632
- Mean gold deletion AUC: 0.051
- Mean leave-one-out deletion AUC: 0.071
- Mean random deletion AUC: 0.106
- Interaction sign recovery: 0.659
- Interaction pairs: 41 evaluable / 60 labelled; 19 skipped

The controlled corpus is searched by the same retrieval code as the PDF path.
Gold-answer attribution and generated-answer attribution are stored separately.
Missing-evidence cases have no gold evidence and are excluded from evidence recall.
Interaction accuracy excludes labelled pairs where either passage was not retrieved.

See `summary.csv` and `artifacts/` for per-case results.
