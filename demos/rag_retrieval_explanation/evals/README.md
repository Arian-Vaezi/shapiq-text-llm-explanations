# RAG Retrieval Explanation Evals

This folder records explanation-centered evaluations for the RAG Shapley demo.
The goal is not to benchmark the retriever against BEIR or CRAG. The goal is to
check whether the explanation behaves faithfully for known evidence patterns.

## What To Evaluate

Use the controlled scenarios in `sample_data.py` as small, repeatable tests:

- direct evidence should receive high positive attribution;
- distractors should not outrank direct evidence;
- missing evidence should keep full-context support low;
- complementary evidence should show positive interaction;
- redundant evidence should split credit and show redundancy.

For uploaded PDFs, use the generic smoke checks to protect the retrieval path
from obvious regressions on fixed questions. These checks are intentionally
offline rather than part of the main Streamlit UI.

## Run Controlled Evals

From the project root:

```bash
make eval
```

This writes a timestamped folder under `runs/` with one subfolder per scenario.
The default uses the local lexical value function, so it is a stable regression
check rather than a final model-backed experiment. If a scenario expectation
does not hold under lexical scoring, treat that as a value-function finding and
rerun with a stronger scorer before making thesis-level claims.

## Run PDF Smoke Evals

Create a local JSON case file based on `pdf_cases.example.json`:

```json
[
  {
    "id": "my_pdf_case",
    "pdf_path": "path/to/file.pdf",
    "question": "What is the main contribution?",
    "method": "TF-IDF",
    "expected_evidence_terms": ["contribution"]
  }
]
```

Then run:

```bash
make pdf-smoke PDF_CASES=path/to/cases.json
```

Use the same PDF files you want to demonstrate or discuss in the thesis. Keep
large PDFs outside git unless they are small, redistributable fixtures.

## Run Record Layout

Use one folder per run:

```text
runs/
  2026-06-02_baseline_lexical_ksii/
    config.yaml
    metrics.json
    shapley_values.csv
    interactions.csv
    deletion_curve.csv
    insertion_curve.csv
    notes.md
```

Recommended `config.yaml` fields:

```yaml
git_commit: "<commit>"
scenario: "2008 Beijing Olympics host city"
input_source: "controlled scenario"
value_function: "Lexical grounding"
interaction_index: "k-SII"
budget: 16
top_k: 4
retrieval_method: "fixture order"
embedding_model: null
generator_model: null
prompt_version: null
```

Recommended `metrics.json` fields:

```json
{
  "full_context_support": 0.91,
  "top_expected_evidence_rank": 1,
  "score_without_top_shapley_chunk": 0.38,
  "top_chunk_deletion_drop": 0.53,
  "deletion_auc_shapley": 0.41,
  "deletion_auc_retrieval": 0.55,
  "deletion_auc_random": 0.63,
  "insertion_auc_shapley": 0.76,
  "interaction_hit": true,
  "missing_evidence_false_positive": false
}
```

Interpret deletion AUC as lower-is-better when the curve plots remaining support
after evidence removal. Interpret insertion AUC as higher-is-better.
