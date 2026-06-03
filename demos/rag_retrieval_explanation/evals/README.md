# RAG Retrieval Explanation Evals

This directory records evaluation runs for the RAG Shapley attribution demo.
The goal is scientific, reproducible comparison of how RAG pipeline design
choices (chunking, retrieval method, embedding model, value function) affect
retrieval quality, Shapley attribution, and attribution stability.

---

## Directory Layout

```
evals/
  cases/
    controlled_cases.yaml   # expected behaviour for the 5 controlled scenarios
    sample_data.py          # Python fixture traces (loaded by controlled runner)
    artemis_pdf_cases.json  # Artemis paper eval cases
    pdf_cases.example.json  # template for your own PDF cases

  configs/
    quick.yaml              # single config, for CI (no downloads)
    chunking_compare.yaml   # 3 chunk sizes × TF-IDF
    embedding_compare.yaml  # TF-IDF + MiniLM + BGE-small
    full_grid.yaml          # full 3×3×2 research grid

  runners/
    run_controlled_eval.py  # controlled fixture runner
    run_pdf_eval.py         # PDF smoke runner
    run_grid_eval.py        # cross-product grid runner

  runs/
    .gitignore              # generated output (not committed by default)
    YYYY-MM-DD_HHMMSS_<name>/
      config.yaml           # resolved run config
      manifest.json         # git commit, timestamp, package versions
      summary.csv           # one row per (case × config), all metrics
      summary.json          # same as CSV, JSON format
      report.md             # auto-generated narrative report
      plots/
        pass_rate_by_config.png
        shapley_mass_heatmap.png
        rank_stability.png
        retrieval_vs_shapley.png
      artifacts/
        <case>_<config>_retrieved_chunks.csv
        <case>_<config>_shapley_values.csv
        <case>_<config>_answer.txt
```

---

## Quick Start

### Controlled evals (no downloads, runs in seconds)

```bash
make eval
# or explicitly:
make eval-controlled
```

Runs all 5 controlled scenarios with the lexical grounding value function.
Output goes to `evals/runs/<timestamp>_controlled_lexical/`.

### PDF smoke eval

Create a case file first:

```bash
make pdf-cases-template PDF_CASES=local/my_cases.json
# then edit local/my_cases.json
make eval-pdf PDF_CASES=local/my_cases.json
```

### Grid comparisons

```bash
# Compare 3 chunk sizes (TF-IDF, no download)
make compare-chunking PDF_CASES=local/my_cases.json

# Compare TF-IDF vs MiniLM vs BGE-small (downloads embedding models)
make compare-embeddings PDF_CASES=local/my_cases.json

# Full 18-config research grid
make compare-full PDF_CASES=local/my_cases.json
```

Each grid run writes the full canonical output under `evals/runs/`.

---

## Case File Format (JSON)

```json
[
  {
    "id": "my_case_001",
    "pdf_path": "demos/rag_retrieval_explanation/data/my_paper.pdf",
    "question": "What is the main scientific contribution?",
    "expected_evidence_terms": ["contribution", "novel"],
    "expected_answer_keywords": ["contribution"],
    "top_k": 4
  }
]
```

Fields:
- `id` — unique case identifier (used in artifact filenames)
- `pdf_path` — path to the PDF (relative to project root)
- `question` — the RAG question
- `expected_evidence_terms` — terms that should appear in the retrieved text
- `expected_answer_keywords` — terms the generated answer should contain
- `top_k` — number of chunks to retrieve (optional, default 4)
- `embedding_device` — override device for dense retrieval (optional)

---

## Eval Config Format (YAML)

```yaml
name: my_comparison

chunking:
  - chunk_words: 120
    overlap: 25
  - chunk_words: 180
    overlap: 35

retrieval:
  - method: tfidf
  - method: dense
    embedding_model: sentence-transformers/all-MiniLM-L6-v2

generation:
  - backend: extractive

value_functions:
  - lexical_grounding

shapley:
  approximator: kernel_shap
  budget: 256
  random_seed: 42
  max_order: 2

output:
  top_k: 4
  save_artifacts: true
  save_plots: true
```

The grid runner expands the Cartesian product of all list-valued fields.

---

## Canonical Run Directory

Every run writes to the same schema:

```
manifest.json       git commit, branch, timestamp, Python + package versions
config.yaml         resolved run config (not just the input YAML)
summary.csv         one row per (case_id × config_id), columns:
                      case_id, config_id, retrieval_method, embedding_model,
                      chunk_words, overlap, value_function,
                      hit_rate, recall_at_k, mrr,
                      answer_contains_expected, answer_supported,
                      top_shapley_is_expected, shapley_mass_on_expected,
                      rank_corr_retrieval_shapley,
                      deletion_auc, insertion_auc,
                      passed, found_terms, missing_terms, error
summary.json        same in JSON
report.md           auto-generated markdown with tables and plot links
plots/              all comparison PNG files
artifacts/          per-case retrieved chunks, Shapley values, answers
```

---

## Metrics Reference

### Retrieval

| Metric | Description |
|---|---|
| `hit_rate` | 1 if any expected evidence chunk is in top-k, else 0 |
| `recall_at_k` | fraction of expected evidence chunks in top-k |
| `mrr` | 1/rank of first expected chunk (None if unknown) |
| `avg_retrieval_score` | mean rerank score of expected chunks |

### Answer

| Metric | Description |
|---|---|
| `answer_contains_expected` | all expected keywords found in answer |
| `citation_hit` | answer text references a retrieved chunk title |
| `answer_supported` | answer does not contain "context insufficient" patterns |

### Attribution

| Metric | Description |
|---|---|
| `top_shapley_is_expected` | highest-attributed chunk is expected evidence |
| `shapley_mass_on_expected` | fraction of Σ|φ| on expected chunks |
| `rank_corr_retrieval_shapley` | Spearman ρ between retrieval rank and Shapley rank |
| `deletion_auc` | area under deletion curve (lower = more decisive) |
| `insertion_auc` | area under insertion curve (higher = more decisive) |
| `kendall_tau` | cross-config Shapley ranking stability (in rank_stability.png) |

---

## Plots Reference

| Plot | What it shows |
|---|---|
| `pass_rate_by_config.png` | bar chart: evidence hit rate per config |
| `shapley_mass_heatmap.png` | heatmap: cases × configs, Shapley mass on expected |
| `rank_stability.png` | pairwise Kendall-τ between config Shapley rankings |
| `retrieval_vs_shapley.png` | scatter: retrieval score vs Shapley mass per (case × config) |
| `<case>_deletion_insertion_curves.png` | deletion and insertion step curves per case |

---

## Controlled Scenario Expectations

See `cases/controlled_cases.yaml` for the full spec. The five scenarios test:

| Scenario | Pattern | Key check |
|---|---|---|
| Marie Curie Nobel | complementary evidence | top 2 chunks ≠ same; positive interaction |
| Beijing Olympics | direct + distractors | direct answer chunk ranked first |
| Eiffel Tower (unsupported) | missing evidence | full context support < 0.5 |
| Australia capital | conflicting context | correct chunk outranks distractors |
| Atlantic Ocean | redundant evidence | negative pairwise interaction |

---

## Adding Your Own Cases

1. Copy `cases/pdf_cases.example.json` → `local/my_cases.json`
2. Edit `pdf_path`, `question`, `expected_evidence_terms`
3. Run `make eval-pdf PDF_CASES=local/my_cases.json`
4. Run `make compare-chunking PDF_CASES=local/my_cases.json` to compare configs

Keep PDF files outside git unless they are small, redistributable fixtures.
The `data/` directory is gitignored for large files.
