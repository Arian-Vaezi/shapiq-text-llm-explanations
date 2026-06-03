.PHONY: app test eval eval-controlled eval-pdf pdf-cases-template \
        compare-chunking compare-embeddings compare-full \
        lint precommit help

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

APP             := demos/rag_retrieval_explanation/app.py
DEMO_DIR        := demos/rag_retrieval_explanation
EVALS_DIR       := $(DEMO_DIR)/evals
RUNNERS_DIR     := $(EVALS_DIR)/runners
CONFIGS_DIR     := $(EVALS_DIR)/configs
CASES_DIR       := $(EVALS_DIR)/cases

PDF_CASES       ?= $(CASES_DIR)/pdf_cases.example.json
PDF_CASE_TEMPLATE := $(CASES_DIR)/pdf_cases.example.json

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

help:
	@echo ""
	@echo "RAG Retrieval Explanation Demo"
	@echo "=============================="
	@echo ""
	@echo "App:"
	@echo "  make app                             Run the Streamlit demo"
	@echo ""
	@echo "Tests:"
	@echo "  make test                            Run retrieval policy unit tests"
	@echo ""
	@echo "Evals (controlled, no downloads):"
	@echo "  make eval                            Alias for eval-controlled"
	@echo "  make eval-controlled                 Controlled Shapley attribution evals"
	@echo ""
	@echo "Evals (PDF, requires a case file):"
	@echo "  make eval-pdf PDF_CASES=...          PDF retrieval smoke eval"
	@echo "  make pdf-cases-template PDF_CASES=local/my_cases.json"
	@echo ""
	@echo "Grid comparisons (PDF, may download models):"
	@echo "  make compare-chunking PDF_CASES=...  3 chunk sizes × TF-IDF"
	@echo "  make compare-embeddings PDF_CASES=.. TF-IDF + MiniLM + BGE"
	@echo "  make compare-full PDF_CASES=...      Full 18-config research grid"
	@echo ""
	@echo "Code quality:"
	@echo "  make lint                            Run pre-commit on all files"
	@echo ""

app:
	uv run streamlit run $(APP)

test:
	uv run pytest tests/test_rag_retrieval_policy.py

# --- Controlled evals (no PDF, no model downloads) ---

eval: eval-controlled

eval-controlled:
	uv run python $(RUNNERS_DIR)/run_controlled_eval.py

# --- PDF smoke evals ---

eval-pdf:
	uv run python $(RUNNERS_DIR)/run_pdf_smoke_eval.py $(PDF_CASES)

pdf-cases-template:
	@if [ "$(PDF_CASES)" = "$(PDF_CASE_TEMPLATE)" ]; then \
		echo "Choose a writable case path, for example:"; \
		echo "  make pdf-cases-template PDF_CASES=local/pdf_cases.json"; \
		exit 1; \
	fi
	@mkdir -p "$$(dirname "$(PDF_CASES)")"
	cp $(PDF_CASE_TEMPLATE) $(PDF_CASES)
	@echo "Created $(PDF_CASES). Edit pdf_path, question, and expected_evidence_terms."

# --- Grid comparison evals ---

compare-chunking:
	uv run python $(RUNNERS_DIR)/run_grid_eval.py \
		--config $(CONFIGS_DIR)/chunking_compare.yaml \
		--cases $(PDF_CASES) \
		--allow-failures

compare-embeddings:
	uv run python $(RUNNERS_DIR)/run_grid_eval.py \
		--config $(CONFIGS_DIR)/embedding_compare.yaml \
		--cases $(PDF_CASES) \
		--allow-failures

compare-full:
	uv run python $(RUNNERS_DIR)/run_grid_eval.py \
		--config $(CONFIGS_DIR)/full_grid.yaml \
		--cases $(PDF_CASES) \
		--allow-failures

# --- Backward-compat aliases (old Makefile targets) ---

pdf-smoke: eval-pdf

pdf-compare:
	uv run python $(RUNNERS_DIR)/run_grid_eval.py \
		--config $(CONFIGS_DIR)/chunking_compare.yaml \
		--cases $(PDF_CASES) \
		--allow-failures

embedding-compare: compare-embeddings

# --- Code quality ---

lint:
	uv run pre-commit run --all-files

precommit: lint
