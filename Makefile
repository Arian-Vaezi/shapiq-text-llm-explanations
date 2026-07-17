.PHONY: app api frontend frontend-build test eval-controlled eval-controlled-model eval-controlled-bge-lexical eval-interactions-controlled eval-report-html eval-stability eval-qasper-smoke eval-qasper verify-report-results setup-rag-local install-rag-local download-rag-model lint precommit help

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RAG_FRONTEND    := demos/rag_retrieval_explanation/frontend
RAG_MODEL_DIR   := models/llm
RAG_MODEL       := $(RAG_MODEL_DIR)/Qwen3-8B-Q4_K_M.gguf
RAG_MODEL_URL   := https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf
RAG_EVALS       := demos/rag_retrieval_explanation/evals
RAG_RESULTS     := $(RAG_EVALS)/results
RAG_RUNS        := $(RAG_EVALS)/runs

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

help:
	@echo ""
	@echo "RAG Retrieval Explanation Demo"
	@echo "=============================="
	@echo ""
	@echo "App:"
	@echo "  make app                             Build and run the local web dashboard"
	@echo "  make api                             Run the FastAPI backend"
	@echo "  make frontend                        Run the Vite development frontend"
	@echo "  make frontend-build                  Build the React dashboard"
	@echo "  make setup-rag-local                 Install llama.cpp and download the GGUF model"
	@echo "  make install-rag-local               Install local llama.cpp demo dependencies"
	@echo "  make download-rag-model              Download the default local GGUF model"
	@echo ""
	@echo "Tests:"
	@echo "  make test                            Run the complete RAG demo test suite"
	@echo ""
	@echo "Evaluation:"
	@echo "  make eval-controlled                 Run corpus retrieval + gold attribution eval"
	@echo "  make eval-controlled-model           Add model knowledge-source experiments"
	@echo "  make eval-controlled-bge-lexical     Run controlled BGE + lexical eval"
	@echo "  make eval-interactions-controlled    Rebuild interaction results from published runs"
	@echo "  make eval-stability                  Rebuild stability results from published runs"
	@echo "  make eval-qasper-smoke               Run the first five frozen QASPER cases"
	@echo "  make eval-qasper                     Run the frozen 50-case QASPER eval"
	@echo ""
	@echo "Report:"
	@echo "  make eval-report-html                Rebuild the static HTML report"
	@echo "  make verify-report-results           Verify reported metrics from published results"
	@echo ""
	@echo "Code quality:"
	@echo "  make lint                            Run pre-commit on all files"
	@echo ""

app:
	npm --prefix $(RAG_FRONTEND) run build
	uv run uvicorn demos.rag_retrieval_explanation.backend.api:app --host 127.0.0.1 --port 8000

api:
	uv run uvicorn demos.rag_retrieval_explanation.backend.api:app --reload --host 127.0.0.1 --port 8000

frontend:
	npm --prefix $(RAG_FRONTEND) run dev

frontend-build:
	npm --prefix $(RAG_FRONTEND) run build

setup-rag-local: install-rag-local download-rag-model

install-rag-local:
	@if [ "$$(uname -s)-$$(uname -m)" = "Darwin-arm64" ]; then \
		CMAKE_ARGS="-DGGML_METAL=on" uv sync --group rag_demo; \
	else \
		uv sync --group rag_demo; \
	fi
	npm --prefix $(RAG_FRONTEND) install

download-rag-model:
	@mkdir -p $(RAG_MODEL_DIR)
	@if [ -f "$(RAG_MODEL)" ]; then \
		echo "Model already exists: $(RAG_MODEL)"; \
	else \
		curl -L --fail --progress-bar "$(RAG_MODEL_URL)" -o "$(RAG_MODEL)"; \
	fi

test:
	uv run pytest demos/rag_retrieval_explanation/tests

eval-controlled:
	uv run python -m demos.rag_retrieval_explanation.evals.experiments.run_controlled_eval

eval-controlled-model:
	uv run python -m demos.rag_retrieval_explanation.evals.experiments.run_controlled_eval --model-path $(RAG_MODEL)

eval-controlled-bge-lexical:
	uv run python -m demos.rag_retrieval_explanation.evals.experiments.run_controlled_eval --method "Dense embeddings" --embedding-model models/embedding/bge-base-en-v1.5 --embedding-device mps --value-function lexical --output-dir $(RAG_RUNS)/controlled_bge_lexical

eval-interactions-controlled:
	uv run python -m demos.rag_retrieval_explanation.evals.reporting.analyze_interactions \
		--run "BGE-base + lexical=$(RAG_RESULTS)/controlled/bge_lexical" \
		--run "BGE-base + target LL=$(RAG_RESULTS)/controlled/bge_target_likelihood" \
		--run "BGE-base + contrastive LL=$(RAG_RESULTS)/controlled/bge_contrastive_likelihood" \
		--run "TF-IDF + contrastive LL=$(RAG_RESULTS)/controlled/tfidf_contrastive_likelihood" \
		--output-dir $(RAG_RESULTS)/derived/interactions

eval-report-html:
	uv run python -m demos.rag_retrieval_explanation.evals.reporting.build_eval_report

eval-stability:
	uv run python -m demos.rag_retrieval_explanation.evals.reporting.analyze_stability

eval-qasper-smoke:
	uv run python -m demos.rag_retrieval_explanation.evals.experiments.run_qasper_eval --limit 5

eval-qasper:
	uv run python -m demos.rag_retrieval_explanation.evals.experiments.run_qasper_eval

verify-report-results:
	uv run python -m demos.rag_retrieval_explanation.evals.reporting.verify_report_results

# --- Code quality ---

lint:
	uv run pre-commit run --all-files

precommit: lint
