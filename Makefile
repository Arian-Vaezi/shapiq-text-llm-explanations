.PHONY: app api frontend frontend-build test eval-controlled eval-controlled-model eval-controlled-bge-lexical eval-interactions-controlled eval-report-html eval-stability eval-qasper-smoke eval-qasper verify-report-results paper setup-rag-local install-rag-local download-rag-model lint precommit help

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RAG_FRONTEND    := demos/rag_retrieval_explanation/frontend
RAG_MODEL_DIR   := models/llm
RAG_MODEL       := $(RAG_MODEL_DIR)/Qwen3-8B-Q4_K_M.gguf
RAG_MODEL_URL   := https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf
RAG_E4B_MODEL   := $(RAG_MODEL_DIR)/gemma-4-E4B-it-Q4_K_M.gguf
RAG_CONTROLLED_BGE_LEXICAL_RUN := demos/rag_retrieval_explanation/evals/runs/20260616_controlled_interactions_bge_lexical
RAG_CONTROLLED_INTERACTION_SUMMARY := demos/rag_retrieval_explanation/evals/runs/20260616_controlled_interaction_summary

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
	@echo "  make test                            Run retrieval policy unit tests"
	@echo "  make eval-controlled                 Run corpus retrieval + gold attribution eval"
	@echo "  make eval-controlled-model           Add model knowledge-source experiments"
	@echo "  make eval-controlled-bge-lexical     Run controlled BGE + lexical eval"
	@echo "  make eval-interactions-controlled    Recompute controlled RQ2 interaction table"
	@echo "  make eval-report-html                Rebuild static HTML report details"
	@echo "  make eval-stability                  Analyze controlled attribution stability"
	@echo "  make eval-qasper-smoke               Run the first five frozen QASPER cases"
	@echo "  make eval-qasper                     Run the frozen 30-case QASPER eval"
	@echo "  make verify-report-results           Recompute paper tables from saved runs"
	@echo "  make paper                           Build the living LaTeX manuscript"
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
	uv run pytest tests/test_rag_retrieval_policy.py tests/test_rag_llama_cpp_backend.py tests/test_rag_web_backend.py tests/test_rag_controlled_eval.py tests/test_rag_qasper_eval.py tests/test_rag_stability_analysis.py tests/test_rag_evaluation_protocol.py

eval-controlled:
	uv run python -m demos.rag_retrieval_explanation.evals.run_controlled_eval

eval-controlled-model:
	uv run python -m demos.rag_retrieval_explanation.evals.run_controlled_eval --model-path $(RAG_MODEL)

eval-controlled-bge-lexical:
	uv run python -m demos.rag_retrieval_explanation.evals.run_controlled_eval --method "Dense embeddings" --embedding-model models/embedding/bge-base-en-v1.5 --embedding-device mps --value-function lexical --output-dir $(RAG_CONTROLLED_BGE_LEXICAL_RUN)

eval-interactions-controlled:
	uv run python -m demos.rag_retrieval_explanation.evals.analyze_interactions --run "BGE-base + lexical=$(RAG_CONTROLLED_BGE_LEXICAL_RUN)" --output-dir $(RAG_CONTROLLED_INTERACTION_SUMMARY)

eval-report-html:
	uv run python -m demos.rag_retrieval_explanation.evals.build_eval_report

eval-stability:
	uv run python -m demos.rag_retrieval_explanation.evals.analyze_stability

eval-qasper-smoke:
	uv run python -m demos.rag_retrieval_explanation.evals.run_qasper_eval --limit 5

eval-qasper:
	uv run python -m demos.rag_retrieval_explanation.evals.run_qasper_eval

verify-report-results:
	uv run python -m demos.rag_retrieval_explanation.evals.verify_report_results

paper:
	TEXMFVAR=/tmp/shapiq-texmf-var latexmk -pdf -interaction=nonstopmode -halt-on-error -cd paper/main.tex

# --- Code quality ---

lint:
	uv run pre-commit run --all-files

precommit: lint
