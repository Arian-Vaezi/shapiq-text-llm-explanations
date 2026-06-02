.PHONY: app test eval pdf-smoke lint precommit help

APP := demos/rag_retrieval_explanation/app.py
PDF_CASES ?= demos/rag_retrieval_explanation/evals/pdf_cases.example.json

help:
	@echo "Common commands:"
	@echo "  make app                         Run the Streamlit RAG explanation demo"
	@echo "  make test                        Run the generic retrieval policy tests"
	@echo "  make eval                        Run controlled Shapley explanation evals"
	@echo "  make pdf-smoke PDF_CASES=...     Run PDF retrieval smoke evals from a JSON case file"
	@echo "  make lint                        Run pre-commit on all files"

app:
	uv run streamlit run $(APP)

test:
	uv run pytest tests/test_rag_retrieval_policy.py

eval:
	uv run python demos/rag_retrieval_explanation/evals/run_controlled_eval.py

pdf-smoke:
	uv run python demos/rag_retrieval_explanation/evals/run_pdf_smoke_eval.py $(PDF_CASES)

lint:
	uv run pre-commit run --all-files

precommit: lint
