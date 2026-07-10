# shapiq-text-llm-explanations
Text imputers and Shapley interaction demos for explaining language models with shapiq.

## useful commands
- `uv run pre-commit run --all-files` : Run pre-commits locally.
- `uv run pytest "tests/shapiq"` : Run all tests.
- `uv run pytest tests/shapiq/tests_unit/tests_imputer/test_text_imputer.py` : Run just the text imputer tests.
- `uv run gradio src/demos/SentimentAnalysis/app.py` : Run the sentiment analysis demo.
- `uv sync`: Sync the project dependencies, it is needed before running the demos from other teammates.
- `uv run streamlit run src/demos/JailbreakAnalysis/app.py`: run the jailbreak analysis demo.
- `python aggregate_results.py --results-dir vulnerability_results --out allResults.json` to aggregate results

## Adding Dependencies
Implementing each demo might require adding new dependencies. It would be good practice to run `uv add <dependencies>` locally, and pushes `pyproject.toml` and `uv.lock` to repo. Other team members run `uv sync` to update dependencies.
