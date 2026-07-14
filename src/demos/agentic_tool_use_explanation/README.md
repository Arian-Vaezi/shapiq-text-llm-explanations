# Agentic Tool-Use Explanation

This demo explains which parts of a user request support or oppose an agent's
tool selection. It combines native Hugging Face tool-calling, request
segmentation, and Shapley attribution in an interactive Streamlit application.

The supported outcomes are:

* `weather_tool`
* `calculator_tool`
* `web_search_tool`
* direct answer (`no_tool`)

## Start here

Run the Streamlit application and select one of the provided example requests.

The app first performs a full-context agent run. The resulting tool or direct
answer is then frozen as the explanation target. Shapley values and pairwise
k-SII interactions show how the user-request segments support or oppose that
target.

The explanation is post-hoc: it characterizes observable model behavior under
coalition masking, but does not recover the model's internal routing mechanism.

## Project structure

```text
agentic_tool_use_explanation/
├── app.py                        Streamlit entry point
├── _app_impl/                    Application flow and UI
│   └── plotting.py               Attribution and interaction plots
├── hf_router.py                  Native Hugging Face inference
├── scorers.py                    Coalition value functions
├── semantic_segmenter.py         Embedding-based user-request segmentation
├── linguistic_segmenter.py       spaCy-based user-request segmentation
├── tool_game.py                  shapiq cooperative game
├── tool_schemas.py               Tool definitions
├── run_holdout_eval.py           Offline evaluation
└── README.md
```

The main data flow is:

```text
full request
    → native agent inference
    → frozen explanation target
    → request segmentation
    → coalition scoring
    → Shapley values and k-SII interactions
    → bar plot and interaction heatmap
```

## Explanation scope

Only user-request segments are treated as players.

The following remain fixed across all coalitions:

* system prompt;
* available tool schemas;
* model and tokenizer;
* chat template;
* frozen target continuation.

Segments outside a coalition are removed while the original order of the
remaining segments is preserved.

For a selected tool (t), the local Hugging Face value function scores a
canonical native tool-identity continuation. For each coalition, the scorer
constructs a template-derived canonical native-format continuation for tool
(t), built with the same selected tokenizer and native chat template used by
inference. This continuation is truncated at the tool identity and excludes
free-form argument tokens, isolating tool-identity evidence from
argument-generation variability:

```text
h_t(S) = mean log P_theta(y_t | x_S)
v_t(S) = h_t(S) - h_t(empty)
```

For direct-answer cases, the same mechanism is applied to a frozen answer
fragment.

The resulting explanation targets which user-request segments increase or
decrease support for the selected tool identity, not argument generation or
raw response reproduction. It does not directly compare direct answering
against every available tool.

## HF Local Model Consistency

The implementation supports multiple HF Local models. Inference and XAI are
required to use the same model, tokenizer, chat template, device, and runtime
configuration. Developer diagnostics report both inference and scorer identity
so stale or cross-model explanations are blocked instead of silently displayed.

## Setup

Run commands from the repository root:

```bash
uv sync
```

For linguistic segmentation, install the English spaCy model:

```bash
uv run python -m spacy download en_core_web_sm
```

Hugging Face models are downloaded on first use and cached locally.

The default local model is:

```text
Qwen/Qwen2.5-3B-Instruct
```

A GPU is recommended for coalition evaluation.

## Run the application

```bash
uv run streamlit run src/demos/agentic_tool_use_explanation/app.py
```

Open the URL printed by Streamlit, normally:

```text
http://localhost:8501
```

The standard workflow is:

1. Select or enter a request.
2. Run the full-context agent.
3. Inspect the selected tool or direct answer.
4. Inspect the request segmentation.
5. Run the explanation.
6. Review the Shapley attribution bar plot and pairwise interaction heatmap.

## Interpreting the results

The attribution bar plot shows the individual contribution of each request
segment:

* positive values support the frozen target;
* negative values oppose the frozen target.

The interaction heatmap shows main and pairwise k-SII effects:

* diagonal cells contain individual main effects;
* positive off-diagonal values indicate reinforcement;
* negative off-diagonal values indicate redundancy or suppression.

Attributions depend on the selected model, prompt template, segmentation, and
masking strategy.

## Run the evaluation

The offline evaluation script runs representative and holdout requests without
the Streamlit interface:

```bash
uv run python src/demos/agentic_tool_use_explanation/run_holdout_eval.py
```

Stored results are model- and configuration-specific and should not be treated
as universal routing benchmarks.

## Verify the demo

Run the relevant tests:

```bash
uv run pytest tests -k "agentic or tool_use"
```

Run code-quality checks:

```bash
uv run pre-commit run --all-files
```

## Limitations

* Coalition masking can produce requests that differ from naturally occurring
  inputs.
* Results depend on how the request is segmented into players.
* The explanation is conditional on the target selected from the full request.
* Exact Shapley computation scales exponentially with the number of players.
* Direct-answer and tool-identity continuation scores are not automatically
  calibrated to the same numerical scale.
* The method explains model support, not the model's hidden causal mechanism.
