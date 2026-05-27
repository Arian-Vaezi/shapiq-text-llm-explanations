# Agentic Tool-Use Explanation Demo

Streamlit framework for the **Agentic tool-use explanations** demo.

This demo treats system prompt rules and user-request phrases as players in a
cooperative game. For each coalition of prompt segments, the app scores how
strongly the visible text supports calling a target tool, then uses shapiq to
estimate first-order attributions and second-order interactions.

The interface follows the same structure as the RAG retrieval demo: scenario
panel, metric cards, input trace, game setup, verdict, interpretation notes, and
result tabs.

The demo intentionally reuses package visualization functions implemented for
sentence/text players:

- `token_attribution_bar_plot`
- `sentence_interaction_heatmap`

## Run

From the repository root:

```bash
MPLCONFIGDIR=.cache/matplotlib XDG_CACHE_HOME=.cache PYTHONPATH=src \
  .venv/bin/streamlit run src/demos/agentic_tool_use_explanation/app.py \
  --server.port 8501 \
  --server.headless true \
  --browser.gatherUsageStats false
```

Then open:

```text
http://localhost:8501
```

`localhost` only works on the machine running Streamlit. Other users need to run
the command on their own machine, or use the displayed Network URL if they are on
the same local network.

The `MPLCONFIGDIR` and `XDG_CACHE_HOME` settings keep Matplotlib/fontconfig
caches inside the project. This avoids slow or failing startup on macOS when the
default user cache directories are not writable.

If port `8501` is already in use, change `--server.port` to another port such as
`8502` and open that local URL instead.

### Minimal Local Environment

This branch has been tested with a local `.venv` using Python 3.12. To recreate
the minimal environment needed for the current Streamlit demo:

```bash
/Users/yililalo/.local/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install streamlit pandas matplotlib scipy tqdm scikit-learn sparse-transform galois colour networkx
```

This does not install a real LLM backend. Do not install `torch` or
`transformers` unless you are working on the later Gemma integration.

## What The App Shows

- A default **Mock LLM Router** mode with a user-input box.
- A mock assistant response that recommends one of the available tools and shows
  a short reason plus per-tool scores.
- A sample-scenario mode with fixed system prompt segments and user request
  segments.
- Target tool selection: `weather_tool`, `calculator_tool`, `web_search_tool`, or `no_tool`.
- Metric cards for number of prompt segments, empty-prompt score, and full-prompt score.
- First-order attribution ranking: which segment most pushes the tool decision.
- Segment interaction heatmap: which system/user segments interact under the selected shapiq index.
- System/user block outlines on the heatmap, so the hierarchy of prompt parts is visible at a glance.
- Coalition audit table: exact target-tool scores for small segment combinations.

## Segmentation

The current demo uses manually curated prompt segments. This makes the demo
stable and easy to explain, because each player has a clear meaning. For a final
model-backed version, the segmentation can be replaced or extended with:

- sentence splitting,
- word/token splitting,
- message-role splitting (`system`, `user`, tool schema),
- tool-schema splitting by tool description and parameter descriptions.

## Scoring Mode

The current scorer is lightweight lexical scaffolding. The Mock LLM Router does
not call a real LLM and does not execute any tools. It only selects which tool
the agent should use for the user request.

The router returns the handoff shape expected by a future local Gemma backend:

```python
{
    "tool": "weather_tool",
    "reason": "Matched rain, berlin, tomorrow; the question asks about weather.",
    "scores": {
        "weather_tool": 0.82,
        "calculator_tool": 0.05,
        "web_search_tool": 0.10,
        "no_tool": 0.03,
    },
}
```

This lets the UI and shapiq explanation flow be developed without API keys, GPU
dependencies, or Hugging Face model downloads.

For the final project demo, replace `ToolUseGame.score_segments` in `tool_game.py`
with a model-backed scorer such as:

- target tool-name log-likelihood from a tool-calling LLM,
- probability from a small tool-router classifier,
- structured tool-call probability from an agent trace,
- contrastive log-odds between the target tool and `no_tool`.

## Demo Story

Use scenarios such as weather, calculator, web search, no-tool, and ambiguous
recency requests to show:

1. Which user phrases trigger tool use.
2. Which system rules push or suppress tool use.
3. Which system-rule/user-phrase interactions explain the final decision.
