# Agentic Tool-Use Explanation Demo

Streamlit framework for the **Agentic tool-use explanations** demo.

This demo treats system prompt rules and user-request phrases as players in a
cooperative game. For each coalition of prompt segments, the app scores how
strongly the visible text supports calling a target tool, then uses shapiq to
estimate first-order attributions and second-order interactions.

The current interface is intentionally compact for development-stage demos:
scenario/input first, a short router summary, a minimal explanation setup, and
results only after running the explanation.

The demo intentionally reuses package visualization functions implemented for
sentence/text players:

- `token_attribution_bar_plot`
- `sentence_interaction_heatmap`

## Run

From the repository root:

```bash
uv run streamlit run src/demos/agentic_tool_use_explanation/app.py
```

Then open:

```text
http://localhost:8501
```

`localhost` only works on the machine running Streamlit. Other users need to run
the command on their own machine, or use the displayed Network URL if they are on
the same local network.

If port `8501` is already in use, change `--server.port` to another port such as
`8502` and open that local URL instead.

If your environment has unwritable Matplotlib/fontconfig cache directories, set
`MPLCONFIGDIR` and `XDG_CACHE_HOME` to writable paths before starting Streamlit.

### Minimal Local Environment

This branch has been tested with a local `.venv` using Python 3.12. To recreate
the minimal environment needed for the current Streamlit demo:

```bash
/Users/yililalo/.local/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install streamlit pandas matplotlib scipy tqdm scikit-learn sparse-transform galois colour networkx
```

This does not install a real LLM backend. Do not install `torch` or
`transformers` unless you are working on the later Gemma integration.

## Current UI Flow

1. Select either a curated sample scenario or the local Mock LLM Router input.
2. Choose the target tool to explain.
3. Pick a scorer backend. The default is the Mock LLM judge; lexical comparison
   is hidden unless selected or enabled in advanced settings.
4. Run the explanation.
5. Inspect the Summary, Attribution, and Interactions tabs.
6. Use advanced/debug controls only when you need prompt segments,
   value-function details, scoring prompt previews, or lexical comparison.

This demo currently uses lexical/mock scoring scaffolding. A real local
HuggingFace/Gemma scorer is the next integration step.

## What The App Shows

- A sample-scenario mode with fixed system prompt segments and user request
  segments.
- A **Mock LLM Router** mode with a user-input box, recommended tool, short
  reason, and per-tool scores.
- Target tool selection: `weather_tool`, `calculator_tool`, `web_search_tool`, or `no_tool`.
- A compact setup panel for the coalition value function.
- Summary metrics for target-tool support after running the explanation.
- First-order attribution ranking: which segment most pushes the tool decision.
- Segment interaction heatmap: which system/user segments interact under the selected shapiq index.
- System/user block outlines on the heatmap, so the hierarchy of prompt parts is visible at a glance.
- Advanced/debug sections for prompt segments, scoring prompts, value-function
  details, and lexical comparison.

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
