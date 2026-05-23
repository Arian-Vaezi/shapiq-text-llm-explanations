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
uv pip install -r demos/agentic_tool_use_explanation/requirements.txt
uv run streamlit run demos/agentic_tool_use_explanation/app.py
```

## What The App Shows

- Editable system prompt segments and user request segments.
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

The current scorer is lightweight lexical scaffolding. It does not call a real
LLM. Its purpose is to validate the game, shapiq computation, and visualization
pipeline without API keys or GPU dependencies.

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
