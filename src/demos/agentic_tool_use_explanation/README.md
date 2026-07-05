# Explaining Tool Selection Demo

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

The embedding segmenter needs no additional setup. For full linguistic
segmentation, install the English spaCy model locally:

```bash
python -m spacy download en_core_web_sm
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

1. Select either an example request or a custom request.
2. Run inference to select the target tool. If no inference result is available,
   the selected explanation scorer is used as a fallback target selector.
3. Pick a scoring method. The default is the Mock model scorer; keyword
   comparison is hidden unless selected or enabled in more options.
4. Run the explanation.
5. Inspect the Summary, Attribution, and Interactions tabs.
6. Use advanced/debug controls only when you need prompt segments,
   value-function details, scoring prompt previews, or keyword comparison.

The UI currently exposes three scoring methods:

1. Mock model scorer for lightweight deterministic demo behavior.
2. Keyword baseline for a transparent lexical baseline.
3. Calibrated multiclass tool log-odds (HF local) for local Hugging Face
   likelihood-based scoring.

## What The App Shows

- An example-request mode with fixed system prompt segments and user request
  segments.
- A custom-request mode with a user-input box, suggested tool, short
  reason, and per-tool scores.
- Tool selection: `weather_tool`, `calculator_tool`, `web_search_tool`, or `no_tool`.
- Optional calibrated multiclass tool log-odds (HF local) scorer backend,
  loaded lazily only after it is selected and the explanation is run.
- A compact setup panel for the coalition value function.
- Summary metrics for target-tool support after running the explanation.
- First-order attribution ranking: which segment most pushes the tool decision.
- Segment interaction heatmap: which system/user segments interact under the selected shapiq index.
- System/user block outlines on the heatmap, so the hierarchy of prompt parts is visible at a glance.
- Advanced/debug sections for prompt segments, scoring prompts, value-function
  details, and keyword comparison.

## Segmentation

The current demo uses manually curated prompt segments. This makes the demo
stable and easy to explain, because each player has a clear meaning. For a final
model-backed version, the segmentation can be replaced or extended with:

- sentence splitting,
- word/token splitting,
- message-role splitting (`system`, `user`, tool schema),
- tool-schema splitting by tool description and parameter descriptions.

## Scoring Methods

The demo keeps several scoring implementations side by side, though the UI only
exposes the presentation-ready methods listed above:

- `LexicalToolScorer` is a fast keyword baseline.
- `LLMToolScorer` is a generation-based numeric judge. It remains in code for
  experiments, but is hidden from the Streamlit scoring-method dropdown.
- `CalibratedToolLogOddsScorer` avoids numeric parsing by scoring a fixed
  routing-label decision code (`A`/`B`/`C`/`D`) for every candidate tool with
  model likelihood, calibrating each candidate against a content-free probe
  prompt, and combining the results into a target-vs-all multiclass log-odds
  value normalized against the true empty coalition.

The selected scorer first evaluates the full fixed context and full user request
for every candidate decision. The highest-scoring candidate becomes the tool to
explain, and the same scorer then evaluates masked user-segment coalitions:

```text
full fixed context + full user request
-> selected scorer evaluates all candidate tools
-> highest-scoring candidate becomes the explanation target
-> the same scorer evaluates masked coalitions
```

The setup preview does not execute any tools. It only selects which tool the
agent should use for the current full prompt. The preview returns:

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

## Optional Calibrated Multiclass Tool Log-Odds Scorer (HF Local)

The default scoring method is the **Mock model scorer** so the demo stays fast and
usable on a clean local machine. The optional **Calibrated multiclass tool
log-odds (HF local)** scorer (`CalibratedToolLogOddsScorer`) uses a local
Hugging Face causal language model.

For each coalition, the LLM is queried with a fixed constrained-classification
routing prompt (see `ROUTING_LABELS` and `build_routing_classification_prompt`
in `scorers.py`): choose exactly one routing decision code (`A`/`B`/`C`/`D` for
`weather_tool`/`calculator_tool`/`web_search_tool`/`no_tool`), each rendered
with its canonical description from `tool_schemas.py`/`TOOLS` (not a second
hard-coded description dictionary) -- so editing a tool's description changes
this prompt, and changes the calibration cache key. Every candidate's
decision-code continuation is scored with model log-likelihood, calibrated
against a content-free probe prompt (`CALIBRATION_USER_REQUEST =
"[NO USER REQUEST]"`) to remove any fixed per-label prior, and combined into a
target-vs-all multiclass log-odds. The result is normalized against the true
empty coalition so that `V(empty_coalition) == 0`. shapiq then uses these
coalition values to compute segment attributions and pairwise interactions.

Unlike a naive target-vs-`no_tool` contrast, `no_tool` is treated as one
routing alternative among all candidates, not a fixed reference -- so a
candidate outscoring `no_tool` is never mistaken for genuine support when a
third candidate is actually strongest. The same canonical routing-classification
prompt builder (`build_routing_classification_prompt`) is shared by HF local
classification routing, HF local logprob scoring, calibration scoring, and
coalition scoring, so token boundaries and any residual template prior stay
identical across all four.

**The target tool being explained is always produced by this same protocol.**
`hf_router.LocalHFClassificationRouter` wraps an already-loaded
`CalibratedToolLogOddsScorer` and selects a tool by scoring every routing-label
continuation for the full (unmasked) request and taking the argmax --
`scorer.score_full_request_labels(...)`, the same method the coalition value
function itself calls. When this scorer mode is selected, the Streamlit UI
ignores the Inference tab's selected tool (from Groq/Gemini/the structured-JSON
`LocalHFRouter`) and instead redetermines the target from this classifier's own
argmax at run time, so the tool being explained and the classifier producing
the coalition values are never different models. This is intentionally
separate from `LocalHFRouter` in `hf_router.py`, which still uses its own
structured JSON protocol to run the real Inference tab agent end to end
(selecting a tool and extracting call arguments such as location/date) -- that
router is for execution, not for what this scorer mode explains.

Run the app with:

```bash
uv run streamlit run src/demos/agentic_tool_use_explanation/app.py
```

Suggested first model:

```text
TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

Suggested first logprob model:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

## Optional Groq Soft-Vote Scorer

The **Groq soft-vote scorer** is a black-box value function for API-only
routers. For each coalition prompt, it samples the Groq router `N` times and
returns the empirical frequency with which the target tool is selected. This is
not calibrated; it is a behavioral soft-vote score under fixed decoding
settings.

Gemma model ids can be tried manually if the machine has enough memory and the
required Hugging Face access.

Colab-style scorer wiring:

```python
from demos.agentic_tool_use_explanation.scorers import CalibratedToolLogOddsScorer
from demos.agentic_tool_use_explanation.tool_game import ToolUseGame

scorer = CalibratedToolLogOddsScorer(model_id="Qwen/Qwen2.5-1.5B-Instruct")
game = ToolUseGame(
    target_tool="weather_tool",
    segments=segments,
    scorer=scorer,
    tool_descriptions=tool_descriptions,
)
```

For the final project demo, pass a richer model-backed scorer into `ToolUseGame`
such as:

- target tool-name log-likelihood from a tool-calling LLM,
- soft-vote score from sampled tool-router decisions,
- structured tool-call selection frequency from an agent trace,
- calibrated target-vs-all multiclass log-odds across every routing decision
  (not a fixed contrast against `no_tool` alone).

## Demo Story

Use scenarios such as weather, calculator, web search, no-tool, and ambiguous
recency requests to show:

1. Which user phrases trigger tool use.
2. Which system rules push or suppress tool use.
3. Which system-rule/user-phrase interactions explain the final decision.
