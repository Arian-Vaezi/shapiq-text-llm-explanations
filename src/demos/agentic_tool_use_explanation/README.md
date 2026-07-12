# Explaining Tool Selection Demo

Streamlit framework for the **Agentic tool-use explanations** demo.

This demo treats user-request phrases as players in a cooperative game while
keeping the system prompt and tool schemas fixed. For each coalition of user
segments, the app scores how strongly the visible text supports the identity of
the tool selected by the full-context agent run, then uses shapiq to estimate
first-order attributions and second-order interactions.

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

1. Select HF Local or API Agent mode and enter a request.
2. Run the full pipeline. The full-context agent run selects the target tool.
3. The XAI stage freezes that Agent Result tool as the explanation target.
4. Inspect the summary, SV attribution, and k-SII interaction results.
5. Use Developer settings only when you need alternate scorers, prompt previews,
   value-function details, or raw backend diagnostics.

The presentation path uses a mixed-fidelity value-function architecture:

1. Executable tool calls use a canonical native tool-identity continuation
   likelihood value function.
2. Direct-answer (`no_tool`) cases use the legacy A/B/C/D forced-choice
   surrogate, explicitly labeled as a surrogate rather than native evidence.

## What The App Shows

- An example-request mode with fixed system prompt segments and user request
  segments.
- A custom-request mode with a user-input box, suggested tool, short
  reason, and per-tool scores.
- Tool selection: `weather_tool`, `calculator_tool`, `web_search_tool`, or
  the internal direct-answer decision `no_tool`.
- HF Local native tool-identity scoring with inference/XAI model consistency
  checks.
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

The demo keeps several scoring implementations side by side. The default
HF-local presentation path is the native tool-identity scorer:

- Full-context native inference determines the selected tool `t`.
- The selected tool is frozen as the XAI target.
- For each coalition, the scorer constructs `y_t`, a template-derived
  canonical native-format continuation for tool `t`, with the same selected
  tokenizer and native chat template used by inference.
- `y_t` is truncated at the tool identity and excludes free-form argument
  tokens. This isolates tool-identity evidence from argument-generation
  variability.
- The explanation targets which user-request segments increase or decrease
  support for the selected tool identity, not argument generation or raw
  response reproduction.

The native tool-identity value function is:

```text
h_t(S) = mean_k log P_theta(y_{t,k} | x_S, y_{t,<k})
v_t(S) = h_t(S) - h_t(empty)
```

Other implementations remain available for development and comparison:

- `LexicalToolScorer` is a fast keyword baseline.
- `LLMToolScorer` is a generation-based numeric judge. It remains in code for
  experiments, but is hidden from the Streamlit scoring-method dropdown.
- `CalibratedToolLogOddsScorer` avoids numeric parsing by scoring a fixed
  routing-label decision code (`A`/`B`/`C`/`D`) for every candidate tool with
  model likelihood, calibrating each candidate against a content-free probe
  prompt, and combining the results into a target-vs-all multiclass log-odds
  value normalized against the true empty coalition.

The primary pipeline freezes the full-context Agent Result before evaluating
coalitions:

```text
full fixed context + full user request
-> native inference selects a tool
-> selected tool identity becomes the explanation target
-> value function evaluates masked user-segment coalitions
```

Developer-only fallback target selection can still ask a selected scorer to
choose a target when no Agent Result exists, but that is not the presentation
path.

This lets the UI and shapiq explanation flow be developed without API keys, GPU
dependencies, or Hugging Face model downloads.

## Legacy A/B/C/D Forced-Choice Scorer (HF Local)

The legacy **Calibrated multiclass tool log-odds (HF local)** scorer
(`CalibratedToolLogOddsScorer`) uses a local Hugging Face causal language model.
It is retained for developer comparison and for the direct-answer branch as a
clearly labeled surrogate.

For each coalition, the LLM is queried with a fixed artificial
constrained-classification routing prompt (see `ROUTING_LABELS` and
`build_routing_classification_prompt` in `scorers.py`): choose exactly one
routing decision code (`A`/`B`/`C`/`D` for
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
artificial routing alternative among all candidates, not a fixed reference -- so
a candidate outscoring `no_tool` is never mistaken for genuine support when a
third candidate is actually strongest. `NoTool`/`no_tool` is not an executable
tool and is never emitted by the native agent; it exists only as an internal
direct-answer label and as an artificial candidate in this legacy probe. The
same canonical routing-classification prompt builder
(`build_routing_classification_prompt`) is shared by HF local classification
routing, HF local logprob scoring, calibration scoring, and coalition scoring,
so token boundaries and any residual template prior stay identical across all
four.

**This branch does not have the same fidelity as native tool-call scoring.**
`hf_router.LocalHFClassificationRouter` wraps an already-loaded
`CalibratedToolLogOddsScorer` and selects a tool by scoring every routing-label
continuation for the full (unmasked) request and taking the argmax. This path
remains available as a developer ablation; direct-answer explanations use it as
a retained legacy surrogate.

## HF Local Model Consistency

The implementation supports multiple HF Local models. Inference and XAI are
required to use the same model, tokenizer, chat template, device, and runtime
configuration. Developer diagnostics report both inference and scorer identity
so stale or cross-model explanations are blocked instead of silently displayed.

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
