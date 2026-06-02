# RAG Retrieval Explanation Demo

This Streamlit app is the main research prototype for explaining retrieved RAG
evidence with Shapley values and Shapley interactions.

The demo treats retrieved chunks as players in a cooperative game. For each
coalition of chunks, the app scores how well the selected context supports a
reference or generated answer, then uses shapiq to estimate chunk-level Shapley
values and pairwise interactions. This makes the central distinction visible:
retrieval scores measure query relevance, while Shapley values measure answer
support under the chosen value function.

The app also performs a finer drilldown by splitting retrieved chunks into
sentence players, running a sentence-level shapiq game, and displaying
sentence-level attribution.

The framework is intentionally isolated from the package code. It only adds
files under `demos/rag_retrieval_explanation/`.

## Research Question

RAG systems usually expose retrieved chunks and retrieval scores, but those
scores do not explain how each chunk contributes to the final answer. This demo
asks:

- Which retrieved chunks actually support the answer?
- Which chunks are distractors despite looking query-relevant?
- Which chunks are complementary and only useful together?
- Which chunks are redundant and split credit?
- Does removing high-attribution evidence reduce answer support?

## Method Overview

The prototype has five conceptual layers:

1. Build or load a RAG trace: question, answer, and retrieved chunks.
2. Treat chunks as cooperative-game players.
3. Define a value function `v(S)` that scores answer support from a chunk subset.
4. Compute Shapley values and pairwise interactions with shapiq.
5. Validate attribution with insertion/deletion curves and baseline orders.

The recommended thesis framing is:

> Shapley-based attribution explains how retrieved evidence contributes to RAG
> answers in ways retrieval scores alone cannot.

## Run

From the repository root:

```bash
uv pip install -r demos/rag_retrieval_explanation/requirements.txt
uv run streamlit run demos/rag_retrieval_explanation/app.py
```

If your environment already has `streamlit` installed, the second command is
enough.

When using a Hugging Face-backed scorer for a gated model, log in once from the
repository root:

```bash
uv run huggingface-cli login
```

The token is cached locally by Hugging Face, so this is not needed for every
Streamlit run unless the token is removed, revoked, or you switch machines.

## What The App Shows

- A RAG question, reference/generated answer, and retrieved chunks.
- A compact top toolbar with input-source selection, scenario or PDF summary,
  selected value-function summary, interaction-index summary, and the run action.
- Main navigation tabs for **Dashboard**, **Explanation Settings**,
  **Value Functions**, and **RAG Pipeline**; the Streamlit sidebar is
  intentionally unused.
- A dedicated **Explanation Settings** view for index, budget, value-function,
  Hugging Face runtime configuration, and uploaded-PDF RAG input.
- A real PDF RAG path: upload a PDF, enter a question, retrieve and rerank
  chunks from extracted PDF text with dense embeddings plus sparse keyword
  retrieval, generate the answer with the configured Gemma/HF model, and
  explain those retrieved chunks.
- Intent-aware final context selection: narrow factual questions stay close to
  normal relevance ranking, while broad synthesis questions use diversity-aware
  context selection so one narrow subsection does not dominate the prompt.
- A coalition game where each chunk can be included or removed from the prompt.
- A selectable value function: local lexical grounding, Hugging Face target-answer likelihood, Hugging Face contrastive likelihood, or generated-answer overlap.
- A run monitor that reports model loading, coalition scoring, chunk-level
  attribution, evaluation, and sentence drilldown progress.
- First-order attribution: which chunk supports the answer most.
- Second-order shapiq interaction heatmap: which chunk pairs interact under the selected
  index (`k-SII`, `STII`, or `FSII`).
- For `SV`, interaction panels show an explicit first-order-only note instead
  of empty second-order plots.
- Retrieved-text highlighting where the chunk background intensity follows the
  attribution score.
- Deletion and insertion validation curves for path-based faithfulness checks,
  including support score and step drop/gain. Deletion compares Shapley order,
  retrieval-score order, and random removal.
- Retrieval debug traces for uploaded PDFs: query intent, expanded queries, raw
  dense/keyword/rerank order, final prompt order, score components, selected
  reasons, and generic coverage summaries.
- Sentence drilldown: all chunks are split into sentence players, then shown as
  sentence-level attribution bars and a compact attribution table.
- Coalition audit table: selected coalitions and their support scores.
- Performance controls for expensive HF-backed runs: approximate mode, player
  limits, cached coalition scores, optional validation/interaction skips, and a
  separate sentence-drilldown button.

When multiple value functions are selected for a run, the dashboard exposes an
**Analysis value function** selector. Evidence cards, the attribution chart,
coalition scores, Shapley interactions, sentence attribution, and validation
tabs all use that active value-function run. **Value Function Comparison** is
the dedicated cross-value-function view.

## Dashboard Layout

The main dashboard is organized for explanation rather than debugging:

1. **Top toolbar:** choose built-in sample traces or PDF upload, inspect the
   selected value functions and interaction index, and run the explanation.
2. **Context panel:** question, reference/generated answer, and retrieved chunks are shown in
   the main page, with subtle attribution highlighting after a run.
3. **Summary panel:** full-context support, top evidence, selected value
   function, and strongest interaction are grouped in one right-side overview.
4. **Advanced analysis tabs:** secondary analysis is grouped into three layers:
   value-function scores, shapiq attribution, and validation.
5. **Value Functions page:** method documentation, formulas, caveats, speed, and
   model requirements live outside the dashboard so the evidence panel remains
   the visual focus.
6. **RAG Pipeline page:** technical documentation for PDF loading, chunking,
   metadata, dense retrieval, TF-IDF, reranking, context selection, grounded
   generation, and Shapley attribution.

## Scenario Pages

The app is organized around clean demo scenarios:

- **Page 1 - Puzzle Pieces:** multiple chunks must be combined to support one
  answer. The included Marie Curie example is meant for interaction indices such
  as `k-SII` and `STII`.
- **Page 2 - Signal vs. Distractors:** one direct evidence chunk competes with
  keyword-related distractors. The included 2008 Beijing Olympics example is
  meant for first-order attribution ranking.
- **Page 3 - Missing Evidence:** the retrieved context does not actually support
  the target answer.
- **Page 4 - Conflicting Context:** one chunk gives the answer while other chunks
  point toward common wrong associations.

## PDF RAG Pipeline

The **PDF upload** input source builds a real RAG trace instead of using curated
retrieved chunks. After upload, ask a question from the dashboard chat input.
The app then:

1. extracts selectable text from the uploaded PDF with `pypdf`,
2. splits page text into overlapping chunks and attaches metadata such as page
   number, section title, chunk type, text length, and filtering flags,
3. expands broad synthesis queries with generic phrases such as overview, main
   reasons, major themes, objectives, motivations, and scientific value,
4. retrieves candidates with dense embedding cosine similarity by default
   (`sentence-transformers/all-MiniLM-L6-v2`) and sparse TF-IDF keyword scores,
5. reranks candidates with relevance, keyword overlap, and metadata quality,
6. selects the final prompt context with an intent-aware policy:
   **narrow factual** questions use high relevance weight, while
   **broad synthesis** questions use a lower relevance weight and stronger
   novelty/diversity pressure,
7. asks the configured Hugging Face model, defaulting to
   `google/gemma-4-E2B-it`, to answer using only the retrieved context,
8. sends the generated answer and actual retrieved chunks into the existing
   shapiq attribution game.

This keeps the retrieval step separate from the model's parametric knowledge:
Gemma receives only the chunks selected by the retriever. Scanned PDFs without
selectable text need OCR first.

### Context Selection Policy

The final context selector is intentionally not hard-coded to one report or one
failure case. It first classifies query intent:

- **narrow_factual:** specific who/what/when/which/list/prioritize/definition
  questions. These keep a high relevance weight and behave close to normal
  rerank/top-k retrieval.
- **broad_synthesis:** why/importance/interesting/overview/summarize/major
  motivations/themes/objectives questions. These use a Maximal Marginal
  Relevance-style policy to avoid redundant or overly narrow context.

For a candidate chunk `c` and already selected context `C`, the selector uses:

```text
S(c | C) = lambda * relevance(c)
         + (1 - lambda) * novelty(c, C)
         + quality_bonus(c)
         - narrow_subsection_penalty(c)
```

Novelty rewards new section titles, new query-term coverage, and low
similarity to already selected chunks. Quality rewards body-text,
overview/summary/objective chunks, sufficient length, and low citation density.
The narrow-subsection penalty discourages highly technical goal/subsection
chunks from becoming the first context chunk for broad synthesis questions,
unless the user explicitly asks about that narrow topic.

Narrow chunks are not removed. They can still appear as supporting context when
relevant, and they are allowed to rank first for narrow questions about that
specific topic.

### Retrieval Debugging

For uploaded PDFs, expand **Retrieval debug** on the dashboard to inspect:

- query intent,
- original and expanded queries,
- raw dense retrieval,
- raw keyword retrieval,
- raw rerank order,
- final prompt order,
- per-selected-chunk relevance, novelty, concept coverage, quality bonus,
  narrow-subsection penalty, final selection score, and selection reason,
- query-term coverage, chunk types, and selected section titles.

## Value Functions

The default scorer is a lightweight lexical scorer so the demo runs without a
GPU or downloaded model. It is meant as scaffolding. The app now also includes
Hugging Face-backed value functions for the real project run:

- **Lexical grounding:** fast local answer-term coverage baseline.
- **HF target likelihood:** average target-answer log-likelihood under a causal
  LM given the selected retrieved context.
- **HF contrastive likelihood:** target-answer likelihood gain relative to a
  no-context prompt.
- **HF generated answer overlap:** generate an answer and score token overlap
  with the target answer.

Set the Hugging Face model id on the **Explanation Settings** page. The default
is `google/gemma-4-E2B-it`, the smallest Gemma 4 instruct option currently used
by this demo. Replace it with a larger Gemma 4 model if your machine or cluster
has enough memory. Model settings are hidden when the selected value functions
do not require an HF/local model, such as the lexical baseline.

The HF runtime settings use dropdown presets for model id, `device_map`,
`torch_dtype`, and `max_new_tokens` so users do not need to remember exact
configuration strings while demoing.

Gemma 4 may require a newer Hugging Face `transformers` build than the version
available in the current environment. If loading fails with `model_type:
gemma4` not recognized, upgrade `transformers` to a build that includes Gemma 4
support or temporarily use a supported model to validate the pipeline.

The relevant hooks are:

- `rag_pipeline.py` for PDF parsing, chunking, retrieval, and RAG answer generation,
- `rag_pipeline.classify_query_intent` and `_select_diverse_context` for the
  intent-aware final context policy,
- `value_functions.py` for adding new value functions,
- `model_backends.py` for model loading/generation/log-likelihood,
- `RAGRetrievalGame.score_context` in `rag_game.py` for the shapiq game call.

Hugging Face scoring is much slower than the lexical scorer because shapiq calls
the value function for many coalitions. For early experiments, keep the number
of chunks small and use a low approximation budget.

HF models are loaded only after pressing **Run chunk attribution** or the
sentence drilldown button. The run monitor shows the current stage and the
number of coalition scoring calls so first-time downloads or slow local
inference do not look like a frozen page.

## Performance Controls

HF-backed attribution can be expensive because shapiq evaluates many coalitions
and each coalition may require a model call. The app keeps the default path
fast and safe:

- **Run chunk attribution** only computes chunk-level attribution.
- Sentence drilldown is not run automatically. For HF-backed value functions,
  the Sentence Attribution tab shows that drilldown was skipped until the user
  explicitly runs it.
- Sentence drilldown can be scoped to all sentences, the top-1 chunk, the top-2
  chunks, or positive-attribution chunks only. The default scope is top-2
  chunks.
- Sentence-level runs enforce player limits. Defaults are 5 players for
  HF-backed sentence attribution and 6 players for lexical sentence attribution.
- Approximate mode is the default. Exact mode is capped by the configured exact
  run cap.
- Coalition scores are cached in Streamlit session state. Cache keys include the
  scenario, value function, model id, question, target answer, player type,
  coalition ids, scoring settings, index, and interaction order.
- HF tokenizers and models are cached separately in `model_backends.py`, so
  reruns and tab switches do not reload model resources.
- Deletion/insertion validation and interaction maps can be skipped from the
  Explanation Settings page when running expensive scorers.

Switching tabs, expanding trace tables, or changing display-only controls does
not recompute attribution. Chunk attribution recomputes only when pressing
**Run chunk attribution**. Sentence attribution recomputes only when pressing
**Run sentence drilldown for active value function**.

## Evaluation

The evaluation tab runs simple faithfulness checks after attribution:

- **Deletion:** remove chunks by descending absolute attribution magnitude. A
  faithful ranking should make the score drop quickly. The chart also overlays
  removal by retriever relevance score and an average random-removal curve; this
  tests whether Shapley contribution ranking explains answer support better than
  query similarity alone.
- **Insertion:** add chunks in the same ranking order. A faithful ranking should
  recover the full-context score quickly.
- **Top-only sufficiency:** compare the top chunk alone with the full context.

Deletion and insertion curves are path-based validation checks. Their step
changes are not identical to Shapley attributions because each chunk is evaluated
under a specific remaining or selected context.

These are deliberately lightweight so they work for both lexical and Hugging
Face value functions. They give you a concrete answer when asked how the
explanation is evaluated.

## Regression Checks

The repository includes small retrieval-policy tests for the PDF RAG selector:

```bash
uv run pytest tests/test_rag_retrieval_policy.py
```

These synthetic tests cover:

- a broad synthesis query where a narrow technical subsection must not become
  the first context chunk,
- a narrow query where the directly requested technical chunk is allowed to rank
  first,
- a specific query where directly matching evidence should remain highly ranked.

For a real PDF report, use the standalone retrieval regression helper:

```bash
uv run python demos/rag_retrieval_explanation/regression_check.py path/to/report.pdf \
  --question "What is the main contribution?" \
  --expected-evidence-term "contribution"
```

This script runs PDF parsing, chunking, retrieval, reranking, and final context
selection without running Gemma generation or Shapley attribution. It prints the
selected chunks, flags, and coverage summary. Optionally pass
`--answer-file path/to/generated_answer.txt`, `--expected-answer-term`, and
`--forbidden-answer-term` to check answer-level smoke expectations.

## Implementation Notes

The comments in the Python files mark the main replacement points:

- `lexical_grounding_score` is demo scaffolding, not the final scientific scorer.
- `HuggingFaceCausalLMBackend` is the first real model-call path. It computes target-answer log-likelihood and greedy generated answers with `transformers`.
- `ProgressScorer` wraps value functions for Streamlit progress reporting. It
  does not change scores; it only updates the run monitor between coalition
  evaluations.
- `split_sentences` is a simple regex splitter for curated examples; replace it with the project segmentation utilities if the final demo needs robust sentence handling.
- The sentence drilldown intentionally runs as a second stage after chunk-level
  attribution: first find important retrieved chunks, then inspect all
  sentences inside the retrieved context.

## Suggested Final Demo Story

Use one real RAG trace with 4-8 retrieved chunks:

1. One chunk directly supports the answer.
2. One chunk contains related but incomplete evidence.
3. One chunk is distracting or misleading.
4. Two chunks together support a fact that neither fully supports alone.

Then show that first-order Shapley values identify the main evidence source,
while second-order shapiq interactions reveal multi-chunk grounding, redundancy,
or distracting relationships between retrieved chunks.

The sentence drilldown is meant as the second stage of the story: after locating
important retrieved chunks, inspect which sentences across all chunks carry the
answer support. Cross-chunk sentence interactions are useful for multi-source RAG
grounding, while same-chunk interactions show local evidence structure.
