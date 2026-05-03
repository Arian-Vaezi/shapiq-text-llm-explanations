# RAG Retrieval Explanation Demo

Streamlit framework for **[Demo] RAG-retrieval explanation #4**.

This demo treats retrieved RAG chunks as players in a cooperative game. For each
coalition of chunks, the app builds a prompt, scores how well the selected
context supports a target answer, and then uses shapiq to estimate chunk-level
Shapley values and pairwise interactions.

The app then performs a second, finer drilldown: it splits all retrieved chunks
into sentence players, runs a sentence-level shapiq game, and visualizes the
sentence attributions/interactions with the package sentence visualization
utilities.

The framework is intentionally isolated from the package code. It only adds
files under `demos/rag_retrieval_explanation/`.

## Run

From the repository root:

```bash
uv pip install -r demos/rag_retrieval_explanation/requirements.txt
uv run streamlit run demos/rag_retrieval_explanation/app.py
```

If your environment already has `streamlit` installed, the second command is
enough.

## What The App Shows

- A RAG question, target answer, and retrieved chunks.
- A coalition game where each chunk can be included or removed from the prompt.
- First-order attribution: which chunk supports the answer most.
- Second-order shapiq interaction heatmap: which chunk pairs interact under the selected
  index (`k-SII`, `STII`, or `FSII`).
- Chunk interaction network: node size shows attribution strength and edge width shows
  second-order interaction strength.
- Sentence drilldown: all chunks are split into sentence players, then plotted with
  `token_attribution_bar_plot` and `sentence_interaction_heatmap`.
- Coalition audit table: selected coalitions and their support scores.

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

## Scoring Modes

The default scorer is a lightweight lexical scorer so the demo runs without a
GPU or downloaded model. It is meant as scaffolding.

For the final project demo, replace or extend the scorer with a model-backed
target such as:

- target-answer log-likelihood from a causal LLM,
- answer entailment / groundedness classifier score,
- contrastive log-odds between a grounded and unsupported answer,
- exact-match / semantic similarity score for generated answers.

The relevant hook is `RAGRetrievalGame.score_context` in `rag_game.py`.

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
grounding, while same-chunk interactions show local evidence structure. This
intentionally reuses the sentence-level visualization features from
`src/shapiq/plot/sentence.py`.
