# RAG Shapley Project Structure

## Positioning

This project should be presented as an explanation framework for RAG evidence,
not as a general-purpose RAG system. The RAG pipeline creates realistic
retrieved chunks, but the core research question is how to explain the
contribution of those chunks to the final answer.

Recommended title:

```text
Explaining Retrieval-Augmented Generation with Shapley Values and Shapley Interactions
```

Core thesis:

```text
Retrieved chunks in RAG are not equally useful, and retrieval scores do not
necessarily reflect how much each chunk supports the final answer. By modeling
retrieved chunks as players in a cooperative game, Shapley values can attribute
answer support to individual chunks and Shapley interactions can reveal
complementarity, redundancy, and conflict between chunks.
```

## Conceptual Pipeline

The system should be explained as five layers:

1. **RAG trace construction**
   - Input question.
   - Retrieved chunks from controlled scenarios or uploaded PDFs.
   - Optional generated answer from the selected context.

2. **Cooperative game definition**
   - Each retrieved chunk is one player.
   - A coalition is a subset of visible chunks.
   - The value function scores how strongly that subset supports the answer.

3. **Shapley attribution**
   - First-order Shapley values identify individual supporting chunks.
   - Pairwise Shapley interactions identify synergy, redundancy, and conflict.

4. **Faithfulness validation**
   - Deletion removes chunks in attribution order.
   - Insertion adds chunks in attribution order.
   - Baselines use retrieval-score order and random order.

5. **Explanation interface**
   - Retrieved chunks are highlighted by attribution.
   - Top evidence and strongest interactions are summarized.
   - Coalition scores and validation curves provide transparency.

## Recommended Code Boundaries

The existing implementation can remain mostly intact, but future refactors
should preserve these conceptual modules:

```text
demos/rag_retrieval_explanation/
  app.py                    # Streamlit UI and run orchestration
  rag_pipeline.py           # PDF parsing, chunking, retrieval, reranking
  rag_game.py               # Cooperative game over retrieved chunks
  value_functions.py        # Coalition support scorers
  evaluation.py             # Deletion/insertion validation utilities
  regression_check.py       # Fixed PDF retrieval regression check
  sample_data.py            # Controlled scenario fixtures
  evals/                    # Expected behavior and run records
```

A later cleanup could split `app.py` into UI modules, but this is not required
for the thesis if the current app is stable.

## Demo Scenarios

The controlled examples should be treated as mini experiments, not just sample
content:

| Scenario | Expected explanation behavior |
| --- | --- |
| Puzzle Pieces | Complementary chunks should both matter, with positive interaction. |
| Signal vs. Distractors | Direct evidence should outrank keyword-related distractors. |
| Missing Evidence | Full-context support should remain low. |
| Conflicting Context | Correct evidence should separate from plausible wrong associations. |
| Redundancy Detection | Near-duplicate chunks should split credit and show redundancy. |

These scenarios directly support the project claim that retrieval relevance and
answer support are different signals.

## Evaluation Records

Each meaningful run should save enough information to reproduce the result:

```text
config.yaml
retrieved_chunks.jsonl
coalition_scores.jsonl
shapley_values.csv
interactions.csv
deletion_curve.csv
insertion_curve.csv
metrics.json
notes.md
```

Minimum metadata:

```yaml
git_commit: <commit>
scenario: <scenario name>
value_function: <value function name>
interaction_index: <SV | k-SII | STII | FSII>
budget: <coalition evaluation budget>
retrieval_method: <scenario | dense embeddings | tf-idf>
embedding_model: <model id or none>
generator_model: <model id or none>
top_k: <retrieved chunk count>
```

Useful metrics:

```text
top_expected_evidence_rank
full_context_support
score_without_top_shapley_chunk
top_chunk_deletion_drop
deletion_auc_shapley
deletion_auc_retrieval
deletion_auc_random
insertion_auc_shapley
interaction_hit
missing_evidence_false_positive
```

## Thesis Outline

1. Introduction
   - RAG improves access to external evidence.
   - Retrieved chunks are hard to interpret.
   - Retrieval scores do not explain answer support.
   - Shapley attribution offers a principled explanation method.

2. Background
   - Retrieval-augmented generation.
   - Cooperative games.
   - Shapley values.
   - Shapley interactions.
   - Faithfulness evaluation.

3. Method
   - Retrieved chunks as players.
   - Coalition value functions.
   - Chunk-level attribution.
   - Pairwise interactions.
   - Sentence-level drilldown.

4. System
   - Controlled scenarios.
   - PDF RAG path.
   - Value function options.
   - Streamlit explanation interface.

5. Evaluation
   - Scenario-level expected behavior.
   - Deletion/insertion curves.
   - Retrieval-score and random baselines.
   - PDF regression case study.

6. Discussion
   - Value function choice matters.
   - Computational cost grows with retrieved chunk count.
   - Attribution explains support under the chosen scorer, not absolute truth.
   - Future work could add stronger entailment-based scorers and broader tasks.

7. Conclusion
   - Shapley-based explanations can reveal how retrieved evidence contributes
     to RAG answers in ways retrieval scores alone cannot.
