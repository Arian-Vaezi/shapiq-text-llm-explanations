# Explaining RAG Evidence with Shapley Values

This project explores how Shapley values and Shapley interactions can explain
which retrieved context chunks support an answer in a retrieval-augmented
generation (RAG) system.

The central idea is to treat retrieved chunks as players in a cooperative game.
For any subset of chunks, the game value measures how strongly that subset
supports the target or generated answer. Shapley values then attribute support
to individual chunks, while Shapley interactions reveal complementarity,
redundancy, and conflict between chunk pairs.

## Project Focus

This is not primarily a RAG benchmark project. The RAG pipeline is included to
produce realistic retrieved context, but the research focus is explanation:

- Which retrieved chunks actually support the answer?
- Do retrieval scores align with answer support?
- Which chunks only become useful together?
- Which chunks are redundant or distracting?
- Does removing high-attribution evidence reduce the support score?

## Main Demo

The main research prototype lives in:

```bash
demos/rag_retrieval_explanation/
```

It provides a Streamlit app with:

- controlled RAG scenarios for direct evidence, puzzle-piece evidence,
  distractors, missing evidence, conflicting context, and redundancy;
- a PDF upload path that builds a small RAG trace from an uploaded document;
- value functions for scoring context support;
- chunk-level Shapley attribution;
- pairwise Shapley interaction heatmaps;
- sentence-level drilldown;
- deletion and insertion validation curves.

Run the demo from the project root:

```bash
make app
```

## Recommended Thesis Framing

The project can be presented around three contributions:

1. **Formulation:** model retrieved RAG chunks as players in a cooperative game.
2. **Attribution:** use Shapley values and Shapley interactions to explain
   answer support, complementarity, redundancy, and distraction.
3. **Validation:** evaluate explanation faithfulness with controlled scenarios,
   deletion/insertion curves, and regression checks for the PDF RAG path.

The strongest claim is:

> Retrieval scores describe how chunks match a query, but Shapley attribution
> describes how chunks support the final answer.

## Evaluation Strategy

The recommended evaluation is explanation-centered rather than benchmark-first:

- **Controlled scenario checks:** verify that known evidence roles produce the
  expected attribution patterns.
- **Deletion validation:** removing top Shapley chunks should reduce support
  faster than removing chunks by retrieval score or random order.
- **Insertion validation:** adding top Shapley chunks should recover support
  faster than baseline orders.
- **Interaction checks:** puzzle-piece scenarios should produce positive
  interactions, while near-duplicate evidence should produce redundancy signals.
- **PDF regression checks:** fixed PDF questions should keep retrieving the
  expected evidence themes after pipeline changes.

See `demos/rag_retrieval_explanation/evals/` for the lightweight evaluation
record structure.

Run the controlled explanation evals:

```bash
make eval
```

## Useful Commands

Run the main demo:

```bash
make app
```

Run the generic retrieval policy tests:

```bash
make test
```

Run controlled explanation evals:

```bash
make eval
```

Run PDF smoke evals from a local JSON case file:

```bash
make pdf-smoke PDF_CASES=path/to/cases.json
```

Run pre-commit:

```bash
make lint
```
