# RAG Evaluation and Paper Design

## Goal

Evaluate the complete retrieval-to-attribution pipeline while keeping the
scientific focus on Shapley explanations of retrieved evidence.

## Experimental Assets

The project uses three complementary assets:

1. A bundled controlled corpus is the primary benchmark. The retriever searches
   the complete corpus; cases do not inject a preselected chunk set.
2. A future QASPER subset provides external validity with naturally occurring
   research-paper questions and human evidence annotations.
3. The Artemis PDF remains a difficult qualitative case study rather than the
   source of the main quantitative claims.

The controlled benchmark contains five evidence patterns with four topics per
pattern: direct evidence, complementary evidence, missing evidence, conflicting
evidence, and redundant evidence. Each case declares a gold answer, gold
evidence passage IDs, and expected interaction labels where appropriate.

## Evaluation Layers

The evaluation reports separate metrics for:

- Retrieval: Recall@k, Precision@k, MRR, and evidence rank.
- Answering: token F1 against the gold answer under closed-book, gold-context,
  and retrieved-context conditions.
- Attribution: gold-evidence attribution mass, top-evidence accuracy, deletion
  AUC, insertion AUC, and comparison with retrieval-score and random orders.
- Interaction: whether labelled complementary and redundant evidence pairs have
  the expected interaction sign and rank among the strongest pairs.

## Knowledge-Source Protocol

For each question, the same local model is evaluated under:

1. Closed-book: no retrieved context.
2. Gold-context: only human-labelled evidence.
3. Retrieved-context: the retriever's top-k chunks.

The comparison supports limited causal statements:

- Closed-book wrong, retrieved-context right: retrieval-enabled answer.
- Retrieved-context wrong, gold-context right: retrieval failure.
- Gold-context wrong: generation or model-capability failure.
- Closed-book and retrieved-context both right: source is not identifiable from
  correctness alone.

Context removal effects and target-likelihood changes provide stronger evidence
of context dependence, but they still do not expose the model's hidden internal
reasoning process.

## Dual Attribution Targets

Every full run keeps these objects distinct:

- `gold_answer`: independently annotated reference answer.
- `generated_answer`: model output under retrieved context.
- `coalition_value`: score assigned to one subset of retrieved chunks.

Gold-answer attribution asks which chunks support the correct answer.
Generated-answer attribution asks which chunks increase the model likelihood of
the answer it actually produced. The report must not treat either explanation
as a factual-correctness certificate.

## Paper Maintenance

`paper/main.tex` is the living manuscript. Changes to the benchmark, metrics,
research questions, experimental settings, or measured results must update the
corresponding manuscript section and `paper/research_log.tex` in the same
change. Placeholder result text must remain visibly marked until produced by a
recorded evaluation run.

