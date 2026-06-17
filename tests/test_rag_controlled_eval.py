"""Tests for the end-to-end controlled RAG evaluation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from demos.rag_retrieval_explanation.core.retrieval import (
    BGE_QUERY_INSTRUCTION,
    _prepare_embedding_texts,
)
from demos.rag_retrieval_explanation.core.schemas import RetrievedChunk
from demos.rag_retrieval_explanation.core.value_functions import (
    ContrastiveLikelihoodValue,
)
from demos.rag_retrieval_explanation.evals.benchmark import (
    benchmark_candidates,
    load_benchmark,
)
from demos.rag_retrieval_explanation.evals.metrics import (
    classify_knowledge_source,
    token_f1,
)
from demos.rag_retrieval_explanation.evals.run_controlled_eval import evaluate_case

if TYPE_CHECKING:
    from demos.rag_retrieval_explanation.core.model_backends import LlamaCppBackend


def test_controlled_benchmark_has_five_patterns_and_real_corpus() -> None:
    benchmark = load_benchmark()

    assert len(benchmark.cases) == 27
    assert len({case.pattern for case in benchmark.cases}) == 5
    assert len(benchmark_candidates(benchmark)) == len(benchmark.passages)
    assert len(benchmark.passages) > len(benchmark.cases)


def test_expanded_interaction_benchmark_has_valid_balanced_labels() -> None:
    benchmark = load_benchmark()
    known_ids = {passage.passage_id for passage in benchmark.passages}
    relation_counts: dict[str, int] = {}

    assert len(benchmark.cases) == 27
    assert len(benchmark.passages) == 108
    for case in benchmark.cases:
        for interaction in case.expected_interactions:
            assert interaction.left in known_ids
            assert interaction.right in known_ids
            assert interaction.expected_sign in {"positive", "negative"}
            relation_counts[interaction.relation] = (
                relation_counts.get(interaction.relation, 0) + 1
            )

    assert relation_counts == {
        "complementary": 10,
        "conflicting": 10,
        "redundant": 10,
    }


def test_direct_case_runs_retrieval_before_attribution() -> None:
    benchmark = load_benchmark()
    case = next(case for case in benchmark.cases if case.case_id == "direct_olympics_host")

    row, artifacts = evaluate_case(benchmark, case)

    assert row["retrieval_recall_at_k"] == 1.0
    assert isinstance(row["retrieved_ids"], str)
    assert "direct_olympics_evidence" in row["retrieved_ids"]
    assert isinstance(artifacts["retrieved"], list)
    assert len(artifacts["retrieved"]) == case.top_k
    assert artifacts["gold_attributions"]


def test_knowledge_source_labels_do_not_overclaim_when_both_answers_are_correct() -> None:
    assert (
        classify_knowledge_source(
            closed_book_f1=0.9,
            gold_context_f1=0.9,
            retrieved_context_f1=0.9,
        )
        == "source_not_identifiable"
    )
    assert (
        classify_knowledge_source(
            closed_book_f1=0.1,
            gold_context_f1=0.9,
            retrieved_context_f1=0.8,
        )
        == "retrieval_enabled"
    )


def test_token_f1_rewards_answer_overlap() -> None:
    assert token_f1("The answer is Ottawa.", "Ottawa is the capital.") > 0.4
    assert token_f1("Sydney", "Canberra") == 0.0


def test_bge_query_instruction_is_not_added_to_documents() -> None:
    model_id = "BAAI/bge-base-en-v1.5"

    assert _prepare_embedding_texts(
        ["capital of France"],
        model_id=model_id,
        is_query=True,
    ) == [f"{BGE_QUERY_INSTRUCTION}capital of France"]
    assert _prepare_embedding_texts(
        ["Paris is the capital of France."],
        model_id=model_id,
        is_query=False,
    ) == ["Paris is the capital of France."]


class FakeBackend:
    """Small deterministic backend for the three-condition model protocol."""

    def generate(self, question: str, context: str) -> object:
        del question
        answer = (
            "Beijing hosted the 2008 Summer Olympics."
            if "hosted by Beijing" in context
            else "The answer is unknown."
        )
        return SimpleNamespace(answer=answer)

    def build_prompt(self, question: str, context: str) -> str:
        return f"{question}\n{context}"

    def target_log_likelihood(self, prompt: str, target_answer: str) -> float:
        del target_answer
        return 5.0 if "hosted by Beijing" in prompt else -5.0


def test_model_protocol_separates_generated_and_gold_attribution() -> None:
    benchmark = load_benchmark()
    case = next(case for case in benchmark.cases if case.case_id == "direct_olympics_host")

    row, artifacts = evaluate_case(
        benchmark,
        case,
        value_function="target_likelihood",
        backend=FakeBackend(),
    )

    assert row["knowledge_source_class"] == "retrieval_enabled"
    assert row["generated_top_attribution_is_gold"] is True
    assert artifacts["gold_attributions"]
    assert artifacts["generated_attributions"]


def test_model_protocol_separates_generated_and_gold_interactions() -> None:
    benchmark = load_benchmark()
    case = next(
        case
        for case in benchmark.cases
        if case.case_id == "complementary_rgb_primary_light"
    )

    row, artifacts = evaluate_case(
        benchmark,
        case,
        method="TF-IDF",
        value_function="target_likelihood",
        backend=FakeBackend(),
    )

    assert row["interaction_total_pairs"] == 3
    assert row["generated_interaction_total_pairs"] == 3
    assert artifacts["interactions"]
    assert artifacts["generated_interactions"]
    assert artifacts["gold_pairwise_interactions"]
    assert artifacts["generated_pairwise_interactions"]
    assert artifacts["gold_coalition_scores"]
    assert artifacts["generated_coalition_scores"]


def test_missing_evidence_takes_priority_over_model_failure_label() -> None:
    benchmark = load_benchmark()
    case = next(case for case in benchmark.cases if case.case_id == "missing_eiffel_designer")

    row, _ = evaluate_case(
        benchmark,
        case,
        value_function="target_likelihood",
        backend=FakeBackend(),
    )

    assert row["knowledge_source_class"] == "missing_evidence"


def test_contrastive_value_caches_empty_context_likelihood() -> None:
    class CountingBackend(FakeBackend):
        def __init__(self) -> None:
            self.empty_context_calls = 0

        def target_log_likelihood(self, prompt: str, target_answer: str) -> float:
            if "(no retrieved context)" in prompt:
                self.empty_context_calls += 1
            return super().target_log_likelihood(prompt, target_answer)

    backend = CountingBackend()
    scorer = ContrastiveLikelihoodValue(backend=cast("LlamaCppBackend", backend))
    chunk = RetrievedChunk(title="Evidence", text="hosted by Beijing")

    scorer("Where were the Olympics hosted?", "Beijing", [chunk])
    scorer("Where were the Olympics hosted?", "Beijing", [chunk])

    assert backend.empty_context_calls == 1
