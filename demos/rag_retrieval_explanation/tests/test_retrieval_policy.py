"""Regression tests for the demo's RAG context-selection policy."""

from __future__ import annotations

from demos.rag_retrieval_explanation.core.retrieval import (
    classify_query_intent,
    retrieve_relevant_chunks_with_debug,
)
from demos.rag_retrieval_explanation.core.schemas import CandidateChunk


def policy_fixture_chunks() -> list[CandidateChunk]:
    """Synthetic domain-neutral chunks for retrieval policy regression tests."""
    return [
        CandidateChunk(
            title="technical appendix",
            text=(
                "Appendix B specifies support platform timeout parameters, retry thresholds, "
                "queue configuration variables, and escalation algorithm constraints. The table "
                "is useful for implementation but narrow for a broad support overview."
            ),
            page_number=9,
            chunk_number=1,
            section_title="Technical Appendix",
            chunk_type="body_text",
            text_length=28,
        ),
        CandidateChunk(
            title="support overview",
            text=(
                "The platform improves customer support by summarizing requests, routing issues "
                "to the right team, reducing repeated manual triage, and giving agents a shared "
                "overview of previous customer interactions."
            ),
            page_number=2,
            chunk_number=2,
            section_title="Support Overview",
            chunk_type="body_text",
            text_length=30,
        ),
        CandidateChunk(
            title="security controls",
            text=(
                "Security controls include role based access, audit logs, data retention limits, "
                "and encryption for customer support records."
            ),
            page_number=4,
            chunk_number=3,
            section_title="Security",
            chunk_type="body_text",
            text_length=18,
        ),
        CandidateChunk(
            title="pricing note",
            text=(
                "The pricing model charges by seat count and monthly ticket volume. Enterprise "
                "plans include support onboarding and service level reviews."
            ),
            page_number=6,
            chunk_number=4,
            section_title="Pricing",
            chunk_type="body_text",
            text_length=20,
        ),
    ]


def retrieve_for_policy(question: str):
    """Run the policy with local sparse retrieval to avoid model downloads."""
    return retrieve_relevant_chunks_with_debug(
        question,
        policy_fixture_chunks(),
        top_k=3,
        method="TF-IDF",
        embedding_model_id="",
        embedding_device="cpu",
    )


def test_broad_synthesis_prefers_overview_over_technical_appendix() -> None:
    """Broad synthesis should start with overview evidence, not a narrow appendix."""
    question = "Summarize the main reasons the platform improves customer support."
    ranked, debug = retrieve_for_policy(question)

    assert classify_query_intent(question) == "broad_synthesis"
    assert debug.query_intent == "broad_synthesis"
    assert ranked[0].chunk.title.endswith("support overview")
    assert "support" in debug.coverage_summary["covered_query_terms"]


def test_narrow_technical_query_allows_appendix_first() -> None:
    """A narrow technical question should allow the directly relevant appendix to dominate."""
    question = "Which timeout parameters and retry thresholds does the support platform specify?"
    ranked, debug = retrieve_for_policy(question)

    assert debug.query_intent == "narrow_factual"
    assert ranked[0].chunk.title.endswith("technical appendix")


def test_specific_security_query_keeps_security_first() -> None:
    """A specific security question should not be hurt by synthesis balancing."""
    question = "What security controls protect customer support records?"
    ranked, debug = retrieve_for_policy(question)

    assert debug.query_intent == "narrow_factual"
    assert ranked[0].chunk.title.endswith("security controls")
