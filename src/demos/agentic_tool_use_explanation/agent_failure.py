"""Structured failure classification shared by agent wrappers and coalition evaluation.

``groq_agent.run_groq_tool_inference`` and ``gemini_agent.run_gemini_tool_inference``
both catch provider exceptions internally and embed them into an opaque result object
(``GroqInferenceResult.error`` / ``GeminiInferenceResult.error``) instead of letting
them propagate. That means a caller several layers up (e.g.
``final_answer_similarity_scorer.FinalAnswerSimilarityScorer``) cannot see the
original exception type to decide whether a failure is worth retrying.

This module is the single, documented, tested place that recovers a structured
:class:`AgentFailureKind` from such a result -- using the real exception type
whenever it is still available (:func:`classify_agent_exception`), and falling back
to isolated string matching only when it is not (:func:`classify_agent_error_text`,
used for opaque/legacy error text such as a result object's ``.error`` message).
"""

from __future__ import annotations

from enum import StrEnum


class AgentFailureKind(StrEnum):
    """Structured classification of why an agent-inference call did not succeed."""

    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    TRANSIENT_PROVIDER = "transient_provider"
    AUTHENTICATION = "authentication"
    INVALID_REQUEST = "invalid_request"
    NO_FINAL_ANSWER = "no_final_answer"
    UNKNOWN = "unknown"


TRANSIENT_AGENT_FAILURE_KINDS = frozenset(
    {
        AgentFailureKind.RATE_LIMIT,
        AgentFailureKind.TIMEOUT,
        AgentFailureKind.TRANSIENT_PROVIDER,
    }
)
"""Failure kinds worth retrying: infrastructure/availability problems, not content."""


class CoalitionTransientFailureError(Exception):
    """Raised at a scorer/agent-wrapper boundary for a classified-transient failure.

    Lets call sites that only have an opaque result object (not the original
    exception) still participate correctly in
    ``coalition_evaluation``'s retry classification: this type is always
    recognized as transient (see ``coalition_evaluation._transient_exception_types``).
    """


def classify_agent_exception(exc: BaseException) -> AgentFailureKind:
    """Classify a raised provider exception, preferring real exception types.

    Checks the ``groq`` SDK's typed exception hierarchy first (when importable),
    then falls back to :func:`classify_agent_error_text` on ``str(exc)`` -- which
    also covers test doubles and providers (e.g. Gemini in this environment) that
    do not expose a typed exception hierarchy here.
    """
    try:
        import groq
    except ImportError:
        pass
    else:
        if isinstance(exc, groq.RateLimitError):
            return AgentFailureKind.RATE_LIMIT
        if isinstance(exc, groq.APITimeoutError):
            return AgentFailureKind.TIMEOUT
        if isinstance(exc, groq.APIConnectionError | groq.InternalServerError):
            return AgentFailureKind.TRANSIENT_PROVIDER
        if isinstance(exc, groq.AuthenticationError | groq.PermissionDeniedError):
            return AgentFailureKind.AUTHENTICATION
        if isinstance(
            exc,
            groq.BadRequestError | groq.NotFoundError | groq.UnprocessableEntityError,
        ):
            return AgentFailureKind.INVALID_REQUEST
    return classify_agent_error_text(str(exc))


def classify_agent_error_text(error_text: str) -> AgentFailureKind:
    """Best-effort classification from error text alone.

    Isolated fallback for opaque result objects whose original exception type is
    unavailable (e.g. a caught-and-stringified provider error, or a provider SDK
    this demo does not import a typed exception hierarchy for). Prefer
    :func:`classify_agent_exception` whenever the real exception object is still in
    hand -- this string matching is inherently best-effort and provider-text
    dependent.
    """
    normalized = error_text.upper()
    if "429" in normalized or "RATE_LIMIT" in normalized or "RATE LIMIT" in normalized:
        return AgentFailureKind.RATE_LIMIT
    if "RESOURCE_EXHAUSTED" in normalized or "QUOTA" in normalized:
        return AgentFailureKind.RATE_LIMIT
    if "TIMEOUT" in normalized or "TIMED OUT" in normalized or "TIMED_OUT" in normalized:
        return AgentFailureKind.TIMEOUT
    if (
        "503" in normalized
        or "500" in normalized
        or "502" in normalized
        or "504" in normalized
        or "UNAVAILABLE" in normalized
        or "OVERLOADED" in normalized
        or "INTERNAL_SERVER" in normalized
        or "INTERNAL SERVER" in normalized
    ):
        return AgentFailureKind.TRANSIENT_PROVIDER
    if (
        "401" in normalized
        or "403" in normalized
        or "AUTH" in normalized
        or "API KEY" in normalized
        or "API_KEY" in normalized
    ):
        return AgentFailureKind.AUTHENTICATION
    if "400" in normalized or "404" in normalized or "NOT_FOUND" in normalized:
        return AgentFailureKind.INVALID_REQUEST
    return AgentFailureKind.UNKNOWN
