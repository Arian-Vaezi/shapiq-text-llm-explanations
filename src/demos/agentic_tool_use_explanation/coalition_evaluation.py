"""Typed, retrying coalition evaluation for the agentic tool-use demo.

Coalition value functions in this demo call out to real providers (Groq, Gemini, a
local HF router) or local embedding models. Those calls can fail transiently (rate
limits, timeouts, temporary 5xx) or fail semantically (no final answer, malformed
output, non-finite score). Before this module existed, a failed coalition could be
silently replaced by a fallback placeholder score (e.g. ``fallback_raw_score`` in
``final_answer_similarity_scorer.py``) and that placeholder would flow into
``ToolUseGame`` and then into ``ExactComputer`` as if it were a real game value.

This module owns:
    * typed evaluation outcomes (:class:`CoalitionEvaluationOutcome`);
    * retry classification and exponential-backoff orchestration;
    * aggregate execution metrics (:class:`CoalitionExecutionMetrics`);
    * the programmatic gate that refuses to let anything but fully real coalition
      values reach ``ExactComputer`` (:func:`ensure_all_real`,
      :class:`CoalitionEvaluationIncompleteError`);
    * conversion of a fully-real evaluation into ``ToolUseGame``'s raw value table.

It intentionally contains no Streamlit rendering and no caching.
"""

from __future__ import annotations

import functools
import math
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy as np
    from tool_game import ToolUseGame


class CoalitionEvaluationStatus(StrEnum):
    """Final disposition of one coalition's evaluation."""

    REAL = "real"
    RETRY_EXHAUSTED = "retry_exhausted"
    SEMANTIC_FAILURE = "semantic_failure"


@dataclass(frozen=True, slots=True)
class CoalitionEvaluationOutcome:
    """Typed result of evaluating one coalition, including retry bookkeeping."""

    score: float | None
    status: CoalitionEvaluationStatus
    attempts: int
    retry_count: int
    remote_request_count: int
    embedding_call_count: int
    failure_reason: str | None = None

    @property
    def is_real(self) -> bool:
        """Whether this outcome is a genuine, usable game value."""
        return self.status is CoalitionEvaluationStatus.REAL


@dataclass(frozen=True, slots=True)
class CoalitionExecutionMetrics:
    """Aggregate counters describing one full coalition-evaluation run."""

    coalition_total: int
    real_count: int
    fallback_count: int
    retry_triggered_count: int
    retry_success_count: int
    retry_exhausted_count: int
    semantic_failure_count: int
    remote_request_count: int
    embedding_call_count: int


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Exponential-backoff retry configuration for transient coalition failures."""

    max_attempts: int = 3
    initial_delay_seconds: float = 0.5
    max_delay_seconds: float = 8.0
    backoff_multiplier: float = 2.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            msg = "max_attempts must be >= 1."
            raise ValueError(msg)
        if self.initial_delay_seconds < 0:
            msg = "initial_delay_seconds must be non-negative."
            raise ValueError(msg)
        if self.max_delay_seconds < 0:
            msg = "max_delay_seconds must be non-negative."
            raise ValueError(msg)
        if self.max_delay_seconds < self.initial_delay_seconds:
            msg = "max_delay_seconds must be >= initial_delay_seconds."
            raise ValueError(msg)
        if self.backoff_multiplier < 1:
            msg = "backoff_multiplier must be >= 1."
            raise ValueError(msg)

    def delay_for_retry(self, retry_index: int) -> float:
        """Return the backoff delay before retry number ``retry_index`` (0-based)."""
        if retry_index < 0:
            msg = "retry_index must be non-negative."
            raise ValueError(msg)
        delay = self.initial_delay_seconds * (self.backoff_multiplier**retry_index)
        return min(delay, self.max_delay_seconds)


DEFAULT_RETRY_POLICY = RetryPolicy()


class CoalitionEvaluationIncompleteError(RuntimeError):
    """Raised when one or more coalition values could not be resolved honestly."""

    def __init__(
        self,
        metrics: CoalitionExecutionMetrics,
        failure_reasons: Sequence[str] = (),
    ) -> None:
        message = (
            f"Explanation aborted: {metrics.real_count} / {metrics.coalition_total} "
            "coalition values were resolved by real executions. "
            f"{metrics.retry_exhausted_count} coalition(s) failed after retries, "
            f"{metrics.semantic_failure_count} coalition(s) failed semantically. "
            "No result was produced because ExactComputer must not receive fallback "
            "or unresolved coalition values."
        )
        if failure_reasons:
            preview = "; ".join(list(failure_reasons)[:5])
            message += f" Failure reasons: {preview}"
        super().__init__(message)
        self.metrics = metrics
        self.failure_reasons = tuple(failure_reasons)


@dataclass(frozen=True, slots=True)
class CoalitionAttemptResult:
    """Outcome of one successful attempt to evaluate a coalition."""

    score: float
    remote_request_count: int = 1
    embedding_call_count: int = 0


@functools.lru_cache(maxsize=1)
def _transient_exception_types() -> tuple[type[BaseException], ...]:
    """Resolve transient exception types from whatever provider SDKs are installed.

    Resolved lazily (and cached) so this module does not require any optional
    provider SDK to be installed, and so test suites never need real network
    dependencies to exercise classification.
    """
    try:
        from demos.agentic_tool_use_explanation.agent_failure import (
            CoalitionTransientFailureError,
        )
    except ModuleNotFoundError:
        from agent_failure import CoalitionTransientFailureError

    transient_types: list[type[BaseException]] = [
        TimeoutError,
        ConnectionError,
        CoalitionTransientFailureError,
    ]
    try:
        import groq

        transient_types.extend(
            [
                groq.RateLimitError,
                groq.APITimeoutError,
                groq.APIConnectionError,
                groq.InternalServerError,
            ]
        )
    except ImportError:
        pass
    try:
        import httpx

        transient_types.extend(
            [
                httpx.TimeoutException,
                httpx.ConnectError,
                httpx.ReadTimeout,
                httpx.RemoteProtocolError,
            ]
        )
    except ImportError:
        pass
    return tuple(transient_types)


def classify_exception(exc: BaseException) -> Literal["transient", "semantic"]:
    """Classify a coalition-evaluation exception as transient (retryable) or semantic.

    Transient: rate limits (HTTP 429), timeouts, temporary 5xx / service-unavailable
    errors, and generic connection failures -- see :func:`_transient_exception_types`.

    Everything else (including malformed output, missing required fields, and
    provider auth/validation errors) is treated as semantic: retrying would not help
    because the failure is not about infrastructure availability.
    """
    if isinstance(exc, _transient_exception_types()):
        return "transient"
    return "semantic"


def evaluate_coalition_with_retry(
    attempt_fn: Callable[[], CoalitionAttemptResult],
    *,
    retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
    sleep_fn: Callable[[float], None] = time.sleep,
    classify_exception_fn: Callable[[BaseException], str] = classify_exception,
) -> CoalitionEvaluationOutcome:
    """Evaluate one coalition, retrying transient failures with exponential backoff.

    ``attempt_fn`` performs exactly one real evaluation attempt and either returns a
    :class:`CoalitionAttemptResult` or raises. Semantic failures (non-transient
    exceptions, or a non-finite returned score) are never retried. This function
    never substitutes a placeholder score: a coalition either ends up ``REAL`` with
    an actual score, or it ends up ``RETRY_EXHAUSTED``/``SEMANTIC_FAILURE`` with
    ``score=None``.
    """
    remote_request_count = 0
    embedding_call_count = 0
    failure_reason: str | None = None

    for attempt in range(1, retry_policy.max_attempts + 1):
        try:
            result = attempt_fn()
        except Exception as exc:  # noqa: BLE001 -- classified explicitly, never silently swallowed
            remote_request_count += 1
            failure_reason = f"{type(exc).__name__}: {exc}"
            if classify_exception_fn(exc) != "transient":
                return CoalitionEvaluationOutcome(
                    score=None,
                    status=CoalitionEvaluationStatus.SEMANTIC_FAILURE,
                    attempts=attempt,
                    retry_count=attempt - 1,
                    remote_request_count=remote_request_count,
                    embedding_call_count=embedding_call_count,
                    failure_reason=failure_reason,
                )
            if attempt >= retry_policy.max_attempts:
                return CoalitionEvaluationOutcome(
                    score=None,
                    status=CoalitionEvaluationStatus.RETRY_EXHAUSTED,
                    attempts=attempt,
                    retry_count=attempt - 1,
                    remote_request_count=remote_request_count,
                    embedding_call_count=embedding_call_count,
                    failure_reason=failure_reason,
                )
            sleep_fn(retry_policy.delay_for_retry(attempt - 1))
            continue
        else:
            remote_request_count += result.remote_request_count
            embedding_call_count += result.embedding_call_count
            if not math.isfinite(result.score):
                return CoalitionEvaluationOutcome(
                    score=None,
                    status=CoalitionEvaluationStatus.SEMANTIC_FAILURE,
                    attempts=attempt,
                    retry_count=attempt - 1,
                    remote_request_count=remote_request_count,
                    embedding_call_count=embedding_call_count,
                    failure_reason="Evaluation returned a non-finite score.",
                )
            return CoalitionEvaluationOutcome(
                score=float(result.score),
                status=CoalitionEvaluationStatus.REAL,
                attempts=attempt,
                retry_count=attempt - 1,
                remote_request_count=remote_request_count,
                embedding_call_count=embedding_call_count,
                failure_reason=None,
            )

    msg = "Unreachable: RetryPolicy.max_attempts validated to be >= 1."  # pragma: no cover
    raise AssertionError(msg)  # pragma: no cover


def aggregate_metrics(outcomes: Sequence[CoalitionEvaluationOutcome]) -> CoalitionExecutionMetrics:
    """Summarize a batch of coalition outcomes into aggregate execution metrics."""
    real_count = 0
    retry_triggered_count = 0
    retry_success_count = 0
    retry_exhausted_count = 0
    semantic_failure_count = 0
    remote_request_count = 0
    embedding_call_count = 0

    for outcome in outcomes:
        remote_request_count += outcome.remote_request_count
        embedding_call_count += outcome.embedding_call_count
        if outcome.retry_count > 0:
            retry_triggered_count += 1
        if outcome.status is CoalitionEvaluationStatus.REAL:
            real_count += 1
            if outcome.retry_count > 0:
                retry_success_count += 1
        elif outcome.status is CoalitionEvaluationStatus.RETRY_EXHAUSTED:
            retry_exhausted_count += 1
        elif outcome.status is CoalitionEvaluationStatus.SEMANTIC_FAILURE:
            semantic_failure_count += 1

    return CoalitionExecutionMetrics(
        coalition_total=len(outcomes),
        real_count=real_count,
        fallback_count=0,
        retry_triggered_count=retry_triggered_count,
        retry_success_count=retry_success_count,
        retry_exhausted_count=retry_exhausted_count,
        semantic_failure_count=semantic_failure_count,
        remote_request_count=remote_request_count,
        embedding_call_count=embedding_call_count,
    )


def evaluate_coalitions(
    attempt_fns: Sequence[Callable[[], CoalitionAttemptResult]],
    *,
    retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
    sleep_fn: Callable[[float], None] = time.sleep,
    classify_exception_fn: Callable[[BaseException], str] = classify_exception,
) -> tuple[list[CoalitionEvaluationOutcome], CoalitionExecutionMetrics]:
    """Evaluate every coalition sequentially and return outcomes plus aggregate metrics.

    Evaluation is strictly sequential (no concurrency); see module non-goals.
    """
    outcomes = [
        evaluate_coalition_with_retry(
            attempt_fn,
            retry_policy=retry_policy,
            sleep_fn=sleep_fn,
            classify_exception_fn=classify_exception_fn,
        )
        for attempt_fn in attempt_fns
    ]
    return outcomes, aggregate_metrics(outcomes)


def ensure_all_real(
    outcomes: Sequence[CoalitionEvaluationOutcome],
    metrics: CoalitionExecutionMetrics,
) -> list[float]:
    """Return numeric scores for ``ToolUseGame``, or raise if any coalition is not real.

    This is the programmatic ExactComputer gate: callers must call this (or
    :func:`evaluate_game_exactly`, which calls it internally) before constructing an
    ``ExactComputer`` / calling ``compute_exact_interactions``. A partially successful
    or fallback-contaminated set of scores must never reach that point.
    """
    if metrics.real_count != metrics.coalition_total:
        failure_reasons = [outcome.failure_reason for outcome in outcomes if outcome.failure_reason]
        raise CoalitionEvaluationIncompleteError(metrics, failure_reasons)
    return [outcome.score for outcome in outcomes]  # type: ignore[misc] -- all REAL, so non-None


def evaluate_game_exactly(
    game: ToolUseGame,
    *,
    retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
    sleep_fn: Callable[[float], None] = time.sleep,
    classify_exception_fn: Callable[[BaseException], str] = classify_exception,
) -> CoalitionExecutionMetrics:
    """Evaluate every coalition of ``game`` exactly (including the empty coalition).

    ``game`` must have been constructed with
    ``ToolUseGame(..., defer_empty_coalition_evaluation=True)``. That is the only
    supported way to avoid the empty coalition being scored twice: once here, and
    once more by ``ToolUseGame.__init__``'s own default (eager, non-deferred)
    normalization. With the deferred flag, construction makes no scorer call at
    all, and every one of the ``2**game.n_players`` coalitions -- including the
    empty coalition -- is evaluated through the same typed-outcome/retry protocol
    here, exactly once.

    On full success:
        * ``game.game_values`` is populated with the real (un-normalized) raw score
          for every coalition, and the game is marked precomputed, so a subsequent
          ``ExactComputer`` run reads from this exact table instead of re-evaluating
          the value function;
        * ``game.normalization_value`` is set to the resolved empty coalition's raw
          score (if ``game.normalize_requested``), and the game's deferred-baseline
          guard is lifted, so ``game(empty_coalition)`` evaluates to exactly
          ``0.0`` -- the same contract non-deferred ``ToolUseGame`` construction
          establishes eagerly, just resolved through retry instead.

    The value-function formula itself is untouched; this only changes *how* and
    *when* the raw table and normalization baseline are filled in.

    Batching limitation: each coalition is evaluated with its own
    ``game.value_function`` call (one row), to get independent per-coalition typed
    outcomes and retry. This does not change ``ToolScorerProtocol.score_batch``'s
    public interface -- a scorer that batches internally (e.g.
    ``CalibratedToolLogOddsScorer``'s GPU micro-batching) still receives a one-prompt list
    here and batches as much as it can per call. Other callers that pass many rows
    at once (e.g. the official shapiq approximators used above
    ``MAX_EXACT_DEMO_PLAYERS``) still get genuine multi-coalition batching; only
    this exact, retry-aware path evaluates one coalition per call. A future
    cache-aware evaluator could batch all cache misses into one ``value_function``
    call on the happy path and only fall back to this per-coalition retry loop for
    the coalitions that batch call reports as failed.

    Raises:
        CoalitionEvaluationIncompleteError: If any coalition (including the empty
            one) ends up ``RETRY_EXHAUSTED`` or ``SEMANTIC_FAILURE``. ``game`` is
            left unmodified in that case: no precomputed table, no normalization
            change, so nothing downstream can call ``ExactComputer`` on partial data.
    """
    import numpy as np

    from shapiq.utils import powerset

    coalition_tuples = list(powerset(range(game.n_players)))
    embedding_call_count = 1 if hasattr(game.scorer, "embedder") else 0

    def make_attempt_fn(coalition_tuple: tuple[int, ...]) -> Callable[[], CoalitionAttemptResult]:
        row: np.ndarray = np.zeros((1, game.n_players), dtype=bool)
        for player_index in coalition_tuple:
            row[0, player_index] = True

        def attempt() -> CoalitionAttemptResult:
            score = float(game.value_function(row)[0])
            return CoalitionAttemptResult(
                score=score,
                remote_request_count=1,
                embedding_call_count=embedding_call_count,
            )

        return attempt

    attempt_fns = [make_attempt_fn(coalition_tuple) for coalition_tuple in coalition_tuples]
    outcomes, metrics = evaluate_coalitions(
        attempt_fns,
        retry_policy=retry_policy,
        sleep_fn=sleep_fn,
        classify_exception_fn=classify_exception_fn,
    )
    values = ensure_all_real(outcomes, metrics)  # raises CoalitionEvaluationIncompleteError

    values_by_tuple = dict(zip(coalition_tuples, (float(value) for value in values), strict=True))
    game.game_values = values_by_tuple
    game._precompute_flag = True  # noqa: SLF001 -- mirrors shapiq.Game.precompute()'s own contract
    # Resolve the normalization baseline from this same retry-protected evaluation,
    # exactly once -- not from any separate, unprotected construction-time call.
    if getattr(game, "normalize_requested", True):
        game.normalization_value = values_by_tuple[()]
    else:
        game.normalization_value = 0.0
    game._normalization_resolved = True  # noqa: SLF001 -- lifts ToolUseGame.__call__'s guard
    return metrics
