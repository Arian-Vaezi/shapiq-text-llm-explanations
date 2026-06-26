"""End-to-end final-answer semantic similarity scorer for the agentic tool-use demo.

Unlike the routing-only scorers in ``router_scorers.py`` and ``scorers.py``, this
scorer re-runs the *complete* tool-calling agent (the same pipeline used by the
Inference tab's "Run inference" action: router/tool-choice -> tool execution ->
final answer) for every coalition, and compares the coalition's final
natural-language answer against the full-request answer with embedding cosine
similarity.

This measures semantic fidelity to the full-run behavior, not guaranteed
factual correctness: a coalition can score highly while still being factually
wrong, as long as its answer resembles the full-run answer.

Limitation -- not always a "post-tool" answer: for the Groq and Gemini
backends (``groq_agent.py`` / ``gemini_agent.py``), ``weather_tool`` and
``web_search_tool`` final answers are template sentences such as "I would use
the weather tool to retrieve the forecast for ..." that do *not* actually
consume the demo tool's (fake) output -- only ``calculator_tool`` answers are
grounded in a real computed result. This scorer measures fidelity to whatever
final answer the configured backend actually produces, which for those two
tools is closer to "would-call" phrasing than a genuinely tool-grounded
answer. See ``groq_agent._build_assistant_answer`` /
``gemini_agent._build_assistant_answer``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

try:
    from demos.agentic_tool_use_explanation.agent_failure import (
        TRANSIENT_AGENT_FAILURE_KINDS,
        CoalitionTransientFailureError,
    )
except ModuleNotFoundError:
    from agent_failure import TRANSIENT_AGENT_FAILURE_KINDS, CoalitionTransientFailureError

try:
    from demos.agentic_tool_use_explanation.scorers import split_coalition_prompt
except ModuleNotFoundError:
    from scorers import split_coalition_prompt

DEFAULT_FINAL_ANSWER_EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


class CoalitionSemanticFailureError(ValueError):
    """Raised when a coalition's agent run completed but produced no usable result.

    Examples: no final-answer text, or the backend reported an error. This is never
    retried by ``coalition_evaluation.classify_exception`` -- retrying would not help
    because the failure is not about infrastructure availability.
    """


def extract_final_answer(inference_result: object) -> str:
    """Return the user-visible final answer text from any backend's inference result.

    Different real-agent backends expose the final answer under different
    attribute names (``agent_response`` for the local HF router; ``assistant_answer``
    or ``final_answer`` for Groq and Gemini). This checks them in the same order the
    Inference tab uses to build the displayed assistant message, so the reference
    answer and every coalition answer are extracted consistently regardless of
    backend.
    """
    for attribute in ("agent_response", "assistant_answer", "final_answer"):
        value = getattr(inference_result, attribute, "")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return a numerically stable cosine similarity between two vectors."""
    left_vector = np.asarray(left, dtype=float)
    right_vector = np.asarray(right, dtype=float)
    denominator = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
    if denominator < 1e-12:
        return 0.0
    return float(np.dot(left_vector, right_vector) / denominator)


class SentenceTransformerAnswerEmbedder:
    """Lazily-loaded sentence-transformers embedder for final-answer texts.

    The model is only downloaded/loaded the first time the embedder is called
    (i.e. when this scorer is actually run), not at construction time, so
    selecting this scorer in the UI does not eagerly load anything.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_FINAL_ANSWER_EMBEDDING_MODEL_ID,
        device: str | int | None = "auto",
    ) -> None:
        self.model_id = model_id
        self.device = device
        self._model: object | None = None

    def _load(self) -> object:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            try:
                from demos.agentic_tool_use_explanation.semantic_segmenter import (
                    resolve_embedding_device,
                )
            except ModuleNotFoundError:
                from semantic_segmenter import resolve_embedding_device

            self._model = SentenceTransformer(
                self.model_id,
                device=resolve_embedding_device(self.device),
            )
        return self._model

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a batch of final-answer texts in one model call."""
        model = self._load()
        return model.encode(
            list(texts),
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )


@dataclass(frozen=True)
class CoalitionAnswer:
    """A successful complete-agent run for one coalition prompt.

    Only ever constructed for a coalition that actually produced usable
    final-answer text -- a failed or no-answer run raises
    :class:`CoalitionSemanticFailureError` (or lets the agent's own exception
    propagate) instead of being cached here, so a coalition's failure can never
    be silently embedded as if it were a real, just-very-dissimilar answer.
    """

    answer: str
    selected_tool: str | None = None


@dataclass
class FinalAnswerSimilarityScorer:
    """Coalition value function: semantic fidelity of a coalition's final answer.

    For each coalition prompt, this scorer:
      1. Reconstructs the masked user request from the coalition prompt
         (:func:`scorers.split_coalition_prompt`, the structural inverse of the
         canonical ``scorers.build_coalition_prompt`` used everywhere coalition
         prompts are built in this demo -- callers must not rebuild prompts by
         naively joining segment strings).
      2. Runs the complete tool-calling agent (``agent_callable``) on that masked
         request -- the same pipeline used by the Inference tab's "Run inference"
         action, including any tool execution -- to obtain a final
         natural-language answer.
      3. Embeds that answer (batched across all coalitions requested in one
         :meth:`score_batch` call) and compares it with the fixed full-request
         reference answer using cosine similarity.

    Raw score for one coalition prompt ``S``:
        ``r(S) = cosine_similarity(embed(y_S), embed(y_full))``
    Score returned by :meth:`score_batch` (normalized so the empty coalition
    always scores ``0.0``):
        ``v(S) = r(S) - r(empty)``

    Subtracting ``r(empty)`` only changes the zero point of the *displayed*
    coalition values to "semantic-fidelity gain above the empty request"; it
    does not change first-order Shapley values or any non-zero-order k-SII
    interaction attribution, since those are computed from *differences*
    between coalition values and a constant offset cancels out of every such
    difference. The raw (un-normalized) empty- and full-coalition similarities
    remain available via :attr:`last_empty_raw_similarity` and per-coalition
    debug output for diagnostics.

    Failure handling: if the agent raises, returns an explicit error, or produces
    no usable final-answer text, this coalition is never cached, never sent to
    the embedding model, and never assigned a placeholder score -- the agent's
    own exception propagates, or :class:`CoalitionSemanticFailureError` is raised for
    a no-answer/backend-error result. Callers (the coalition-evaluation
    orchestration boundary in ``coalition_evaluation.py``) are responsible for
    classifying and retrying transient failures; this scorer itself never
    retries and never substitutes a fallback float. The reference answer (the
    full request's answer) is validated at construction time and is never
    allowed to silently fall back; a missing/failed reference raises
    immediately.

    This is an end-to-end output-fidelity measure, not an objective correctness
    metric.
    """

    agent_callable: Callable[[str], object]
    embedder: Callable[[Sequence[str]], np.ndarray]
    reference_answer: str
    empty_prompt: str
    last_debug_outputs: list[dict[str, object]] = field(default_factory=list, init=False)
    last_empty_raw_similarity: float | None = field(default=None, init=False)
    _answer_cache: dict[str, CoalitionAnswer] = field(default_factory=dict, init=False, repr=False)
    _embedding_cache: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.reference_answer, str) or not self.reference_answer.strip():
            msg = (
                "FinalAnswerSimilarityScorer requires a non-empty reference_answer (the "
                "full request's final answer). Resolve and validate the full-run answer "
                "before constructing this scorer; do not pass a placeholder."
            )
            raise ValueError(msg)

    def score_batch(
        self,
        prompts: list[str],
        *,
        target_tool: str,
        tool_descriptions: dict[str, str],
    ) -> list[float]:
        """Return one empty-coalition-normalized similarity score per coalition prompt."""
        if target_tool not in tool_descriptions:
            msg = f"Target tool {target_tool!r} is not a known decision candidate."
            raise ValueError(msg)

        work_prompts = list(prompts)
        if self.empty_prompt not in work_prompts:
            work_prompts.append(self.empty_prompt)
        for prompt in work_prompts:
            self._ensure_answer(prompt)
        self._ensure_embeddings(work_prompts)

        empty_raw_similarity = self._raw_similarity(self.empty_prompt)
        self.last_empty_raw_similarity = empty_raw_similarity

        self.last_debug_outputs = []
        scores = []
        for prompt in prompts:
            raw_similarity = self._raw_similarity(prompt)
            normalized_score = raw_similarity - empty_raw_similarity
            scores.append(normalized_score)
            coalition_answer = self._answer_cache[prompt]
            self.last_debug_outputs.append(
                {
                    "target_tool": target_tool,
                    "masked_user_request": split_coalition_prompt(prompt)[1],
                    "selected_tool": coalition_answer.selected_tool,
                    "final_answer_preview": coalition_answer.answer[:240],
                    "raw_similarity": raw_similarity,
                    "normalized_score": normalized_score,
                    "execution_status": "ok",
                    "execution_error": None,
                    "final_score": normalized_score,
                    "prompt_preview": prompt[:240],
                }
            )
        return scores

    def _ensure_answer(self, prompt: str) -> None:
        """Run the complete agent for one coalition prompt, caching the successful result.

        Caches only successful outcomes, so a coalition is never re-run once it has
        produced a usable final answer. A failure is never cached and never
        converted into a placeholder score:

        * if the agent itself raises, that exception propagates unchanged, so
          transient-vs-semantic classification at the coalition-evaluation boundary
          sees the real exception type (e.g. ``groq.RateLimitError``);
        * if the agent instead returns a result object with ``.error`` set (the
          shape ``groq_agent``/``gemini_agent`` use to report a caught provider
          failure without raising), that result's ``failure_kind`` -- a structured
          :class:`agent_failure.AgentFailureKind`, not a string match against the
          error message -- decides whether this raises
          :class:`agent_failure.CoalitionTransientFailureError` (retryable) or
          :class:`CoalitionSemanticFailureError` (not retryable);
        * a missing final answer with no reported backend error raises
          :class:`CoalitionSemanticFailureError` directly.

        Either way, a failed attempt can be retried by simply calling this again --
        nothing is cached.
        """
        if prompt in self._answer_cache:
            return
        _system_prompt, user_request = split_coalition_prompt(prompt)
        inference_result = self.agent_callable(user_request)

        backend_error = getattr(inference_result, "error", None)
        selected_tool = getattr(inference_result, "selected_tool", None)
        answer = extract_final_answer(inference_result)
        if answer:
            self._answer_cache[prompt] = CoalitionAnswer(answer=answer, selected_tool=selected_tool)
            return
        if backend_error:
            failure_kind = getattr(inference_result, "failure_kind", None)
            msg = f"Agent backend reported an error for this coalition: {backend_error}"
            if failure_kind in TRANSIENT_AGENT_FAILURE_KINDS:
                raise CoalitionTransientFailureError(msg)
            raise CoalitionSemanticFailureError(msg)
        msg = "Agent produced no usable final-answer text for this coalition."
        raise CoalitionSemanticFailureError(msg)

    def _ensure_embeddings(self, prompts: Sequence[str]) -> None:
        """Embed every not-yet-cached successful answer text (and the reference) in one batch.

        By the time this runs, every prompt in ``prompts`` has a successful, cached
        :class:`CoalitionAnswer` -- :meth:`_ensure_answer` never caches a failure.
        """
        texts_to_embed: list[str] = []
        seen: set[str] = set()
        candidate_texts = [self.reference_answer]
        candidate_texts.extend(self._answer_cache[prompt].answer for prompt in prompts)
        for text in candidate_texts:
            if text not in self._embedding_cache and text not in seen:
                texts_to_embed.append(text)
                seen.add(text)
        if not texts_to_embed:
            return
        vectors = self.embedder(texts_to_embed)
        if len(vectors) != len(texts_to_embed):
            msg = "embedder must return one embedding vector per input text."
            raise ValueError(msg)
        for text, vector in zip(texts_to_embed, vectors, strict=True):
            self._embedding_cache[text] = np.asarray(vector, dtype=float)

    def _raw_similarity(self, prompt: str) -> float:
        """Return the raw cosine similarity between one coalition's answer and the reference."""
        coalition_answer = self._answer_cache[prompt]
        similarity = _cosine_similarity(
            self._embedding_cache[coalition_answer.answer],
            self._embedding_cache[self.reference_answer],
        )
        if not math.isfinite(similarity):
            msg = "Cosine similarity must be finite."
            raise ValueError(msg)
        return similarity
