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
    from typing import Literal

try:
    from demos.agentic_tool_use_explanation.scorers import split_coalition_prompt
except ModuleNotFoundError:
    from scorers import split_coalition_prompt

DEFAULT_FINAL_ANSWER_EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_FALLBACK_RAW_SCORE = 0.0


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
    """Structured outcome of running the complete agent for one coalition prompt.

    Replaces ad hoc placeholder strings (e.g. ``"[NO_FINAL_ANSWER]"``) with an
    explicit status so a coalition's failure is never silently embedded as if it
    were a real, just-very-dissimilar answer.
    """

    answer: str | None
    status: Literal["ok", "failed", "no_answer"]
    error_message: str | None = None
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

    Failure handling: if the agent raises, returns an explicit error, or
    produces no usable final-answer text, the coalition is recorded as a
    :class:`CoalitionAnswer` with ``status in {"failed", "no_answer"}`` and is
    *not* sent to the embedding model -- it is assigned ``fallback_raw_score``
    (default ``0.0``) directly. The reference answer (the full request's
    answer) is validated at construction time and is never allowed to silently
    fall back; a missing/failed reference raises immediately.

    This is an end-to-end output-fidelity measure, not an objective correctness
    metric.
    """

    agent_callable: Callable[[str], object]
    embedder: Callable[[Sequence[str]], np.ndarray]
    reference_answer: str
    empty_prompt: str
    fallback_raw_score: float = DEFAULT_FALLBACK_RAW_SCORE
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
                    "final_answer_preview": (
                        coalition_answer.answer[:240]
                        if coalition_answer.answer
                        else f"<{coalition_answer.status}>"
                    ),
                    "raw_similarity": raw_similarity,
                    "normalized_score": normalized_score,
                    "execution_status": coalition_answer.status,
                    "execution_error": coalition_answer.error_message,
                    "final_score": normalized_score,
                    "prompt_preview": prompt[:240],
                }
            )
        return scores

    def _ensure_answer(self, prompt: str) -> None:
        """Run the complete agent for one coalition prompt, caching the structured result.

        Caches both successful and failed/no-answer outcomes, so a coalition is never
        re-run within the lifetime of this scorer instance once its outcome is known.
        """
        if prompt in self._answer_cache:
            return
        _system_prompt, user_request = split_coalition_prompt(prompt)
        try:
            inference_result = self.agent_callable(user_request)
        except Exception as error:  # noqa: BLE001
            self._answer_cache[prompt] = CoalitionAnswer(
                answer=None,
                status="failed",
                error_message=str(error),
            )
            return

        backend_error = getattr(inference_result, "error", None)
        selected_tool = getattr(inference_result, "selected_tool", None)
        answer = extract_final_answer(inference_result)
        if answer:
            self._answer_cache[prompt] = CoalitionAnswer(
                answer=answer,
                status="ok",
                selected_tool=selected_tool,
            )
        elif backend_error:
            self._answer_cache[prompt] = CoalitionAnswer(
                answer=None,
                status="failed",
                error_message=str(backend_error),
                selected_tool=selected_tool,
            )
        else:
            self._answer_cache[prompt] = CoalitionAnswer(
                answer=None,
                status="no_answer",
                selected_tool=selected_tool,
            )

    def _ensure_embeddings(self, prompts: Sequence[str]) -> None:
        """Embed every not-yet-cached *successful* answer text (and the reference) in one batch.

        Failed/no-answer coalitions never reach the embedder; their raw score is
        ``fallback_raw_score`` instead (see :meth:`_raw_similarity`).
        """
        texts_to_embed: list[str] = []
        seen: set[str] = set()
        candidate_texts = [self.reference_answer]
        for prompt in prompts:
            coalition_answer = self._answer_cache[prompt]
            if coalition_answer.status == "ok" and coalition_answer.answer is not None:
                candidate_texts.append(coalition_answer.answer)
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
        """Return the raw cosine similarity for one coalition, or the configured fallback.

        Coalitions whose agent run failed or produced no final answer never reach the
        embedding model; they receive ``fallback_raw_score`` directly.
        """
        coalition_answer = self._answer_cache[prompt]
        if coalition_answer.status != "ok" or coalition_answer.answer is None:
            return self.fallback_raw_score
        similarity = _cosine_similarity(
            self._embedding_cache[coalition_answer.answer],
            self._embedding_cache[self.reference_answer],
        )
        if not math.isfinite(similarity):
            msg = "Cosine similarity must be finite."
            raise ValueError(msg)
        return similarity
