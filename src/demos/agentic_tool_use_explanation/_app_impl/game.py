"""Agent execution and full-run reference-answer helpers."""

# ruff: noqa: F405

from __future__ import annotations

# Mechanical re-export chain preserves the monolith's shared global namespace.
from .plotting import *  # noqa: F403


def build_complete_agent_callable(
    *,
    inference_backend: str,
    inference_model_name: str,
    system_prompt: str,
    tool_context: str,
    hf_model_config: SelectedHFModelConfig | None = None,
    hf_max_new_tokens: int = DEFAULT_NATIVE_HF_MAX_NEW_TOKENS,
    hf_trust_remote_code: bool = False,
    calibrated_hf_mode: bool = False,
    logprob_model_id: str = DEFAULT_LOGPROB_MODEL_ID,
    logprob_max_pairs_per_batch: int = 1,
) -> Callable[[str], object]:
    """Build a backend-agnostic callable that runs the complete tool-calling agent.

    Mirrors the Inference tab's "Run inference" action (router/tool-choice -> tool
    execution -> final answer) exactly, but parameterized over the user request so
    it can be re-run once per Shapley coalition by
    ``final_answer_similarity_scorer.FinalAnswerSimilarityScorer``.

    HF Local uses model-native structured tool schemas through the tokenizer's
    chat template. The selected HF model id is always the single
    ``HF_MODEL_ID_SESSION_KEY`` dropdown selection from the Inference tab.
    """
    if inference_backend == "HF local":
        hf_model_config = hf_model_config or selected_hf_model_config(inference_model_name)

    def run(user_request: str) -> object:
        if inference_backend == "Groq":
            return run_groq_tool_inference(
                user_request,
                get_executable_tool_schemas(),
                inference_model_name,
                system_prompt=system_prompt,
                tool_context=tool_context,
            )
        if inference_backend == "Gemini":
            return run_gemini_tool_inference(
                user_request,
                get_executable_tool_schemas(),
                inference_model_name,
                system_prompt=system_prompt,
                tool_context=tool_context,
            )
        if calibrated_hf_mode:
            try:
                primary_scorer = load_logprob_scorer(
                    logprob_model_id,
                    max_pairs_per_batch=logprob_max_pairs_per_batch,
                    device=hf_model_config.device if hf_model_config is not None else None,
                    dtype=hf_model_config.dtype if hf_model_config is not None else "auto",
                    quantization_mode=(
                        hf_model_config.quantization_mode if hf_model_config is not None else "none"
                    ),
                    tokenizer_id=(
                        hf_model_config.tokenizer_id if hf_model_config is not None else None
                    ),
                )
                classification_router = LocalHFClassificationRouter(primary_scorer)
                _hf_lifecycle_log(
                    "[HF ROUTING] LocalHFClassificationRouter reusing scorer instance",
                    f"model_id={logprob_model_id}",
                )
                target_choice = classification_router.choose_tool(
                    user_request,
                    system_prompt=system_prompt,
                    tool_descriptions=TOOLS,
                )
                argument_extractor = HFArgumentExtractor(
                    model=primary_scorer.model,
                    tokenizer=primary_scorer.tokenizer,
                    device=primary_scorer.device,
                )
                scorer_tokenizer_lock = getattr(primary_scorer, "tokenizer_lock", None)
                if scorer_tokenizer_lock is not None:
                    argument_extractor.tokenizer_lock = scorer_tokenizer_lock
                _hf_lifecycle_log(
                    "[HF ARGUMENTS] HFArgumentExtractor reusing scorer model/tokenizer",
                    f"model_id={logprob_model_id}",
                )
                tool_arguments = argument_extractor.extract_arguments(
                    user_request=user_request,
                    selected_tool=target_choice.tool,
                    tool_descriptions=TOOLS,
                )
                agent_response = (
                    f"I would use {target_choice.tool} for this request (calibrated "
                    "A/B/C/D routing evidence, same model instance as the coalition scorer)."
                )
                return types.SimpleNamespace(
                    selected_tool=target_choice.tool,
                    tool_arguments=tool_arguments,
                    agent_response=agent_response,
                    raw_response=f"calibrated_scores={target_choice.scores}",
                    debug_prompt=None,
                    error=None,
                    available=True,
                    calibrated_hf_mode=True,
                    # Display-only: the router's raw calibrated log-odds per tool, exposed so
                    # the UI can render a presentation-level probability distribution (via a
                    # local softmax) without re-deriving it from scorer internals.
                    tool_scores=dict(target_choice.scores),
                )
            except Exception as error:  # noqa: BLE001
                return types.SimpleNamespace(
                    selected_tool=None,
                    tool_arguments={},
                    agent_response="",
                    raw_response="",
                    debug_prompt=None,
                    error=f"HF calibrated inference failed: {error}",
                    available=False,
                )
        try:
            _hf_lifecycle_log(
                "[HF LOAD] using native LocalHFRouter", f"model_name={inference_model_name}"
            )
            hf_router = load_local_hf_router(
                inference_model_name,
                int(hf_max_new_tokens),
                trust_remote_code=bool(hf_trust_remote_code),
                quantization_mode=(
                    hf_model_config.quantization_mode if hf_model_config is not None else "none"
                ),
                device=hf_model_config.device if hf_model_config is not None else "auto",
                dtype=hf_model_config.dtype if hf_model_config is not None else "auto",
            )
            try:
                return hf_router.choose_tool(user_request, TOOLS, system_prompt=system_prompt)
            except TypeError as error:
                if "system_prompt" not in str(error):
                    raise
                return hf_router.choose_tool(user_request, TOOLS)
        except Exception as error:  # noqa: BLE001
            return types.SimpleNamespace(
                selected_tool=None,
                tool_arguments={},
                agent_response="",
                raw_response="",
                debug_prompt=None,
                error=f"HF local inference failed: {error}",
                available=False,
            )

    return run


def resolve_full_run_reference_answer(
    *,
    agent_callable: Callable[[str], object],
    user_request: str,
    inference_result: object | None,
    inference_result_is_current: bool,
) -> tuple[str, bool, str | None]:
    """Return (reference final answer, whether an existing result was reused, error reason).

    Reuses the answer already produced by the normal "Run inference" action when it
    matches the current request, system/tool configuration, and backend/model
    configuration. Otherwise runs the full prompt once through the same agent
    pipeline and uses that answer as the reference. The full-run reference answer
    must never be missing or a placeholder: if it cannot be obtained, the third
    return value carries a concrete, user-actionable reason so the caller can fail
    the explanation clearly instead of silently computing meaningless scores.
    """
    if (
        inference_result_is_current
        and inference_result is not None
        and not getattr(inference_result, "error", None)
    ):
        answer = extract_final_answer(inference_result)
        if answer:
            return answer, True, None

    try:
        full_run_result = agent_callable(user_request)
    except Exception as error:  # noqa: BLE001
        return "", False, f"The agent raised an error while answering the full request: {error}"

    backend_error = getattr(full_run_result, "error", None)
    answer = extract_final_answer(full_run_result)
    if answer:
        return answer, False, None
    if backend_error:
        return "", False, str(backend_error)
    return "", False, "The agent returned no final-answer text for the full request."


__all__ = [name for name in globals() if not name.startswith("__")]
