"""Main Streamlit user interface."""

# ruff: noqa: F405, PLC0415

from __future__ import annotations

import datetime
import uuid

from persistence import build_pairwise_interactions, write_result_export_safely

# Mechanical re-export chain preserves the monolith's shared global namespace.
from .cards import *  # noqa: F403

MAIN_TAB_LABELS = ("1. Agent Result", "2. XAI Explanation")


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Header: logo + title + subtitle (left), developer mode toggle (right).
    # The toggle is a real Streamlit widget, so it cannot live inside the
    # hand-authored HTML on the left -- st.columns keeps them on one visual
    # row instead.
    # ------------------------------------------------------------------
    header_left, header_right = st.columns([5, 1], vertical_alignment="center")
    with header_left:
        st.markdown(
            """
            <div class="app-header">
                <div class="app-logo">T</div>
                <div>
                    <h1>Agentic Tool-Use Explanation</h1>
                    <p>Explain why an agent selected a tool by attributing the decision to
                    user-request segments.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with header_right:
        developer_mode = st.toggle(
            "Developer mode",
            value=False,
            key="agentic_developer_mode",
            help="Show segmentation, scorer, and raw-diagnostic controls for debugging.",
        )
    if ROBOT_IMAGE_PATH.exists():
        robot_base64 = _image_to_base64(ROBOT_IMAGE_PATH)
        st.markdown(
            f"""
            <div class="floating-robot-wrap">
                <img
                    src="data:image/png;base64,{robot_base64}"
                    class="floating-robot"
                    title="Image from Canva"
                    alt="Decorative robot illustration"
                />
            </div>
            """,
            unsafe_allow_html=True,
        )

    if "has_run" not in st.session_state:
        st.session_state.has_run = False
    if "result" not in st.session_state:
        st.session_state.result = None
    if "pending_run" not in st.session_state:
        st.session_state.pending_run = False
    if "agentic_inferred_tool" not in st.session_state:
        st.session_state["agentic_inferred_tool"] = None
    if "agentic_inference_result" not in st.session_state:
        st.session_state["agentic_inference_result"] = None
    if "agentic_inference_signature" not in st.session_state:
        st.session_state["agentic_inference_signature"] = None
    if "agentic_request_text" not in st.session_state:
        st.session_state["agentic_request_text"] = DEFAULT_MOCK_QUERY
    if "agentic_export_session_id" not in st.session_state:
        st.session_state["agentic_export_session_id"] = str(uuid.uuid4())
        st.session_state["agentic_export_session_started_at"] = datetime.datetime.now(
            tz=datetime.UTC
        )

    example_placeholder = "Choose an example..."
    pending_example_request = st.session_state.pop("agentic_pending_example_request", None)
    if pending_example_request is not None:
        st.session_state["agentic_request_text"] = pending_example_request
        st.session_state["agentic_inferred_tool"] = None
        st.session_state["agentic_inference_result"] = None
        st.session_state.has_run = False
        st.session_state.result = None
        st.session_state.result_signature = None
        st.session_state["agentic_try_example_select"] = example_placeholder

    trace_name = "Custom request"

    # ------------------------------------------------------------------
    # Mode selector: HF Local vs. API Agent, styled as two pill/cards.
    # st.container(key=...) is the only way to get a real, stable DOM
    # wrapper around a widget (a hand-written <div> in st.markdown would
    # not actually nest the radio inside it -- each Streamlit call renders
    # as its own sibling element), so the pill/card CSS targets the
    # `st-key-agentic_mode_card_row` class Streamlit derives from `key=`.
    # ------------------------------------------------------------------
    with st.container(key="agentic_mode_card_row"):
        explanation_mode = st.radio(
            "Explanation mode",
            ["HF Local", "API Agent"],
            captions=[
                "Model-internal routing evidence",
                "Black-box Groq tool-use trajectory",
            ],
            horizontal=True,
            key="agentic_explanation_mode",
            label_visibility="collapsed",
        )

    inference_backend = "HF local" if explanation_mode == "HF Local" else "Groq"

    if inference_backend == "HF local":
        initialize_hf_model_session_state(st.session_state)
        inference_model_name = st.selectbox(
            "HF model",
            HF_LOCAL_MODEL_OPTIONS,
            key=HF_MODEL_ID_SESSION_KEY,
            help=(
                "Local HF model used for both native tool-call inference and explanation scorer."
            ),
        )
        selected_hf_config = selected_hf_model_config(inference_model_name)
    else:
        selected_hf_config = None
        inference_model_name = st.text_input(
            "Groq model",
            value="llama-3.1-8b-instant",
            key="agentic_groq_inference_model",
        )
        if not os.getenv("GROQ_API_KEY"):
            st.warning("GROQ_API_KEY is not set. Add it to run Groq inference.")

    # ------------------------------------------------------------------
    # Shared input bar: request + example + run button, grouped into one
    # bordered/shadowed container (see the mode-selector comment above for
    # why st.container(key=...) is what actually makes this one visual bar).
    # ------------------------------------------------------------------
    def apply_selected_example() -> None:
        selected = st.session_state.get("agentic_try_example_select")
        if selected and selected != example_placeholder:
            example_request = " ".join(SAMPLE_TRACES[selected]["user_segments"])
            st.session_state["agentic_pending_example_request"] = example_request

    def format_try_example_option(name: str) -> str:
        if name == example_placeholder:
            return example_placeholder
        return " ".join(SAMPLE_TRACES[name]["user_segments"])

    example_options = [example_placeholder, *list(SAMPLE_TRACES)]

    with st.container(key="agentic_input_bar"):
        input_col, button_col = st.columns([4.3, 1.4], vertical_alignment="center")
        with input_col:
            st.markdown('<div class="input-bar-label">User request</div>', unsafe_allow_html=True)
            st.text_area(
                "User request",
                height=96,
                key="agentic_request_text",
                label_visibility="collapsed",
                help=(
                    "This preview chooses a tool from the fixed context and request. "
                    "It does not call the selected tool."
                ),
            )
            st.markdown('<div class="input-bar-label">Try example</div>', unsafe_allow_html=True)
            st.selectbox(
                "Try example",
                example_options,
                format_func=format_try_example_option,
                key="agentic_try_example_select",
                on_change=apply_selected_example,
                label_visibility="collapsed",
            )
        with button_col:
            run_full_pipeline_clicked = st.button(
                "Run full pipeline",
                type="primary",
                key="agentic_run_full_pipeline",
                use_container_width=True,
                help="Runs the agent, then prepares the shapiq explanation for the selected tool.",
            )
        user_request = st.session_state["agentic_request_text"]
        trace = build_mock_trace(user_request)
        system_segments = build_segments(trace["system_segments"], "system")
        system_prompt = build_system_prompt(system_segments)
        tool_context = format_tool_context(TOOLS)
    if st.session_state.get("agentic_pending_example_request") is not None:
        st.rerun()

    # Includes inference_backend/inference_model_name/explanation_mode so that switching
    # the HF model (e.g. 1.5B -> 3B) or the mode itself invalidates the previous agent
    # result exactly like changing the request text does -- otherwise a stale
    # `agentic_inferred_tool` from the old model can survive into the next run.
    current_inference_signature = (
        user_request,
        system_prompt,
        tool_context,
        inference_backend,
        inference_model_name,
        selected_hf_config.cache_key(scorer_mode="inference")
        if selected_hf_config is not None
        else None,
        explanation_mode,
    )
    if st.session_state.get("agentic_inference_signature") != current_inference_signature:
        st.session_state["agentic_inference_signature"] = current_inference_signature
        st.session_state["agentic_inferred_tool"] = None
        st.session_state["agentic_inference_result"] = None
        st.session_state["agentic_inference_backend"] = None
        st.session_state["agentic_inference_model"] = None
        st.session_state.has_run = False
        st.session_state.result = None
        st.session_state.result_signature = None

    # ------------------------------------------------------------------
    # 6. Developer mode: grouped controls (hidden entirely when OFF)
    # ------------------------------------------------------------------
    latest_inference_backend = st.session_state.get("agentic_inference_backend")
    has_groq_inference_result = (
        latest_inference_backend == "Groq"
        and st.session_state.get("agentic_inference_result") is not None
    )
    groq_reference_result = (
        st.session_state.get("agentic_inference_result") if has_groq_inference_result else None
    )
    trajectory_match_available = (
        groq_reference_result is not None
        and st.session_state.get("agentic_inferred_tool") in TOOLS
        and bool(getattr(groq_reference_result, "tool_arguments", None))
    )

    scorer_backend_key = SCORER_BACKEND_SESSION_KEY
    # Always the current HF model selection, regardless of Developer mode: this is what
    # the LOGPROB_SCORER_LABEL explanation scorer loads, and it must never silently stay
    # on a stale default (e.g. after the user switches HF models with Developer mode
    # off) -- otherwise the explanation step's classifier can disagree with the agent
    # step's, which used `inference_model_name` directly.
    logprob_model_id = (
        selected_hf_config.model_id
        if selected_hf_config is not None
        else st.session_state.get(HF_MODEL_ID_SESSION_KEY, DEFAULT_LOGPROB_MODEL_ID)
    )
    max_pairs_per_batch = 1
    router_model_id = DEFAULT_GROQ_ROUTER_MODEL_ID
    soft_vote_model_id = DEFAULT_GROQ_ROUTER_MODEL_ID
    soft_vote_n_samples = DEFAULT_GROQ_SOFT_VOTE_N_SAMPLES
    soft_vote_temperature = DEFAULT_GROQ_SOFT_VOTE_TEMPERATURE
    soft_vote_max_retries = DEFAULT_GROQ_SOFT_VOTE_MAX_RETRIES
    soft_vote_seed = None
    final_answer_embedding_model_id = DEFAULT_FINAL_ANSWER_EMBEDDING_MODEL_ID
    show_lexical_comparison = False
    enable_fallback_target_selection = False
    segmenter_choice = "Linguistic (spaCy chunking)"
    segment_threshold = 0.72
    segment_window = 3
    min_segment_words = 1
    show_value_function_details = False
    show_scoring_prompt_preview = False

    forced_default_scorer = (
        NATIVE_HF_SCORER_LABEL
        if inference_backend == "HF local"
        else ("Groq soft-vote scorer" if has_groq_inference_result else NATIVE_HF_SCORER_LABEL)
    )

    if not developer_mode:
        st.session_state[scorer_backend_key] = forced_default_scorer
        scorer_backend = forced_default_scorer
    else:
        st.markdown('<div class="section-label">Developer settings</div>', unsafe_allow_html=True)
        with st.expander("Segmentation", expanded=False):
            segmenter_choice = st.selectbox(
                "Segmenter",
                [
                    "Embedding (semantic similarity)",
                    "Linguistic (spaCy chunking)",
                ],
                index=1,
                key="agentic_segmenter",
            )
            segment_threshold = st.slider(
                "semantic threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.72,
                step=0.01,
                key="agentic_segment_threshold",
            )
            segment_window = st.slider(
                "context window",
                min_value=1,
                max_value=10,
                value=3,
                step=1,
                key="agentic_segment_window",
            )
            min_segment_words = st.slider(
                "min words per segment",
                min_value=1,
                max_value=8,
                value=1,
                step=1,
                key="agentic_min_segment_words",
            )

        with st.expander("Value function / scorer diagnostics", expanded=False):
            show_developer_scorers = st.checkbox(
                "Show developer scoring methods",
                value=False,
                key="agentic_show_developer_scorers",
            )
            scorer_options = [NATIVE_HF_SCORER_LABEL, FINAL_ANSWER_SIMILARITY_LABEL]
            if inference_backend == "Groq" or has_groq_inference_result:
                scorer_options.append("Groq soft-vote scorer")
                scorer_options.append("Groq deterministic router")
            if trajectory_match_available:
                scorer_options.append("Trajectory match: tool + normalized args")
            if show_developer_scorers:
                st.caption(
                    "Developer scorers are intended for debugging and should not be used for "
                    "final demo results."
                )
                scorer_options.extend([LOGPROB_SCORER_LABEL, "Keyword scorer", "Mock model scorer"])
            if scorer_backend_key not in st.session_state:
                st.session_state[scorer_backend_key] = forced_default_scorer
            if st.session_state[scorer_backend_key] not in scorer_options:
                st.session_state[scorer_backend_key] = forced_default_scorer
            scorer_backend = st.selectbox(
                "Scoring method",
                scorer_options,
                key=scorer_backend_key,
            )
            if scorer_backend == NATIVE_HF_SCORER_LABEL:
                # `logprob_model_id` is already set above from HF_MODEL_ID_SESSION_KEY,
                # unconditional on Developer mode -- just display it here.
                st.caption(f"Inherited from HF model selection: `{logprob_model_id}`")
                st.caption(SAME_HF_MODEL_EXPLANATION)
                max_pairs_per_batch = st.number_input(
                    "HF continuation batch size",
                    min_value=1,
                    max_value=16,
                    value=1,
                    step=1,
                    key="agentic_logprob_pair_batch_size",
                    help=(
                        "Number of prompt/continuation pairs scored per local-model forward pass. "
                        "Use 1 on Colab T4 to avoid CUDA out-of-memory errors."
                    ),
                )
                st.caption(NATIVE_HF_SCORER_HELP)
            elif scorer_backend == LOGPROB_SCORER_LABEL:
                # A/B/C/D ablation. LocalHFClassificationRouter may be used only by
                # explicit developer-mode inference, never by the primary native scorer.
                st.caption(f"Inherited from HF model selection: `{logprob_model_id}`")
                max_pairs_per_batch = st.number_input(
                    "HF A/B/C/D pair batch size",
                    min_value=1,
                    max_value=16,
                    value=1,
                    step=1,
                    key="agentic_logprob_pair_batch_size",
                )
                st.caption(LOGPROB_SCORER_HELP)
            elif scorer_backend == FINAL_ANSWER_SIMILARITY_LABEL:
                final_answer_embedding_model_id = st.text_input(
                    "Embedding model",
                    value=DEFAULT_FINAL_ANSWER_EMBEDDING_MODEL_ID,
                    key="agentic_final_answer_embedding_model_id",
                    help=(
                        "A sentence-transformers model used only to embed final answers for "
                        "this scorer. Loaded lazily the first time you run this scorer, not "
                        "at app startup."
                    ),
                )
                if inference_backend == "Groq" and not os.getenv("GROQ_API_KEY"):
                    st.warning("GROQ_API_KEY is not set. Add it to use this scorer with Groq.")
            elif scorer_backend == "Groq deterministic router":
                router_model_id = st.text_input(
                    "Groq router model",
                    value=DEFAULT_GROQ_ROUTER_MODEL_ID,
                    key="agentic_groq_router_scorer_model",
                )
                st.caption(
                    "Calls the real Groq API once per distinct coalition prompt to ask which "
                    "tool it would route to, and scores 1.0 if that matches the target tool, "
                    "else 0.0."
                )
                if not os.getenv("GROQ_API_KEY"):
                    st.warning("GROQ_API_KEY is not set. Add it to use the Groq router scorer.")
            elif scorer_backend == "Groq soft-vote scorer":
                soft_vote_model_id = st.text_input(
                    "Groq soft-vote model",
                    value=DEFAULT_GROQ_ROUTER_MODEL_ID,
                    key="agentic_groq_soft_vote_model",
                )
                soft_vote_n_samples = st.number_input(
                    "Groq soft-vote samples",
                    min_value=1,
                    max_value=25,
                    value=DEFAULT_GROQ_SOFT_VOTE_N_SAMPLES,
                    step=1,
                    key="agentic_groq_soft_vote_samples",
                )
                soft_vote_temperature = st.slider(
                    "Groq soft-vote temperature",
                    min_value=0.0,
                    max_value=1.5,
                    value=DEFAULT_GROQ_SOFT_VOTE_TEMPERATURE,
                    step=0.05,
                    key="agentic_groq_soft_vote_temperature",
                )
                soft_vote_max_retries = st.number_input(
                    "Groq soft-vote max retries",
                    min_value=0,
                    max_value=5,
                    value=DEFAULT_GROQ_SOFT_VOTE_MAX_RETRIES,
                    step=1,
                    key="agentic_groq_soft_vote_max_retries",
                )
                use_soft_vote_seed = st.checkbox(
                    "set Groq soft-vote seed",
                    value=False,
                    key="agentic_groq_soft_vote_use_seed",
                )
                if use_soft_vote_seed:
                    soft_vote_seed = int(
                        st.number_input(
                            "Groq soft-vote seed",
                            min_value=0,
                            max_value=2_147_483_647,
                            value=42,
                            step=1,
                            key="agentic_groq_soft_vote_seed",
                        )
                    )
                st.caption(
                    "Soft-vote score: empirical target-tool selection frequency across sampled "
                    "Groq router calls."
                )
                if not os.getenv("GROQ_API_KEY"):
                    st.warning("GROQ_API_KEY is not set. Add it to use the Groq soft-vote scorer.")
            elif scorer_backend == "Trajectory match: tool + normalized args":
                st.warning(
                    "This scorer re-runs the real Groq agent once per distinct coalition "
                    "prompt to get an actual tool call (name + arguments), not just a routing "
                    "decision, then compares it against the recorded inference result. This "
                    "can be slow and costly: expect one real Groq API call per coalition "
                    "shapiq samples."
                )
                if groq_reference_result is not None:
                    st.caption(
                        f"Reference tool: `{groq_reference_result.selected_tool}` with "
                        f"arguments {dict(groq_reference_result.tool_arguments)}."
                    )
                if not os.getenv("GROQ_API_KEY"):
                    st.warning(
                        "GROQ_API_KEY is not set. Add it to use the trajectory match scorer."
                    )

            show_lexical_comparison = st.checkbox(
                "show keyword comparison",
                value=False,
                key="agentic_show_lexical_comparison",
            )
            enable_fallback_target_selection = st.checkbox(
                "enable fallback target selection",
                value=False,
                key="agentic_enable_fallback_target_selection",
                help=(
                    "Use the selected explanation scorer to choose a target tool when no "
                    "inference result is available."
                ),
            )

        with st.expander("Computation diagnostics", expanded=False):
            st.caption(f"Individual effects: `{SV_INDEX}`")
            st.caption(f"Pairwise interactions: `{KSII_INDEX}`, max_order=`{KSII_MAX_ORDER}`")
            st.caption("Players: user-request segments")
            st.caption("Fixed context: system prompt + tool definitions")
            st.caption("Value function: selected-tool support under the chosen scorer")
            show_value_function_details = st.checkbox(
                "show value function details",
                value=False,
                key="agentic_show_value_function_details",
            )
            show_scoring_prompt_preview = st.checkbox(
                "show scoring prompt preview",
                value=False,
                key="agentic_show_scoring_prompt_preview",
            )

    def execute_tool_inference() -> object:
        calibrated_hf_mode = (
            inference_backend == "HF local"
            and st.session_state.get(SCORER_BACKEND_SESSION_KEY) == LOGPROB_SCORER_LABEL
        )
        agent_callable = build_complete_agent_callable(
            inference_backend=inference_backend,
            inference_model_name=inference_model_name,
            system_prompt=system_prompt,
            tool_context=tool_context,
            hf_model_config=selected_hf_config,
            hf_max_new_tokens=DEFAULT_NATIVE_HF_MAX_NEW_TOKENS,
            hf_trust_remote_code=False,
            calibrated_hf_mode=calibrated_hf_mode,
            logprob_model_id=inference_model_name
            if calibrated_hf_mode
            else DEFAULT_LOGPROB_MODEL_ID,
            logprob_max_pairs_per_batch=int(
                st.session_state.get("agentic_logprob_pair_batch_size", 1)
            ),
        )
        return agent_callable(user_request)

    if run_full_pipeline_clicked:
        if hasattr(st, "status"):
            with st.status("Running agent...", expanded=True) as status:
                st.write(f"Mode: {explanation_mode}")
                st.write(f"Model: {inference_model_name}")
                inference_result = execute_tool_inference()
                status.update(
                    label="Agent step complete. Preparing explanation...", state="complete"
                )
        else:
            with st.spinner("Running agent..."):
                inference_result = execute_tool_inference()
        inference_result.backend = inference_backend
        inference_result.model = inference_model_name
        st.session_state["agentic_inference_backend"] = inference_backend
        st.session_state["agentic_inference_model"] = inference_model_name
        st.session_state["agentic_inference_result"] = inference_result
        st.session_state["agentic_inferred_tool"] = (
            inference_result.selected_tool if inference_result.selected_tool in TOOLS else None
        )
        # Hard-reset any previous explanation before computing the new one: a fresh
        # agent result must never be paired with a stale XAI result computed for a
        # different backend/model/tool (e.g. after switching the HF model).
        st.session_state.has_run = False
        st.session_state.result = None
        st.session_state.result_signature = None
        st.session_state.pending_run = not bool(getattr(inference_result, "parse_error", None))
        st.rerun()

    # ------------------------------------------------------------------
    # 3. Two tabs, feeling like two stages of one pipeline
    # ------------------------------------------------------------------
    agent_tab, xai_tab = st.tabs(MAIN_TAB_LABELS)

    # ------------------------------------------------------------------
    # 4. Tab 1: Agent Result
    # ------------------------------------------------------------------
    with agent_tab:
        inference_result = st.session_state.get("agentic_inference_result")

        if inference_result is None:
            st.markdown(
                """
                <div class="empty-state-card">
                    <h3>Run the agent first</h3>
                    <p>Click <strong>Run full pipeline</strong> to select a tool and prepare
                    the explanation.</p>
                    <div class="pipeline-hint">User request &rarr; tool decision &rarr;
                    XAI explanation</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            inference_error = getattr(inference_result, "error", None)
            parse_error = getattr(inference_result, "parse_error", None)
            selected_tool = getattr(inference_result, "selected_tool", None)
            result_backend = getattr(inference_result, "backend", inference_backend)
            result_model = getattr(inference_result, "model", inference_model_name)

            if inference_error:
                st.markdown(
                    f"""
                    <div class="error-card">
                        <h3>Agent run failed</h3>
                        <p>{escape(str(inference_error))}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    render_agent_result_card(
                        inference_result,
                        backend=result_backend,
                        model=result_model,
                    ),
                    unsafe_allow_html=True,
                )

            if developer_mode:
                with st.expander("Developer diagnostics", expanded=False):
                    st.write(f"Backend: `{result_backend}`")
                    st.write(f"Model: `{result_model}`")
                    st.write(f"Internal route label: `{selected_tool}`")
                    st.write(f"Parser status: `{'error' if parse_error else 'ok'}`")
                    st.markdown("**Raw tool arguments**")
                    st.json(getattr(inference_result, "tool_arguments", {}))
                    raw_trace = getattr(inference_result, "raw_trace", None)
                    if raw_trace is not None:
                        st.markdown("**Raw trace**")
                        st.json(raw_trace)
                    debug_prompt = getattr(inference_result, "debug_prompt", None)
                    if debug_prompt is not None:
                        st.markdown("**Debug prompt**")
                        st.code(str(debug_prompt), language="text")
                    raw_response = getattr(inference_result, "raw_response", None)
                    if raw_response:
                        st.markdown("**Raw response**")
                        st.code(str(raw_response), language="text")
                    normalized_response = getattr(
                        inference_result,
                        "normalized_response_used_for_parsing",
                        None,
                    )
                    if normalized_response:
                        st.markdown("**Normalized response used for parsing**")
                        st.code(str(normalized_response), language="text")
                    extracted_tool_call_json = getattr(
                        inference_result,
                        "extracted_tool_call_json",
                        None,
                    )
                    if extracted_tool_call_json:
                        st.markdown("**Extracted tool-call JSON**")
                        st.code(str(extracted_tool_call_json), language="json")
                    cleaned_direct_answer = getattr(
                        inference_result,
                        "cleaned_direct_answer",
                        None,
                    )
                    if cleaned_direct_answer:
                        st.markdown("**Cleaned direct answer**")
                        st.code(str(cleaned_direct_answer), language="text")
                    st.write(
                        "Removed direct-answer sentinel: "
                        f"`{getattr(inference_result, 'removed_direct_answer_sentinel', False)}`"
                    )
                    generation_parameters = getattr(
                        inference_result,
                        "generation_parameters",
                        None,
                    )
                    if generation_parameters:
                        st.markdown("**Generation parameters**")
                        st.json(generation_parameters)

    # ------------------------------------------------------------------
    # 5. Tab 2: XAI Explanation
    # ------------------------------------------------------------------
    with xai_tab:
        current_inference_result = st.session_state.get("agentic_inference_result")
        if has_untrusted_native_parse_result(current_inference_result):
            st.error(
                "Model output contained a tool-call structure, but it could not be "
                "parsed safely. The result is not treated as no_tool, and XAI is "
                "blocked because there is no trustworthy target tool."
            )
            return
        try:
            if segmenter_choice == "Linguistic (spaCy chunking)":
                segmenter = load_linguistic_segmenter()
            else:
                segmenter = load_semantic_segmenter(
                    segment_threshold,
                    segment_window,
                    min_segment_words,
                )
            semantic_user_texts, segment_debug_rows = segment_user_request(
                segmenter,
                user_request,
            )
        except Exception as error:  # noqa: BLE001
            st.error(f"Could not segment the user request with {segmenter_choice}: {error}")
            return
        user_segments = build_segments(semantic_user_texts, "user")
        labels = [segment.label for segment in user_segments]
        using_exact_computation = len(user_segments) <= MAX_EXACT_DEMO_PLAYERS
        budget = budget_for_demo(len(user_segments)) if not using_exact_computation else None

        if len(user_segments) < 1:
            st.warning("Add a user request with at least one segment.")
            return

        full_prompt = build_coalition_prompt(
            user_segments,
            system_prompt=system_prompt,
            tool_context=tool_context,
        )
        empty_prompt = build_coalition_prompt(
            [],
            system_prompt=system_prompt,
            tool_context=tool_context,
        )

        inferred_tool = st.session_state.get("agentic_inferred_tool")
        using_inferred_tool = inferred_tool in TOOLS
        result = st.session_state.result
        result_target_tool = None
        result_target_source = None
        if isinstance(result, dict):
            result_target_tool = result.get("target_tool")
            result_target_source = result.get("target_source")
        if using_inferred_tool:
            target_tool = inferred_tool
            target_source = "Agent Result"
        elif result_target_tool in TOOLS and result_target_source == "fallback explanation scorer":
            target_tool = str(result_target_tool)
            target_source = "fallback explanation scorer"
        else:
            target_tool = None
            target_source = "fallback explanation scorer"

        if not using_inferred_tool and target_tool is None:
            if enable_fallback_target_selection:
                st.info(
                    "No agent result is available. Fallback target selection is enabled; "
                    "the explanation scorer will choose the target when the pipeline runs."
                )
            else:
                st.info(
                    "Click **Run full pipeline** above to run the agent and explain its decision."
                )

        signature_target = target_tool if target_tool is not None else "__pending_target__"
        trajectory_reference_signature = (
            tuple(sorted(groq_reference_result.tool_arguments.items()))
            if scorer_backend == "Trajectory match: tool + normalized args"
            and groq_reference_result is not None
            else None
        )
        signature = (
            trace_name,
            inference_backend,
            inference_model_name,
            selected_hf_config.cache_key(scorer_mode=str(scorer_backend))
            if selected_hf_config is not None
            else None,
            user_request,
            signature_target,
            scorer_backend,
            logprob_model_id,
            int(max_pairs_per_batch),
            router_model_id,
            soft_vote_model_id,
            int(soft_vote_n_samples),
            float(soft_vote_temperature),
            int(soft_vote_max_retries),
            soft_vote_seed,
            trajectory_reference_signature,
            final_answer_embedding_model_id,
            bool(enable_fallback_target_selection),
            show_lexical_comparison,
            segmenter_choice,
            segment_threshold,
            segment_window,
            min_segment_words,
            tuple(semantic_user_texts),
        )
        if st.session_state.get("result_signature") != signature:
            # Note: unlike a stale-settings mismatch, `pending_run` is deliberately left
            # untouched here. It is only ever set True immediately before a rerun, right
            # after this exact signature was (re)computed from the freshly stored agent
            # result -- so a mismatch at this point reflects that legitimate change, not
            # a stale request that should cancel the run the user just triggered.
            st.session_state.has_run = False
            st.session_state.result = None
            st.session_state.result_signature = signature
            result = None
            if not using_inferred_tool:
                target_tool = None
                st.session_state.result_signature = (
                    trace_name,
                    inference_backend,
                    inference_model_name,
                    selected_hf_config.cache_key(scorer_mode=str(scorer_backend))
                    if selected_hf_config is not None
                    else None,
                    user_request,
                    "__pending_target__",
                    scorer_backend,
                    logprob_model_id,
                    int(max_pairs_per_batch),
                    router_model_id,
                    soft_vote_model_id,
                    int(soft_vote_n_samples),
                    float(soft_vote_temperature),
                    int(soft_vote_max_retries),
                    soft_vote_seed,
                    trajectory_reference_signature,
                    final_answer_embedding_model_id,
                    bool(enable_fallback_target_selection),
                    show_lexical_comparison,
                    segmenter_choice,
                    segment_threshold,
                    segment_window,
                    min_segment_words,
                    tuple(semantic_user_texts),
                )

        # ---- Run-gating + compute happens here, ahead of every visual section below
        # (including the executive summary), so the whole layout has final results
        # available on first paint instead of computing mid-scroll. ----

        run = st.session_state.pending_run
        st.session_state.pending_run = False

        if run:
            try:
                from tool_game import (
                    ToolUseGame,
                )
            except Exception as error:  # noqa: BLE001
                st.error(
                    "The demo controls are ready, but the full shapiq explanation stack "
                    f"could not be imported in this local environment: {error}"
                )
                return

        if run:
            lexical_scorer = LexicalToolScorer()
            effective_scorer_backend = (
                hf_value_function_label_for_target(target_tool, scorer_backend)
                if inference_backend == "HF local"
                else scorer_backend
            )
            frozen_direct_answer_target = None
            if effective_scorer_backend == "Keyword scorer":
                primary_scorer = lexical_scorer
                primary_label = "Keyword scorer"
                native_hf_consistency = None
            elif effective_scorer_backend == NATIVE_HF_SCORER_LABEL:
                try:
                    selected_native_hf_config = require_selected_hf_config(selected_hf_config)
                    hf_router = load_local_hf_router(
                        selected_native_hf_config.model_id,
                        DEFAULT_NATIVE_HF_MAX_NEW_TOKENS,
                        trust_remote_code=False,
                        quantization_mode=selected_native_hf_config.quantization_mode,
                        device=selected_native_hf_config.device,
                        dtype=selected_native_hf_config.dtype,
                    )
                    primary_scorer = build_native_hf_scorer_from_router(
                        hf_router,
                        max_pairs_per_batch=int(max_pairs_per_batch),
                        selected_config=selected_native_hf_config,
                    )
                    native_hf_consistency = validate_hf_inference_xai_consistency(
                        selected_config=selected_native_hf_config,
                        router=hf_router,
                        scorer=primary_scorer,
                    )
                except Exception as error:  # noqa: BLE001
                    st.error(
                        "Could not prepare the native HF tool-call scorer. Install/check "
                        "`transformers` and `torch`, try a smaller causal language model, "
                        f"or check your environment. Details: {error}"
                    )
                    return
                primary_label = NATIVE_HF_SCORER_LABEL
                target_source = "Agent Result"
            elif effective_scorer_backend == NATIVE_DIRECT_ANSWER_SCORER_LABEL:
                try:
                    selected_native_hf_config = require_selected_hf_config(selected_hf_config)
                    direct_answer_target_text = require_trusted_direct_answer_target(
                        st.session_state.get("agentic_inference_result"),
                        inference_backend=inference_backend,
                    )
                    hf_router = load_local_hf_router(
                        selected_native_hf_config.model_id,
                        DEFAULT_NATIVE_HF_MAX_NEW_TOKENS,
                        trust_remote_code=False,
                        quantization_mode=selected_native_hf_config.quantization_mode,
                        device=selected_native_hf_config.device,
                        dtype=selected_native_hf_config.dtype,
                    )
                    # Frozen once, here -- not rebuilt per coalition below.
                    frozen_direct_answer_target = build_canonical_direct_answer_target(
                        direct_answer_target_text,
                        tokenizer=hf_router.tokenizer,
                    )
                    primary_scorer = build_native_direct_answer_scorer_from_router(
                        hf_router,
                        direct_answer_target=frozen_direct_answer_target,
                        max_pairs_per_batch=int(max_pairs_per_batch),
                        selected_config=selected_native_hf_config,
                    )
                    native_hf_consistency = validate_hf_inference_xai_consistency(
                        selected_config=selected_native_hf_config,
                        router=hf_router,
                        scorer=primary_scorer,
                    )
                except RuntimeError as error:
                    st.error(str(error))
                    return
                except Exception as error:  # noqa: BLE001
                    st.error(
                        "Could not prepare the native direct-answer continuation scorer. "
                        "Install/check `transformers` and `torch`, try a smaller causal "
                        f"language model, or check your environment. Details: {error}"
                    )
                    return
                primary_label = NATIVE_DIRECT_ANSWER_SCORER_LABEL
                target_source = "Agent Result"
            elif effective_scorer_backend in {LOGPROB_SCORER_LABEL, NO_TOOL_SURROGATE_SCORER_LABEL}:
                # No explicit st.spinner() wrapper here: load_logprob_scorer's own
                # @st.cache_resource(show_spinner="Preparing local HF scorer...") already
                # shows a friendly progress message on a real (non-cached) load.
                try:
                    primary_scorer = load_logprob_scorer(
                        logprob_model_id,
                        max_pairs_per_batch=int(max_pairs_per_batch),
                        device=selected_hf_config.device if selected_hf_config else None,
                        dtype=selected_hf_config.dtype if selected_hf_config else "auto",
                        quantization_mode=(
                            selected_hf_config.quantization_mode if selected_hf_config else "none"
                        ),
                        tokenizer_id=selected_hf_config.tokenizer_id
                        if selected_hf_config
                        else None,
                    )
                except Exception as error:  # noqa: BLE001
                    st.error(
                        "Could not load the calibrated log-odds scorer. Install/check "
                        "`transformers` and `torch`, try a smaller causal language model, "
                        f"or check your environment. Details: {error}"
                    )
                    return
                primary_label = effective_scorer_backend
                # Developer ablation only: LocalHFClassificationRouter remains available
                # through build_complete_agent_callable(calibrated_hf_mode=True), but this
                # XAI stage must not reroute and overwrite the Agent Result target.
                if using_inferred_tool:
                    target_source = "Agent Result"
                native_hf_consistency = None
            elif effective_scorer_backend == FINAL_ANSWER_SIMILARITY_LABEL:
                # No explicit st.spinner() wrapper here: load_final_answer_embedder's own
                # @st.cache_resource(show_spinner="Loading embedding model...") already
                # shows a friendly progress message on a real (non-cached) load.
                try:
                    embedder = load_final_answer_embedder(final_answer_embedding_model_id)
                except Exception as error:  # noqa: BLE001
                    st.error(
                        "Could not load embedding model "
                        f"{final_answer_embedding_model_id!r}: {error}"
                    )
                    return
                agent_callable = build_complete_agent_callable(
                    inference_backend=inference_backend,
                    inference_model_name=inference_model_name,
                    system_prompt=system_prompt,
                    tool_context=tool_context,
                    hf_model_config=selected_hf_config,
                    hf_max_new_tokens=DEFAULT_NATIVE_HF_MAX_NEW_TOKENS,
                    hf_trust_remote_code=False,
                )
                inference_result_is_current = (
                    st.session_state.get("agentic_inference_signature")
                    == current_inference_signature
                    and st.session_state.get("agentic_inference_backend") == inference_backend
                    and st.session_state.get("agentic_inference_model") == inference_model_name
                )
                with st.spinner("Resolving full-run reference answer..."):
                    reference_answer, reused_existing_inference, reference_error = (
                        resolve_full_run_reference_answer(
                            agent_callable=agent_callable,
                            user_request=user_request,
                            inference_result=inference_result,
                            inference_result_is_current=inference_result_is_current,
                        )
                    )
                if not reference_answer:
                    st.error(
                        "Could not obtain a final answer for the full user request, so the "
                        "final answer similarity scorer cannot run. "
                        f"Reason: {reference_error or 'unknown error'}"
                    )
                    return
                primary_scorer = FinalAnswerSimilarityScorer(
                    agent_callable=agent_callable,
                    embedder=embedder,
                    reference_answer=reference_answer,
                    empty_prompt=empty_prompt,
                )
                primary_label = FINAL_ANSWER_SIMILARITY_LABEL
                native_hf_consistency = None
            elif effective_scorer_backend == "Groq deterministic router":
                if not os.getenv("GROQ_API_KEY"):
                    st.error(
                        "GROQ_API_KEY is not set. Add it to the environment to use the Groq "
                        "deterministic router scorer."
                    )
                    return
                primary_scorer = GroqDeterministicRouterScorer(model_name=router_model_id)
                primary_label = "Groq deterministic router"
                native_hf_consistency = None
            elif effective_scorer_backend == "Groq soft-vote scorer":
                if not os.getenv("GROQ_API_KEY"):
                    st.error(
                        "GROQ_API_KEY is not set. Add it to the environment to use the Groq "
                        "soft-vote scorer."
                    )
                    return
                primary_scorer = GroqSoftVoteToolScorer(
                    model_name=soft_vote_model_id,
                    n_samples=int(soft_vote_n_samples),
                    temperature=float(soft_vote_temperature),
                    max_retries=int(soft_vote_max_retries),
                    seed=soft_vote_seed,
                )
                primary_label = "Groq soft-vote scorer"
                native_hf_consistency = None
            elif effective_scorer_backend == "Trajectory match: tool + normalized args":
                if not os.getenv("GROQ_API_KEY"):
                    st.error(
                        "GROQ_API_KEY is not set. Add it to the environment to use the "
                        "trajectory match scorer."
                    )
                    return
                if groq_reference_result is None or not groq_reference_result.tool_arguments:
                    st.error(
                        "No real Groq inference result with tool arguments is available. "
                        "Run Groq inference first."
                    )
                    return
                reference_trajectory = ToolTrajectory(
                    selected_tool=groq_reference_result.selected_tool,
                    tool_arguments=dict(groq_reference_result.tool_arguments),
                )
                trajectory_provider = build_groq_inference_trajectory_provider(
                    getattr(groq_reference_result, "model", DEFAULT_GROQ_ROUTER_MODEL_ID),
                    get_executable_tool_schemas(),
                    tool_context=tool_context,
                )
                primary_scorer = TrajectoryArgumentMatchScorer(
                    reference_trajectory=reference_trajectory,
                    trajectory_provider=trajectory_provider,
                )
                primary_label = "Trajectory match: tool + normalized args"
                native_hf_consistency = None
            else:
                primary_scorer = LLMToolScorer(llm=MockLLM())
                primary_label = "Mock model scorer"
                native_hf_consistency = None

            fallback_choice = None
            if target_tool is None:
                if not enable_fallback_target_selection:
                    st.error(
                        "No agent result is available. Run the full pipeline first, or enable "
                        "fallback target selection in developer settings."
                    )
                    return
                with st.spinner("Selecting fallback target tool with the explanation scorer..."):
                    fallback_choice = choose_tool_with_scorer(
                        primary_scorer,
                        full_prompt,
                        tool_descriptions=TOOLS,
                    )
                target_tool = fallback_choice.tool
                target_source = "fallback explanation scorer"

            full_score = primary_scorer.score_batch(
                [full_prompt],
                target_tool=target_tool,
                tool_descriptions=TOOLS,
            )[0]
            logprob_full_diagnostics = (
                dict(primary_scorer.last_debug_outputs[0])
                if primary_label in {LOGPROB_SCORER_LABEL, NO_TOOL_SURROGATE_SCORER_LABEL}
                and primary_scorer.last_debug_outputs
                else None
            )
            empty_score = primary_scorer.score_batch(
                [empty_prompt],
                target_tool=target_tool,
                tool_descriptions=TOOLS,
            )[0]

            with st.spinner("Computing SV and k-SII..."):
                game = ToolUseGame(
                    target_tool=target_tool,
                    user_segments=user_segments,
                    system_prompt=system_prompt,
                    tool_context=tool_context,
                    scorer=primary_scorer,
                    tool_descriptions=TOOLS,
                    defer_empty_coalition_evaluation=using_exact_computation,
                )
                try:
                    sv_explanation, sv_algorithm_label, ksii_explanation, ksii_algorithm_label = (
                        compute_dual_index_explanations(game=game, budget=budget)
                    )
                except (ExactComputationLimitError, UnsupportedExactIndexError) as error:
                    st.error(f"Could not compute the {SV_INDEX}/{KSII_INDEX} explanations: {error}")
                    return
                except CoalitionEvaluationIncompleteError as error:
                    st.error(str(error))
                    metrics = error.metrics
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {"metric": "coalition_total", "value": metrics.coalition_total},
                                {"metric": "real_count", "value": metrics.real_count},
                                {"metric": "fallback_count", "value": metrics.fallback_count},
                                {
                                    "metric": "retry_triggered_count",
                                    "value": metrics.retry_triggered_count,
                                },
                                {
                                    "metric": "retry_success_count",
                                    "value": metrics.retry_success_count,
                                },
                                {
                                    "metric": "retry_exhausted_count",
                                    "value": metrics.retry_exhausted_count,
                                },
                                {
                                    "metric": "semantic_failure_count",
                                    "value": metrics.semantic_failure_count,
                                },
                                {
                                    "metric": "remote_request_count",
                                    "value": metrics.remote_request_count,
                                },
                                {
                                    "metric": "embedding_call_count",
                                    "value": metrics.embedding_call_count,
                                },
                            ]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                    return
                first_order_sv = sv_explanation.get_n_order(order=1)
                attribution_frame = values_to_frame(first_order_sv, user_segments)
                pairwise_ksii = ksii_explanation.get_n_order(order=2)
                # Uses the full (unfiltered) ksii_explanation, not pairwise_ksii above --
                # the combined matrix needs its order-1 singleton entries for the
                # diagonal, which get_n_order(order=2) strips out.
                pairwise_matrix = pairwise_matrix_from_explanation(ksii_explanation, game.n_players)
                pair_label, pair_value = strongest_pair(pairwise_matrix, labels)
                notes = build_interpretation_notes(
                    attribution_frame,
                    pair_label,
                    pair_value,
                    full_score,
                    empty_score,
                )

            top = attribution_frame.iloc[0] if not attribution_frame.empty else None
            top_label = "No segment" if top is None else f"{top['segment']} ({top['source']})"
            top_score = 0.0 if top is None else float(top["attribution"])
            interpretation_sentence = (
                notes[0] if notes else "No interpretation is available for this run."
            )

            compare_with_lexical = show_lexical_comparison and primary_label != "Keyword scorer"
            llm_debug_outputs = getattr(primary_scorer, "last_debug_outputs", [])
            lexical_result = None
            if compare_with_lexical:
                with st.spinner("Computing lexical baseline comparison..."):
                    lexical_game = ToolUseGame(
                        target_tool=target_tool,
                        user_segments=user_segments,
                        system_prompt=system_prompt,
                        tool_context=tool_context,
                        scorer=lexical_scorer,
                        tool_descriptions=TOOLS,
                        defer_empty_coalition_evaluation=using_exact_computation,
                    )
                    try:
                        (
                            lexical_sv_explanation,
                            _lexical_sv_algorithm_label,
                            lexical_ksii_explanation,
                            _lexical_ksii_algorithm_label,
                        ) = compute_dual_index_explanations(
                            game=lexical_game,
                            budget=budget,
                        )
                    except (
                        ExactComputationLimitError,
                        UnsupportedExactIndexError,
                        CoalitionEvaluationIncompleteError,
                    ) as error:
                        st.warning(f"Could not compute the keyword-scorer comparison: {error}")
                        compare_with_lexical = False
                        lexical_result = None
                    else:
                        lexical_first_order_sv = lexical_sv_explanation.get_n_order(order=1)
                        lexical_frame = values_to_frame(lexical_first_order_sv, user_segments)
                        lexical_pairwise_ksii = lexical_ksii_explanation.get_n_order(order=2)
                        # Only used for strongest_pair's off-diagonal ranking below, never
                        # displayed -- the diagonal-free variant is enough here.
                        lexical_matrix = pairwise_only_matrix_from_explanation(
                            lexical_pairwise_ksii,
                            lexical_game.n_players,
                        )
                        lexical_pair_label, lexical_pair_value = strongest_pair(
                            lexical_matrix,
                            labels,
                        )
                        lexical_full_score = lexical_scorer.score_batch(
                            [full_prompt],
                            target_tool=target_tool,
                            tool_descriptions=TOOLS,
                        )[0]
                        lexical_empty_score = lexical_scorer.score_batch(
                            [empty_prompt],
                            target_tool=target_tool,
                            tool_descriptions=TOOLS,
                        )[0]
                        lexical_top = lexical_frame.iloc[0] if not lexical_frame.empty else None
                        lexical_result = {
                            "label": "Keyword scorer",
                            "full_score": lexical_full_score,
                            "empty_score": lexical_empty_score,
                            "top": "No segment"
                            if lexical_top is None
                            else f"{lexical_top['segment']} ({lexical_top['source']})",
                            "top_value": 0.0
                            if lexical_top is None
                            else float(lexical_top["attribution"]),
                            "pair": lexical_pair_label,
                            "pair_value": lexical_pair_value,
                        }

            scoring_prompt_preview = build_scoring_prompt_preview(
                primary_scorer,
                full_prompt,
                target_tool=target_tool,
                tool_descriptions=TOOLS,
            )
            value_function_type, value_function_fidelity = value_function_metadata(primary_label)
            st.session_state.has_run = True
            result_signature = (
                trace_name,
                inference_backend,
                inference_model_name,
                selected_hf_config.cache_key(scorer_mode=str(primary_label))
                if selected_hf_config is not None
                else None,
                user_request,
                target_tool,
                scorer_backend,
                logprob_model_id,
                int(max_pairs_per_batch),
                router_model_id,
                soft_vote_model_id,
                int(soft_vote_n_samples),
                float(soft_vote_temperature),
                int(soft_vote_max_retries),
                soft_vote_seed,
                trajectory_reference_signature,
                final_answer_embedding_model_id,
                bool(enable_fallback_target_selection),
                show_lexical_comparison,
                segmenter_choice,
                segment_threshold,
                segment_window,
                min_segment_words,
                tuple(semantic_user_texts),
            )
            st.session_state.result_signature = result_signature
            st.session_state.result = {
                "target_tool": target_tool,
                "target_source": target_source,
                "fallback_choice_scores": None
                if fallback_choice is None
                else fallback_choice.scores,
                "primary_label": primary_label,
                "scorer_label": primary_label,
                "value_function_type": value_function_type,
                "value_function_fidelity": value_function_fidelity,
                "sv_algorithm_label": sv_algorithm_label,
                "ksii_algorithm_label": ksii_algorithm_label,
                "full_score": full_score,
                "empty_score": empty_score,
                "first_order_sv": first_order_sv,
                "attribution_frame": attribution_frame,
                "sv_explanation": sv_explanation,
                "ksii_explanation": ksii_explanation,
                "pairwise_ksii": pairwise_ksii,
                "pairwise_matrix": pairwise_matrix,
                "pair_label": pair_label,
                "pair_value": pair_value,
                "top_label": top_label,
                "top_score": top_score,
                "interpretation_sentence": interpretation_sentence,
                "compare_with_lexical": compare_with_lexical,
                "llm_debug_outputs": llm_debug_outputs,
                "lexical_result": lexical_result,
                "scoring_prompt": scoring_prompt_preview,
                "logprob_full_diagnostics": logprob_full_diagnostics,
                "native_hf_consistency": native_hf_consistency,
                "hf_selected_model_config": selected_hf_config,
                "direct_answer_target": frozen_direct_answer_target,
                "final_answer_scorer_meta": (
                    {
                        "embedding_model_id": final_answer_embedding_model_id,
                        "reference_answer": primary_scorer.reference_answer,
                        "reused_existing_inference": reused_existing_inference,
                        "empty_raw_similarity": primary_scorer.last_empty_raw_similarity,
                    }
                    if primary_label == FINAL_ANSWER_SIMILARITY_LABEL
                    else None
                ),
            }
            inference_result = st.session_state.get("agentic_inference_result")
            routing_diagnostics = logprob_full_diagnostics or {}
            raw_scores = routing_diagnostics.get("raw_log_scores") or getattr(
                inference_result,
                "raw_scores",
                {},
            )
            calibrated_scores = routing_diagnostics.get("calibrated_scores") or getattr(
                inference_result,
                "tool_scores",
                {},
            )
            write_result_export_safely(
                warning_callback=st.warning,
                session_id=st.session_state["agentic_export_session_id"],
                session_started_at=st.session_state["agentic_export_session_started_at"],
                hf_model_id=inference_model_name,
                user_request=user_request,
                system_prompt=system_prompt,
                player_segments=user_segments,
                raw_scores=raw_scores,
                calibrated_scores=calibrated_scores,
                selected_tool=getattr(inference_result, "selected_tool", target_tool),
                raw_argmax=routing_diagnostics.get("argmax_tool")
                or (max(raw_scores, key=raw_scores.get) if raw_scores else None),
                calibrated_argmax=(
                    max(calibrated_scores, key=calibrated_scores.get) if calibrated_scores else None
                ),
                target_tool=target_tool,
                baseline_h_empty=empty_score,
                full_h_n=full_score,
                pairwise_interactions=build_pairwise_interactions(
                    player_segments=user_segments,
                    pairwise_matrix=pairwise_matrix,
                ),
                value_function_type=value_function_type,
                value_function_fidelity=value_function_fidelity,
                primary_label=primary_label,
            )

        result = st.session_state.result
        if result is None:
            st.error("No explanation result is available. Click Run full pipeline to compute one.")
            return
        primary_label = result["primary_label"]
        scorer_label = result.get("scorer_label", primary_label)
        value_function_type = result.get("value_function_type", "")
        value_function_fidelity = result.get("value_function_fidelity", "")
        sv_algorithm_label = result["sv_algorithm_label"]
        ksii_algorithm_label = result["ksii_algorithm_label"]
        full_score = result["full_score"]
        empty_score = result["empty_score"]
        first_order_sv = result["first_order_sv"]
        attribution_frame = result["attribution_frame"]
        sv_explanation = result["sv_explanation"]
        ksii_explanation = result["ksii_explanation"]
        pairwise_ksii = result.get("pairwise_ksii")
        if pairwise_ksii is None:
            pairwise_ksii = ksii_explanation.get_n_order(order=2)
        pairwise_matrix = result.get("pairwise_matrix")
        if pairwise_matrix is None:
            # Full (unfiltered) ksii_explanation, not pairwise_ksii -- see the
            # matching comment at the primary computation call site above.
            pairwise_matrix = pairwise_matrix_from_explanation(ksii_explanation, len(user_segments))
        pair_label = result["pair_label"]
        pair_value = result["pair_value"]
        compare_with_lexical = result["compare_with_lexical"]
        llm_debug_outputs = result["llm_debug_outputs"]
        lexical_result = result["lexical_result"]
        final_answer_scorer_meta = result.get("final_answer_scorer_meta")
        logprob_full_diagnostics = result.get("logprob_full_diagnostics")
        native_hf_consistency = result.get("native_hf_consistency")
        hf_selected_model_config = result.get("hf_selected_model_config")
        is_final_answer_result = primary_label == FINAL_ANSWER_SIMILARITY_LABEL
        target_tool = result["target_tool"]
        target_source = result["target_source"]

        # ---- Defensive consistency check ----
        # The cached `result` dict must never explain a different tool than the current
        # Agent Result (e.g. a stale result computed for the previous HF model before a
        # 1.5B -> 3B switch). This should never trigger given the signature invalidation
        # above; it exists as a last-resort guard so a mismatch is caught and surfaced
        # instead of silently rendering the wrong explanation. Only checked when there is
        # a genuine current agent-selected tool to compare against -- the fallback
        # explanation scorer path legitimately has no `agentic_inferred_tool` to match.
        current_agentic_inferred_tool = st.session_state.get("agentic_inferred_tool")
        if current_agentic_inferred_tool in TOOLS and target_tool != current_agentic_inferred_tool:
            st.session_state.has_run = False
            st.session_state.result = None
            st.session_state.result_signature = None
            st.warning(
                "The previous explanation no longer matches the current Agent Result "
                f"(`{current_agentic_inferred_tool}`). Click **Run full pipeline** above "
                "to recompute the explanation."
            )
            return

        # ================================================================
        # Visual layout: a readable explanation first (0), then progressively
        # more detail (1-4). Every value below is read from the `result` dict
        # already unpacked above -- no new computation is introduced here.
        # ================================================================

        # The calibrated log-odds scorer normalizes its game against the empty coalition.
        # Its debug record preserves the corresponding raw h(empty)/h(full) values, so use
        # their already-computed difference for the summary instead of displaying the
        # normalized zero baseline as though it were the underlying continuation score.
        raw_log_odds_summary = (
            None
            if is_final_answer_result
            else _extract_raw_log_odds_summary(logprob_full_diagnostics)
        )
        if raw_log_odds_summary is not None:
            raw_full_score, raw_empty_score = raw_log_odds_summary
            explained_increase = raw_full_score - raw_empty_score
        else:
            explained_increase = float(full_score) - float(empty_score)

        # Reused below by the k-SII mini-table and interpretation card.
        top_pairs = (
            top_pairwise_interactions(pairwise_matrix, user_segments)
            if pairwise_matrix.shape[0] >= 2
            else []
        )
        short_labels = [short_player_label(segment) for segment in user_segments]

        # ---- 0. Technically scoped explanation summary ----
        st.markdown(
            render_xai_summary_section(
                selected_tool=target_tool,
                model_name=inference_model_name,
                backend_name=inference_backend,
                score_change=explained_increase,
                attribution_frame=attribution_frame,
                pairwise_matrix=pairwise_matrix,
                user_segments=user_segments,
            ),
            unsafe_allow_html=True,
        )

        # ---- Setup chips: compact, user-facing run metadata only ----
        if using_exact_computation:
            coalition_chip_label = "Coalitions"
            coalition_chip_value = f"{2 ** len(user_segments)} exact"
        else:
            coalition_chip_label = "Budget"
            coalition_chip_value = f"{budget} (approx.)"
        st.markdown(
            f"""
            <div class="setup-chip-row">
                <span class="setup-chip">
                    <strong>Players:</strong> {len(user_segments)} user-request segments
                </span>
                <span class="setup-chip"><strong>Indices:</strong> SV + k-SII</span>
                <span class="setup-chip">
                    <strong>{coalition_chip_label}:</strong> {coalition_chip_value}
                </span>
                <span class="setup-chip">
                    <strong>VF:</strong>
                    {escape(scorer_short_label(scorer_label))}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if (
            developer_mode
            and target_source == "fallback explanation scorer"
            and isinstance(result, dict)
            and result.get("fallback_choice_scores")
        ):
            with st.expander("Fallback scorer diagnostic", expanded=False):
                st.caption(
                    "These scores come from the selected explanation scorer and were used to "
                    "choose the fallback target tool (no agent result was available)."
                )
                score_frame = pd.DataFrame(
                    [
                        {"tool": tool, "score": score}
                        for tool, score in sorted(
                            result["fallback_choice_scores"].items(),
                            key=lambda item: item[1],
                            reverse=True,
                        )
                    ]
                )
                st.dataframe(score_frame, use_container_width=True, hide_index=True, height=178)

        # ---- 1. Player segmentation card, with fixed context nested inside it ----
        with st.container(key="agentic_player_card"):
            player_chip_html = "".join(build_player_chip_html(segment) for segment in user_segments)
            st.markdown(
                f"""
                <div class="player-card-header">Player segmentation</div>
                <div class="player-chip-grid">{player_chip_html}</div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("Fixed context — system prompt + tool definitions", expanded=False):
                st.caption("System prompt")
                for segment in system_segments:
                    st.markdown(
                        (
                            "<div class='segment-box'>"
                            f"<h4>{segment.label}</h4><p>{escape(segment.text)}</p></div>"
                        ),
                        unsafe_allow_html=True,
                    )
                st.caption("Tool definitions")
                for tool_name, description in TOOLS.items():
                    st.markdown(
                        (
                            "<div class='segment-box'>"
                            f"<h4>{escape(tool_name)}</h4><p>{escape(description)}</p></div>"
                        ),
                        unsafe_allow_html=True,
                    )
                if developer_mode and segment_debug_rows:
                    # Duck-typed rather than isinstance(...): Streamlit's local file watcher
                    # can hot-reload a same-directory module while an older cached segmenter
                    # instance from st.cache_resource survives the rerun, which breaks
                    # isinstance checks against the freshly reloaded class.
                    is_linguistic_segmenter = hasattr(segmenter, "stray_merge")
                    diagnostic_label = (
                        "Linguistic segment diagnostics"
                        if is_linguistic_segmenter
                        else "Semantic boundary diagnostics"
                    )
                    st.markdown(f"**{diagnostic_label}**")
                    st.dataframe(
                        pd.DataFrame(segment_debug_rows),
                        use_container_width=True,
                        hide_index=True,
                    )

        # ---- 2 & 3. SV | k-SII evidence, two balanced cards side by side ----
        token_attribution_bar_plot, sentence_interaction_heatmap, plot_import_error = (
            load_text_plotters()
        )
        bar_xlabel = (
            "Final-answer semantic-fidelity attribution"
            if is_final_answer_result
            else "Target-tool attribution"
        )
        sv_col, ksii_col = st.columns(2, gap="medium")
        with sv_col, st.container(key="agentic_sv_card"):
            st.markdown(
                """
                <div class="evidence-card-header">
                    <div class="evidence-card-title">Segment contributions — Shapley values</div>
                    <div class="evidence-card-caption">
                        Positive supports the target tool; negative opposes it.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if token_attribution_bar_plot is None:
                st.warning(
                    "The shapiq text attribution plot is unavailable in this environment. "
                    f"Showing a simple fallback chart instead. Details: {plot_import_error}"
                )
                show_fallback_attribution_chart(attribution_frame)
            else:
                try:
                    fig_ax = token_attribution_bar_plot(first_order_sv, short_labels, show=False)
                except Exception as error:  # noqa: BLE001
                    st.warning(
                        "The shapiq text attribution plot failed. "
                        f"Showing a simple fallback chart instead. Details: {error}"
                    )
                    show_fallback_attribution_chart(attribution_frame)
                else:
                    if fig_ax is not None:
                        fig, ax = fig_ax
                        st.pyplot(polish_bar(fig, ax, xlabel=bar_xlabel), clear_figure=True)
            # Compact by default; for larger player sets, show the top rows inline and
            # keep the rest in an expander instead of growing the card unboundedly.
            # `attribution_frame` is already ranked by |attribution| descending.
            show_all_sv_inline = len(user_segments) <= 6
            sv_rows_to_show = attribution_frame if show_all_sv_inline else attribution_frame.head(5)
            st.markdown(build_sv_mini_table_html(sv_rows_to_show), unsafe_allow_html=True)
            if not show_all_sv_inline:
                with st.expander("Show full SV attribution table", expanded=False):
                    st.dataframe(attribution_frame, use_container_width=True, hide_index=True)

        with ksii_col, st.container(key="agentic_ksii_card"):
            st.markdown(
                """
                <div class="evidence-card-header">
                    <div class="evidence-card-title">Main and pairwise effects — k-SII</div>
                    <div class="evidence-card-caption">
                        Diagonal: main effects &middot; Off-diagonal: pairwise interactions<br>
                        Blue: negative &middot; Red: positive
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if sentence_interaction_heatmap is None:
                st.warning(
                    "The shapiq text interaction heatmap is unavailable in this environment. "
                    f"Showing a fallback interaction table instead. Details: {plot_import_error}"
                )
                show_fallback_interaction_table(pairwise_matrix, labels)
            else:
                try:
                    fig_ax = sentence_interaction_heatmap(ksii_explanation, labels, show=False)
                except Exception as error:  # noqa: BLE001
                    st.warning(
                        "The shapiq text interaction heatmap failed. "
                        f"Showing a fallback interaction table instead. Details: {error}"
                    )
                    show_fallback_interaction_table(pairwise_matrix, labels)
                else:
                    if fig_ax is not None:
                        fig, ax = fig_ax
                        st.pyplot(polish_heatmap(fig, ax, user_segments), clear_figure=True)
            st.markdown(build_player_legend_html(user_segments), unsafe_allow_html=True)

            # `top_pairwise_interactions` already ranks by |k-SII| descending; requesting
            # every pair (instead of the default top-5) here is the same ranking, just
            # unsliced, so the compact table below can show them all when there are few
            # enough players.
            all_pairs = (
                top_pairwise_interactions(
                    pairwise_matrix,
                    user_segments,
                    n=max(len(user_segments) * (len(user_segments) - 1) // 2, 1),
                )
                if pairwise_matrix.shape[0] >= 2
                else []
            )
            show_all_ksii_inline = len(user_segments) <= 6
            ksii_rows_to_show = all_pairs if show_all_ksii_inline else all_pairs[:5]

            if ksii_rows_to_show:
                st.markdown(build_ksii_mini_table_html(ksii_rows_to_show), unsafe_allow_html=True)
            else:
                st.caption("Only one player -- no pairwise interactions to show.")

            if not show_all_ksii_inline and all_pairs:
                with st.expander("Show all pairwise interactions", expanded=False):
                    st.markdown(build_ksii_mini_table_html(all_pairs), unsafe_allow_html=True)

            with st.expander("Show full interaction matrix", expanded=False):
                show_fallback_interaction_table(pairwise_matrix, labels)

        # ---- Detailed interpretation (collapsed: already summarized in the top card) ----
        with st.expander("Detailed interpretation", expanded=False):
            interpretation_html = build_interaction_interpretation(
                pair_label=pair_label,
                pair_value=pair_value,
                top_pairs=top_pairs,
            )
            st.markdown(
                f'<div class="interpretation-card">{interpretation_html}</div>',
                unsafe_allow_html=True,
            )

        # ---- Developer mode: computation diagnostics + raw backend diagnostics ----
        if developer_mode:
            with st.expander("Show computation diagnostics", expanded=False):
                if using_exact_computation:
                    coalition_count = 2 ** len(user_segments)
                    st.caption(
                        "Algorithm: `shapiq ExactComputer` "
                        f"(exact evaluation: `{coalition_count}` / `{coalition_count}` coalitions)"
                    )
                else:
                    st.caption(
                        f"`{len(user_segments)}` players exceeds the exact limit of "
                        f"`{MAX_EXACT_DEMO_PLAYERS}`. Algorithm: official shapiq approximation, "
                        f"budget: `{budget}` auto."
                    )
                sv_sum = float(sum(getattr(first_order_sv, "dict_values", {}).values()))
                sv_residual = abs(sv_sum - explained_increase)
                sv_efficiency_status = (
                    "passes" if sv_residual <= SV_EFFICIENCY_TOLERANCE else "exceeds tolerance"
                )
                st.caption(
                    f"SV efficiency check: sum(SV)={sv_sum:.3f}, "
                    f"support change={explained_increase:.3f}, "
                    f"residual={sv_residual:.6g} ({sv_efficiency_status})"
                )
                diag = interaction_order_diagnostics(
                    ksii_explanation,
                    full_value=full_score,
                    empty_value=empty_score,
                )
                diag_frame = pd.DataFrame(
                    [
                        {
                            "quantity": "sum of order-1 values",
                            "value": f"{diag['order_1_sum']:.6f}",
                        },
                        {
                            "quantity": "sum of order-2 pairwise values",
                            "value": f"{diag['order_2_sum']:.6f}",
                        },
                        {
                            "quantity": "total game value v(N) - v(empty)",
                            "value": f"{diag['total_game_value']:.6f}",
                        },
                        {
                            "quantity": "efficiency residual",
                            "value": f"{diag['residual']:.6g}",
                        },
                    ]
                )
                st.dataframe(diag_frame, use_container_width=True, hide_index=True)
                if abs(diag["residual"]) <= EFFICIENCY_RESIDUAL_TOLERANCE:
                    st.caption(
                        "k-SII order-2 efficiency holds: first- and second-order values "
                        "sum to the total game value."
                    )
                else:
                    st.warning(
                        f"Efficiency residual {diag['residual']:.6g} exceeds tolerance "
                        f"{EFFICIENCY_RESIDUAL_TOLERANCE:g}. "
                        "This may indicate approximation error above the exact-computation limit."
                    )

                if show_value_function_details:
                    st.markdown("**Value function details**")
                    st.write(
                        f"Individual effects: `{SV_INDEX}`, max_order=`{SV_MAX_ORDER}`, "
                        f"algorithm: `{sv_algorithm_label}`"
                    )
                    st.write(
                        f"Pairwise interactions: `{KSII_INDEX}`, max_order=`{KSII_MAX_ORDER}`, "
                        f"algorithm: `{ksii_algorithm_label}`"
                    )
                    st.write("Players: user-request segments")
                    st.write("Fixed context: system prompt + tool definitions")
                    st.write("Value function: selected-tool support under the chosen scorer")
                    st.write(f"Value function type: `{value_function_type}`")
                    st.write(f"Value function fidelity: `{value_function_fidelity}`")
                    if not using_exact_computation:
                        st.write(f"Budget: `{budget}`")
                    st.write("Full coalition prompt:")
                    st.code(full_prompt, language="text")
                    st.write("Empty coalition prompt:")
                    st.code(empty_prompt, language="text")

            with st.expander("Raw backend diagnostics", expanded=False):
                if native_hf_consistency is not None:
                    st.markdown("**HF native inference/XAI consistency**")
                    inference_identity = native_hf_consistency["inference"]
                    scorer_identity = native_hf_consistency["scorer"]
                    st.write(
                        "Inference/XAI model match: "
                        f"`{'yes' if native_hf_consistency['match'] else 'no'}`"
                    )
                    st.write(f"Inference model ID: `{inference_identity['model_id']}`")
                    st.write(f"Scorer model ID: `{scorer_identity['model_id']}`")
                    st.write(f"Inference tokenizer ID: `{inference_identity['tokenizer_id']}`")
                    st.write(f"Scorer tokenizer ID: `{scorer_identity['tokenizer_id']}`")
                    st.write(f"Model family: `{native_hf_consistency['model_family']}`")
                    st.write(
                        f"Requested quantization: `{inference_identity['requested_quantization']}`"
                    )
                    st.write(f"Actual quantization: `{inference_identity['actual_quantization']}`")
                    st.write(f"Device: `{inference_identity['device']}`")
                    st.write(f"Dtype / compute dtype: `{inference_identity['dtype']}`")
                    st.write(
                        "Native tool template source: `selected tokenizer.apply_chat_template`"
                    )
                    continuation_diagnostics = (
                        NATIVE_DIRECT_ANSWER_CONTINUATION_DIAGNOSTICS
                        if primary_label == NATIVE_DIRECT_ANSWER_SCORER_LABEL
                        else NATIVE_HF_CONTINUATION_DIAGNOSTICS
                    )
                    for (
                        diagnostic_name,
                        diagnostic_value,
                    ) in continuation_diagnostics.items():
                        st.write(f"{diagnostic_name}: `{diagnostic_value}`")
                    st.write(f"Scorer mode: `{primary_label}`")
                    if isinstance(hf_selected_model_config, SelectedHFModelConfig):
                        st.write(
                            "Cache key: "
                            f"`{hf_selected_model_config.cache_key(scorer_mode=str(primary_label))}`"
                        )
                    st.write(f"Target tool: `{target_tool}`")
                    if llm_debug_outputs:
                        token_count = llm_debug_outputs[0].get("continuation_token_count")
                        st.write(f"Continuation token count: `{token_count}`")

                if llm_debug_outputs:
                    st.markdown("**Model output diagnostics**")
                    displayed_debug_outputs = llm_debug_outputs[:10]
                    debug_frame = pd.DataFrame(displayed_debug_outputs)
                    debug_columns = [
                        "score_kind",
                        "score_description",
                        "selected_tools",
                        "target_matches",
                        "n_samples",
                        "temperature",
                        "raw_output",
                        "raw_outputs",
                        "parsed_score",
                        "used_fallback",
                        "fallback_score",
                        "target_tool",
                        "target_source",
                        "target_label",
                        "continuation_type",
                        "continuation_scope",
                        "arguments_included",
                        "candidate_tools",
                        "candidate_labels",
                        "candidate_continuations",
                        "raw_log_scores",
                        "calibration_log_scores",
                        "calibrated_scores",
                        "target_vs_all_log_odds",
                        "empty_coalition_log_odds",
                        "calibrated_probabilities",
                        "argmax_tool",
                        "final_score",
                        "prompt_preview",
                        "masked_user_request",
                        "raw_similarity",
                        "normalized_score",
                        "execution_status",
                        "execution_error",
                        "model_id",
                        "tokenizer_id",
                        "requested_quantization",
                        "actual_quantization",
                        "device",
                        "dtype",
                        "continuation_token_count",
                    ]
                    st.dataframe(
                        debug_frame[[column for column in debug_columns if column in debug_frame]],
                        use_container_width=True,
                        hide_index=True,
                    )

                if (
                    primary_label in {LOGPROB_SCORER_LABEL, NO_TOOL_SURROGATE_SCORER_LABEL}
                    and logprob_full_diagnostics is not None
                ):
                    st.markdown("**Legacy A/B/C/D probe diagnostics (full user request)**")
                    argmax_tool = logprob_full_diagnostics.get("argmax_tool")
                    is_router_argmax = argmax_tool == target_tool
                    diagnostic_softmax = logprob_full_diagnostics["calibrated_probabilities"].get(
                        target_tool
                    )
                    st.caption(
                        NO_TOOL_SURROGATE_HELP
                        if primary_label == NO_TOOL_SURROGATE_SCORER_LABEL
                        else LOGPROB_SCORER_HELP
                    )
                    logprob_metric_left, logprob_metric_mid, logprob_metric_right = st.columns(3)
                    logprob_metric_left.metric(
                        "Diagnostic softmax score",
                        f"{diagnostic_softmax:.3f}" if diagnostic_softmax is not None else "n/a",
                    )
                    logprob_metric_mid.metric(
                        "Target-vs-all log-odds",
                        f"{logprob_full_diagnostics['target_vs_all_log_odds']:.3f}",
                    )
                    logprob_metric_right.metric(
                        "Local-router argmax match",
                        "Yes" if is_router_argmax else "No",
                        help=f"Local-router argmax candidate: `{argmax_tool}`.",
                    )
                    st.caption(
                        f"{SAME_HF_MODEL_EXPLANATION} Loaded model: `{logprob_model_id}`. "
                        "No second model is loaded for target-tool selection. Argument "
                        "extraction was not executed; this run explains routing only."
                    )

                if final_answer_scorer_meta is not None:
                    st.markdown("**Final answer similarity details**")
                    diagnostics = interaction_order_diagnostics(
                        ksii_explanation,
                        full_value=full_score,
                        empty_value=empty_score,
                    )
                    st.write(f"Embedding model: `{final_answer_scorer_meta['embedding_model_id']}`")
                    st.write("Normalized by empty-coalition raw similarity: `True`")
                    empty_raw_similarity = final_answer_scorer_meta["empty_raw_similarity"]
                    st.write(
                        f"Raw empty-coalition similarity: `{empty_raw_similarity:.3f}`"
                        if empty_raw_similarity is not None
                        else "Raw empty-coalition similarity: not yet computed"
                    )
                    st.write(
                        "Reused the existing agent result as the reference answer: "
                        f"`{final_answer_scorer_meta['reused_existing_inference']}`"
                    )
                    raw_full_similarity = (
                        float(diagnostics["full_value"]) + float(empty_raw_similarity)
                        if empty_raw_similarity is not None
                        else None
                    )
                    diagnostic_frame = pd.DataFrame(
                        [
                            {
                                "quantity": "raw empty-coalition cosine similarity",
                                "value": empty_raw_similarity,
                            },
                            {
                                "quantity": "raw full-coalition cosine similarity",
                                "value": raw_full_similarity,
                            },
                            {
                                "quantity": "normalized full-coalition game value",
                                "value": diagnostics["full_value"],
                            },
                            {
                                "quantity": "sum of order-1 interaction values",
                                "value": diagnostics["order_1_sum"],
                            },
                            {
                                "quantity": "sum of unique order-2 pairwise interaction values",
                                "value": diagnostics["order_2_sum"],
                            },
                            {
                                "quantity": "k-SII efficiency residual",
                                "value": diagnostics["residual"],
                            },
                        ]
                    )
                    diagnostic_frame["value"] = diagnostic_frame["value"].map(
                        lambda value: "n/a" if value is None else f"{float(value):.6f}"
                    )
                    st.dataframe(
                        diagnostic_frame,
                        use_container_width=True,
                        hide_index=True,
                    )
                    if abs(diagnostics["residual"]) > EFFICIENCY_RESIDUAL_TOLERANCE:
                        st.warning(
                            "k-SII efficiency residual exceeds tolerance "
                            f"{EFFICIENCY_RESIDUAL_TOLERANCE:g}: "
                            f"{diagnostics['residual']:.6g}"
                        )
                    failed_coalition_rows = [
                        row
                        for row in llm_debug_outputs
                        if row.get("execution_status") not in (None, "ok")
                    ]
                    if failed_coalition_rows:
                        st.warning(
                            f"{len(failed_coalition_rows)} of {len(llm_debug_outputs)} "
                            "sampled coalitions did not produce a usable final answer "
                            "(agent error or empty answer) and were scored with the "
                            "configured fallback raw similarity instead of being embedded. "
                            "See the model output diagnostics table above for per-coalition "
                            "execution_status/execution_error."
                        )
                    else:
                        st.caption("No coalition execution failures for this run.")
                    st.markdown("**Reference final answer (full request)**")
                    st.code(final_answer_scorer_meta["reference_answer"], language="text")

                if lexical_result is not None:
                    st.markdown("**Scorer comparison**")
                    # `raw_log_odds_summary` (computed above for the summary card) is
                    # non-None exactly when the primary scorer's debug record exposes
                    # raw h(∅)/h(N) log-odds -- currently only CalibratedToolLogOddsScorer,
                    # whose normalized Shapley game value is 0 at the empty coalition by
                    # construction. Showing that normalized `empty_score` here would
                    # always read 0.000 regardless of the actual routing evidence, so
                    # this table substitutes the same raw values the summary card already
                    # uses instead of introducing a second, differently-derived number.
                    if raw_log_odds_summary is not None:
                        primary_full_score, primary_empty_score = raw_log_odds_summary
                        st.caption(
                            f"`{primary_label}` normalizes its Shapley game value so "
                            "V(∅) = 0 by construction. The full_score/empty_score below "
                            "are the raw, non-normalized h(N)/h(∅) log-odds instead of "
                            "the normalized game value."
                        )
                    else:
                        primary_full_score, primary_empty_score = full_score, empty_score
                    comparison_rows = [
                        {
                            "scorer": primary_label,
                            "full_score": primary_full_score,
                            "empty_score": primary_empty_score,
                            "top_segment": result["top_label"],
                            "top_attribution": result["top_score"],
                            "strongest_pair": pair_label,
                            "pair_value": pair_value,
                        },
                        {
                            "scorer": lexical_result["label"],
                            "full_score": lexical_result["full_score"],
                            "empty_score": lexical_result["empty_score"],
                            "top_segment": lexical_result["top"],
                            "top_attribution": lexical_result["top_value"],
                            "strongest_pair": lexical_result["pair"],
                            "pair_value": lexical_result["pair_value"],
                        },
                    ]
                    comparison_frame = pd.DataFrame(comparison_rows)
                    st.dataframe(comparison_frame, use_container_width=True, hide_index=True)

                if show_scoring_prompt_preview:
                    st.markdown("**Scoring prompt preview**")
                    scoring_prompt = result.get("scoring_prompt")
                    if scoring_prompt:
                        st.code(scoring_prompt, language="text")
                    else:
                        st.caption(
                            "A separate scoring-prompt preview is not available for this "
                            "scoring backend."
                        )

        st.caption(f"Demo path: `{display_demo_path()}`")


__all__ = [name for name in globals() if not name.startswith("__")]
