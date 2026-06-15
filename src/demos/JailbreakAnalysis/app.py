from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")  # non-interactive backend — required for server-side rendering
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import shapiq
from demos.JailbreakAnalysis.JailbreakAnalysisGame import JailbreakGame
from demos.shared.hf_model import HFModelWrapper, is_api_model_name
from shapiq.plot import sentence_interaction_heatmap

# Above this many players, second-order interactions become slow to approximate
# (quadratic number of pairs) and the plots get unreadable. We still allow it,
# but warn the user first.
SECOND_ORDER_PLAYER_WARN = 12

st.set_page_config(
    page_title="Shapiq Jailbreak Explainability",
    page_icon="🔍",
    layout="wide",
)


# --- Caching & State ---
@st.cache_resource
def get_model(model_name: str, temperature: float = 0.0) -> object:
    return HFModelWrapper(model_name=model_name, device="cuda", temperature=temperature)


def top_interaction_pairs(
    result: shapiq.InteractionValues,
    players: np.ndarray,
    top_k: int = 8,
) -> list[tuple[str, str, float]]:
    """Return the strongest order-2 k-SII interactions sorted by absolute value.

    Each tuple is ``(player_i, player_j, interaction_value)``. Positive values are
    synergies (the pair matters more together), negative values are redundancies.
    """
    order2 = {k: v for k, v in result.interaction_lookup.items() if len(k) == 2}
    ranked = sorted(order2.items(), key=lambda kv: abs(result.values[kv[1]]), reverse=True)[:top_k]
    return [
        (str(players[idx[0]]), str(players[idx[1]]), float(result.values[pos]))
        for idx, pos in ranked
    ]


# Multipliers applied to the recommended budget by the "Approximation quality" control.
QUALITY_MULTIPLIERS = {"Fast": 0.5, "Auto (recommended)": 1.0, "Thorough": 2.0}


def recommended_budget(n_players: int, *, second_order: bool, multiplier: float = 1.0) -> int:
    """Pick a coalition budget that scales with players and interaction order.

    The regression estimates one coefficient per attribution: ``n`` for first-order,
    ``n + n(n-1)/2`` for order-2 (k-SII). We oversample ~4x for stability, keep the
    system identifiable, and never exceed the exact number of coalitions (2**n), beyond
    which KernelSHAP's border-trick makes extra budget pointless.
    """
    n_coeff = n_players + n_players * (n_players - 1) // 2 if second_order else n_players
    budget = int((4 * n_coeff + 2) * multiplier)
    budget = max(budget, n_coeff + 2)  # keep the regression identifiable
    if n_players <= 20:  # for small games, exact is cheap — don't waste calls past 2**n
        budget = min(budget, 2**n_players)
    return budget


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "llama-3.3-70b-versatile"

# --- Sidebar Config ---
st.sidebar.title("Configuration")

model_choices = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "openai/gpt-oss-safeguard-20b",
    "qwen/qwen3-32b",
    "gemini-2.5-flash",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]

selected_model = st.sidebar.selectbox("Model", model_choices, index=0)
st.session_state.selected_model = selected_model

temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)

# --- Navigation ---
tab_inference, tab_explanation = st.tabs(["💬 Inference", "🔍 Explanation"])

# --- Inference Tab ---
with tab_inference:
    st.markdown("## Chat with the Model")
    st.markdown(
        f"**Current Model:** `{st.session_state.selected_model}` | **Temperature:** `{temperature}`"
    )

    # Display chat
    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])

    prompt = st.chat_input("Enter your prompt...")
    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            try:
                model = get_model(st.session_state.selected_model, temperature=temperature)
                # Note: caching the model uses the previous temp, so we explicitly pass it below.
                stream = model.generate_text_stream(
                    prompt=prompt, chat=True, temperature=temperature
                )
                response = st.write_stream(stream)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            except Exception as e:  # noqa: BLE001
                st.error(f"Error during inference: {e}")

# --- Explanation Tab ---
with tab_explanation:
    st.markdown("## Explanation with shapiq")
    st.markdown(
        "Analyze the compliance of the model based on Shapley values. This evaluates how individual parts of your prompt influence the model's output compliance."
    )

    # Explanation config
    with st.expander("Explanation Settings", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            scoring_mode = st.selectbox("Value Function", ["llm-as-a-judge", "logprob"], index=0)

            judge_model = None
            if scoring_mode == "llm-as-a-judge":
                default_judge_index = (
                    model_choices.index(st.session_state.selected_model)
                    if st.session_state.selected_model in model_choices
                    else model_choices.index("Qwen/Qwen2.5-1.5B-Instruct")
                )
                judge_model = st.selectbox("Judge Model", model_choices, index=default_judge_index)

            masking_strategy = st.selectbox(
                "Masking Strategy", ["remove", "mask", "distributional", "generative"], index=0
            )

        with col2:
            segmentation = st.selectbox(
                "Segmentation Level", ["semantic", "sentence", "word", "token"], index=0
            )

            explanation_order = st.radio(
                "Explanation Order",
                ["First-order only", "Second-order (k-SII)"],
                index=0,
                help=(
                    "First-order shows each segment's individual Shapley value. "
                    "Second-order additionally estimates pairwise k-SII interactions "
                    "(synergy/redundancy between segments) and plots them."
                ),
            )

            approx_quality = st.selectbox(
                "Approximation Quality",
                list(QUALITY_MULTIPLIERS),
                index=list(QUALITY_MULTIPLIERS).index("Auto (recommended)"),
                help=(
                    "Coalition budget scales automatically with the number of segments "
                    "and the interaction order. Each coalition is one (or two) LLM calls, "
                    "so 'Thorough' is more accurate but slower; 'Fast' is cheaper."
                ),
            )

            semantic_window = 4
            semantic_threshold = 0.50
            if segmentation == "semantic":
                semantic_window = st.number_input("Segmentation Window Size", value=4, min_value=1)
                semantic_threshold = st.number_input(
                    "Similarity Threshold", value=0.50, min_value=0.0, max_value=1.0, step=0.05
                )

    # Input for explanation
    explain_prompt = st.text_area("Prompt to explain", height=100)

    if st.button("Explain with shapiq", type="primary"):
        if not explain_prompt:
            st.warning("Please enter a prompt to explain.")
        elif scoring_mode == "logprob" and is_api_model_name(st.session_state.selected_model):
            st.error(
                "Logprob scoring is only available for local HuggingFace models. "
                "Use llm-as-a-judge for Groq/Gemini API models."
            )
        else:
            with st.status("Running explanation...") as status:
                try:
                    st.write("Loading model...")
                    # We pass temperature=0.0 for deterministic explanations
                    model = get_model(st.session_state.selected_model, temperature=0.0)

                    st.write("Initializing Jailbreak Game...")
                    game = JailbreakGame(
                        model_name=st.session_state.selected_model,
                        input_text=explain_prompt,
                        scoring_mode=scoring_mode,
                        mask_strategy=masking_strategy,
                        segmentation=segmentation,
                        device="cuda",
                        hf_model=model,
                        judge_model_name=judge_model or "Qwen/Qwen2.5-1.5B-Instruct",
                        semantic_window=int(semantic_window),
                        semantic_threshold=float(semantic_threshold),
                    )

                    st.write("Calculating compliance score...")
                    full_coalition = np.ones((1, game.n_players))
                    compliance_score = float(game.value_function(full_coalition)[0])

                    second_order = explanation_order.startswith("Second-order")
                    if second_order and game.n_players > SECOND_ORDER_PLAYER_WARN:
                        st.warning(
                            f"Second-order analysis on {game.n_players} players means "
                            f"{game.n_players * (game.n_players - 1) // 2} pairs. The budget "
                            "(and so the number of LLM calls) grows quadratically, making this "
                            "slow, and the plots get crowded. Consider a coarser segmentation "
                            "(e.g. sentence) or a shorter prompt."
                        )

                    budget = recommended_budget(
                        game.n_players,
                        second_order=second_order,
                        multiplier=QUALITY_MULTIPLIERS[approx_quality],
                    )

                    if second_order:
                        st.write(
                            f"Running Shapiq approximation (k-SII, order 2, budget={budget})..."
                        )
                        approx = shapiq.KernelSHAPIQ(
                            n=game.n_players,
                            index="k-SII",
                            max_order=2,
                            random_state=42,
                        )
                    else:
                        st.write(f"Running Shapiq approximation (budget={budget})...")
                        approx = shapiq.KernelSHAP(n=game.n_players, random_state=42)
                    result = approx.approximate(budget=budget, game=game)

                    player_values = np.array([float(result[(i,)]) for i in range(game.n_players)])

                    status.update(label="Explanation complete!", state="complete", expanded=False)

                    # Rendering results
                    st.success(
                        f"**Model:** {st.session_state.selected_model}  |  **Compliance Score:** `{compliance_score:+.4f}`"
                    )

                    st.markdown("### Shapley Values")

                    # Custom HTML for colorized bars
                    html = "<table><thead><tr><th>Player</th><th>Shapley Value</th><th>Contribution</th></tr></thead><tbody>"
                    for p, val in zip(game.players, player_values, strict=False):
                        bar_len = min(int(abs(val) * 15), 10)
                        bar_color = "#10b981" if val >= 0 else "#ef4444"
                        bar = f'<span style="color:{bar_color}">{"█" * bar_len}</span>'
                        html += (
                            f"<tr><td><code>{p}</code></td><td>{val:+.4f}</td><td>{bar}</td></tr>"
                        )
                    html += "</tbody></table>"

                    st.html(html)

                    # --- Second-order interactions ---
                    if second_order:
                        st.markdown("### Top Interaction Pairs (k-SII)")
                        pairs = top_interaction_pairs(result, game.players, top_k=8)
                        if not pairs:
                            st.info("No pairwise interactions to display.")
                        else:
                            int_html = (
                                "<table><thead><tr><th>Pair</th><th>k-SII</th>"
                                "<th>Type</th></tr></thead><tbody>"
                            )
                            for p1, p2, val in pairs:
                                kind = "🟢 synergy" if val >= 0 else "🔵 redundancy"
                                int_html += (
                                    f"<tr><td><code>{p1}</code> + <code>{p2}</code></td>"
                                    f"<td>{val:+.4f}</td><td>{kind}</td></tr>"
                                )
                            int_html += "</tbody></table>"
                            st.html(int_html)
                            st.caption(
                                "Synergy: the pair influences compliance more together than "
                                "individually. Redundancy: the segments overlap/compete."
                            )

                        players = list(game.players)

                        col_hm, col_net = st.columns(2)
                        with col_hm:
                            st.markdown("#### Interaction Heatmap")
                            try:
                                fig, _ = sentence_interaction_heatmap(result, players, show=False)
                                st.pyplot(fig)
                                plt.close(fig)
                            except Exception as e:  # noqa: BLE001
                                st.info(f"Heatmap unavailable: {e}")
                        with col_net:
                            st.markdown("#### Interaction Network")
                            try:
                                net = result.plot_network(feature_names=players, show=False)
                                if net is None:
                                    st.info("Network plot unavailable.")
                                else:
                                    fig = net[0] if isinstance(net, tuple) else net
                                    st.pyplot(fig)
                                    plt.close(fig)
                            except Exception as e:  # noqa: BLE001
                                st.info(f"Network plot unavailable: {e}")

                except Exception as e:  # noqa: BLE001
                    status.update(label="Error during explanation.", state="error")
                    st.error(f"Error during explanation: {e}")
