"""Gradio web application for the Jailbreak Analysis demo.

This file contains the UI layout and interaction handlers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gradio as gr
import numpy as np

from demos.JailbreakAnalysis.ui_configColumn import ExplanationConfigColumn
from demos.shared.hf_model import HFModelWrapper

if TYPE_CHECKING:
    from collections.abc import Iterator

# ============================================================
# MODEL CACHE
# ============================================================

MODEL_CACHE = {}

PRELOAD_MODELS = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # "google/gemma-2-2b-it",
    # "microsoft/phi-2",
]


# ============================================================
# PRELOAD MODELS
# ============================================================


def preload_models() -> None:
    """Preloads the configured models in PRELOAD_MODELS into the MODEL_CACHE."""
    print("Preloading models...")

    for model_name in PRELOAD_MODELS:
        if model_name in MODEL_CACHE:
            continue
        MODEL_CACHE[model_name] = HFModelWrapper(
            model_name=model_name,
            device="cuda",
        )

    print("Finished preloading.")


# ============================================================
# GET MODEL
# ============================================================


def get_model(model_name: str) -> HFModelWrapper:
    """Retrieves a model from the cache, lazy-loading it if not present."""
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]

    print(f"Lazily loading {model_name}")

    MODEL_CACHE[model_name] = HFModelWrapper(
        model_name=model_name,
        device="cuda",
    )

    return MODEL_CACHE[model_name]


# ============================================================
# CHAT RESPONSE
# ============================================================


def respond(
    message: str,
    chat_history: list[tuple[str, str]],
    dropdown_model: str,
) -> Iterator[tuple[object, object, object, object]]:
    print("[respond] Streaming response.")

    model = get_model(dropdown_model)

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": ""})

    for token in model.generate_text_stream(prompt=message, max_new_tokens=128, chat=True):
        chat_history[-1]["content"] += token
        yield (
            chat_history,
            "",
            gr.update(visible=True),
            gr.update(open=False, visible=False),
        )

    print("[respond] Done.")


# ============================================================
# BUILD EXPLANATION HTML
# ============================================================


def build_explanation_html(
    message: str,
    dropdown_model: str,
    compliance_score: float,
    players: list[str],
    sv_values: np.ndarray,
) -> str:
    sv_rows = ""

    for player, val in zip(players, sv_values, strict=False):
        bar_len = min(int(abs(val) * 15), 10)
        bar_color = "#10b981" if val >= 0 else "#ef4444"
        bar = f'<span style="color:{bar_color}">{"█" * bar_len}</span>'

        sv_rows += f"<tr><td>{player}</td><td>{val:+.4f}</td><td>{bar}</td></tr>\n"

    return f"""
<h2>Shapiq Explanation</h2>
<p><b>Model:</b> {dropdown_model}</p>
<p><b>Input:</b> {message}</p>
<p><b>Compliance Score:</b> {compliance_score:+.4f}</p>

<h3>Shapley Values</h3>
<table>
  <thead>
    <tr>
      <th>Players</th>
      <th>Shapley Value</th>
      <th>Contribution</th>
    </tr>
  </thead>
  <tbody>
    {sv_rows}
  </tbody>
</table>
"""


# ============================================================
# SHOW EXPLANATION
# ============================================================


def show_explanation(
    message: str,
    dropdown_model: str,
    dropdown_segmentation: str,
    dropdown_masking: str,
) -> Iterator[tuple[object, object]]:
    import shapiq
    from demos.JailbreakAnalysis.JailbreakAnalysisGame import JailbreakGame

    print(f"[show_explanation] Starting explanation for: '{message[:60]}...'")
    print(f"    - Model: {dropdown_model}")
    print(f"    - Segmentation: {dropdown_segmentation}")
    print(f"    - Masking: {dropdown_masking}")

    yield (
        gr.update(value="<p><i>⏳ Explanation in progress...</i></p>", visible=True),
        gr.update(visible=True, open=True),
    )
    cached_model = MODEL_CACHE.get(dropdown_model)
    print("[show_explanation] creating game instance")
    game = JailbreakGame(
        model_name=dropdown_model,
        input_text=message,
        mask_strategy=dropdown_masking,
        segmentation=dropdown_segmentation,
        device="cuda",
        hf_model=cached_model,
    )
    print(game.players.tolist())

    # Compliance score: value of the full coalition (all tokens present)
    full_coalition = np.ones((1, game.n_players))
    compliance_score = float(game.value_function(full_coalition)[0])
    print(f"[show_explanation] calculating compliance score: {compliance_score}")

    approx = shapiq.KernelSHAP(
        n=game.n_players,
        random_state=42,
    )
    print("[show_explanation] running approximation")
    result = approx.approximate(budget=10, game=game)

    print("[show_explanation] building html")
    html = build_explanation_html(
        message=message,
        dropdown_model=dropdown_model,
        compliance_score=compliance_score,
        players=game.players.tolist(),
        sv_values=result.values,
    )

    yield (
        gr.update(value=html, visible=True),  # explanation_html
        gr.update(visible=True, open=True),  # explanation_accordion
    )


# ============================================================
# UI
# ============================================================

with gr.Blocks() as demo:
    gr.Markdown("# Shapiq Jailbreak Explainability Demo")

    with gr.Row():
        # ====================================================
        # CONFIG
        # ====================================================

        config_col = ExplanationConfigColumn()
        dropdown_model = config_col.dropdown_model
        dropdown_segmentation = config_col.dropdown_segmentation
        dropdown_masking = config_col.dropdown_masking

        # ====================================================
        # CHAT
        # ====================================================

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Chat History",
                height=500,
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Enter prompt...",
                    show_label=False,
                    container=False,
                    scale=8,
                )

                send_btn = gr.Button(
                    "Send",
                    variant="primary",
                    scale=1,
                )

            explain_btn = gr.Button(
                "Explain with shapiq",
                visible=True,
                variant="secondary",
            )

            with gr.Accordion(
                "Explanation and Analysis",
                open=False,
                visible=True,
            ) as explanation_accordion:
                explanation_html = gr.HTML(visible=False)

    # ========================================================
    # EVENTS
    # ========================================================

    respond_inputs = [
        msg_input,
        chatbot,
        dropdown_model,
    ]

    respond_outputs = [
        chatbot,
        msg_input,
        explain_btn,
        explanation_accordion,
    ]

    # explain reads directly from msg_input — independent of chat
    explanation_inputs = [
        msg_input,
        dropdown_model,
        dropdown_segmentation,
        dropdown_masking,
    ]

    explanation_outputs = [
        explanation_html,
        explanation_accordion,
    ]

    msg_input.submit(respond, respond_inputs, respond_outputs)
    send_btn.click(respond, respond_inputs, respond_outputs)
    explain_btn.click(show_explanation, explanation_inputs, explanation_outputs)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    preload_models()
    demo.launch()
