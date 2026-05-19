"""Gradio web application for the Jailbreak Analysis demo.

This file contains the UI layout and interaction handlers.
"""

from __future__ import annotations

import gradio as gr

from demos.JailbreakAnalysis.ui_configColumn import ExplanationConfigColumn
from demos.shared.hf_model import HFModelWrapper

# ============================================================
# MODEL CACHE
# ============================================================

MODEL_CACHE = {}

PRELOAD_MODELS = [
    #"google/gemma-2-2b-it",
    #"microsoft/phi-2",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
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
    dropdown_segmentation: str,
    dropdown_masking: str,
):
    """Generates the chat response using the selected model."""
    _ = dropdown_segmentation
    _ = dropdown_masking

    model = get_model(dropdown_model)

    bot_message = model.generate_text(
        prompt=message,
        max_new_tokens=64,
        chat=True,
    )

    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": bot_message})

    return (
        chat_history,                          # chatbot
        "",                                    # msg_input (clear)
        gr.update(visible=True),               # explain_btn
        gr.update(open=False, visible=False),  # explanation_accordion (reset)
    )


# ============================================================
# SHOW EXPLANATION
# ============================================================


def show_explanation(
    message: str,
    dropdown_model: str,
    dropdown_segmentation: str,
    dropdown_masking: str,
):
    from demos.JailbreakAnalysis.JailbreakAnalysisGame import JailbreakGame
    import shapiq

    cached_model = MODEL_CACHE.get(dropdown_model)

    game = JailbreakGame(
        model_name=dropdown_model,
        input_text=message,
        mask_strategy=dropdown_masking,
        segmentation=dropdown_segmentation,
        device="cuda",
        hf_model=cached_model,
    )

    approx = shapiq.KernelSHAP(
        n=game.n_players,
        random_state=42,
    )

    result = approx.approximate(budget=100, game=game)

    explanation_text = (
        f"## Shapiq Explanation\n\n"
        f"**Model:** {dropdown_model}  \n"
        f"**Input:** {message}  \n"
        f"**Shapley values:** `{result.values}`"
    )

    return (
        gr.update(value=explanation_text, visible=True),  # explanation_md
        gr.update(visible=True, open=True),               # explanation_accordion
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
                explanation_md = gr.Markdown(visible=False)

    # ========================================================
    # EVENTS
    # ========================================================

    respond_inputs = [
        msg_input,
        chatbot,
        dropdown_model,
        dropdown_segmentation,
        dropdown_masking,
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
        explanation_md,
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