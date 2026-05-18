import gradio as gr

from demos.shared.hf_model import HFModelWrapper
from demos.JailbreakAnalysis import JailbreakGame

# ============================================================
# MODEL CACHE
# ============================================================

MODEL_CACHE = {}

PRELOAD_MODELS = [
    "google/gemma-2-2b-it",
]


# ============================================================
# PRELOAD MODELS
# ============================================================

def preload_models():

    print("Preloading models...")

    MODEL_CACHE["google/gemma-2-2b-it"] = HFModelWrapper(
        model_name="google/gemma-2-2b-it",
        device="cuda",
    )

    print("Finished preloading.")


# ============================================================
# GET MODEL
# ============================================================

def get_model(model_name: str) -> HFModelWrapper:

    # already loaded
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]

    # lazy-load non-preloaded models
    print(f"Lazily loading {model_name}")

    MODEL_CACHE[model_name] = HFModelWrapper(
        model_name=model_name,
        device="cuda",
    )

    return MODEL_CACHE[model_name]


# ============================================================
# EXPLANATION
# ============================================================

def get_explanation(
    dropdown_model,
    dropdown_segmentation,
    dropdown_masking,
):

    return (
        f"## Explanation\n\n"
        f"- Model: {dropdown_model}\n"
        f"- Segmentation: {dropdown_segmentation}\n"
        f"- Masking: {dropdown_masking}\n\n"
        f"Placeholder explanation output."
    )


# ============================================================
# CHAT RESPONSE
# ============================================================

def respond(
    message,
    chat_history,
    dropdown_model,
    dropdown_segmentation,
    dropdown_masking,
):

    model = get_model(dropdown_model)

    bot_message = model.generate_text(
        prompt=message,
        max_new_tokens=16,
    )

    chat_history.append(
        (message, bot_message)
    )

    return (
        chat_history,
        "",
        gr.update(visible=True),   # show explain button
        gr.update(open=False, visible=False),
    )


# ============================================================
# SHOW EXPLANATION
# ============================================================

def show_explanation(
    dropdown_model,
    dropdown_segmentation,
    dropdown_masking,
):

    explanation_content = get_explanation(
        dropdown_model,
        dropdown_segmentation,
        dropdown_masking,
    )

    return (
        gr.update(value=explanation_content, visible=True),
        gr.update(visible=True, open=True),
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

        with gr.Column(scale=1, min_width=100):

            gr.Markdown("## Explaination Config")

            dropdown_model = gr.Dropdown(
                label="Model for Explanation",
                choices=[
                    "google/gemma-2-2b-it",
                     "microsoft/phi-2",
                     "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                     "EleutherAI/gpt-neo-1.3B",

                ],
                value="google/gemma-2-2b-it",
                allow_custom_value=False,
            )

            dropdown_segmentation = gr.Dropdown(
                label="Segmentation",
                choices=[
                    "word-level",
                    "token-level",
                ],
                value="word-level",
                allow_custom_value=False,
            )

            dropdown_masking = gr.Dropdown(
                label="Masking Strategy",
                choices=[
                    "remove",
                    "mask",
                    "distributional sampling",
                    "generative infilling",
                ],
                value="remove",
                allow_custom_value=False,
            )

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

            # =================================================
            # Explain button BELOW chat
            # =================================================

            explain_btn = gr.Button(
                "Explain Last Response",
                visible=False,
                variant="secondary",
            )

            # =================================================
            # Explanation Accordion
            # =================================================

            with gr.Accordion(
                "Explanation and Analysis",
                open=False,
                visible=False,
            ) as explanation_accordion:

                explanation_md = gr.Markdown(
                    visible=False
                )

    # ========================================================
    # EVENTS
    # ========================================================

    msg_input.submit(
        respond,
        [
            msg_input,
            chatbot,
            dropdown_model,
            dropdown_segmentation,
            dropdown_masking,
        ],
        [
            chatbot,
            msg_input,
            explain_btn,
            explanation_accordion,
        ],
    )

    send_btn.click(
        respond,
        [
            msg_input,
            chatbot,
            dropdown_model,
            dropdown_segmentation,
            dropdown_masking,
        ],
        [
            chatbot,
            msg_input,
            explain_btn,
            explanation_accordion,
        ],
    )

    explain_btn.click(
        show_explanation,
        [
            dropdown_model,
            dropdown_segmentation,
            dropdown_masking,
        ],
        [
            explanation_md,
            explanation_accordion,
        ],
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    preload_models()
    demo.launch()