import gradio as gr
from demos.shared.hf_model import HFModelWrapper

# ============================================================
# GLOBAL MODEL CACHE
# ============================================================

MODEL_CACHE = {}


def get_model(model_name: str) -> HFModelWrapper:

    if model_name not in MODEL_CACHE:

        MODEL_CACHE[model_name] = HFModelWrapper(
            model_name=model_name,
            device="cuda",
        )

    return MODEL_CACHE[model_name]


def get_explanation(dropdown_model, dropdown_segmentation, dropdown_masking):
    """
    Simulates generating an explanation based on the chosen config.
    """
    return (
        f"## Explanation\n\n"
        f"- **Model:** {dropdown_model}\n"
        f"- **Segmentation:** {dropdown_segmentation}\n"
        f"- **Masking Strategy:** {dropdown_masking}\n\n"
        f"This section contains the explanation for the last response."
    )

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
        max_new_tokens=128,
    )

    chat_history.append(
        (message, bot_message)
    )

    return (
        chat_history,
        "",
        gr.update(open=False, visible=False),
    )

def show_explanation(
    dropdown_model,
    dropdown_segmentation,
    dropdown_masking,
):
    """
    Generates and reveals the explanation section.
    """

    explanation_content = get_explanation(
        dropdown_model,
        dropdown_segmentation,
        dropdown_masking,
    )

    return (
        gr.update(value=explanation_content, visible=True),
        gr.update(visible=True, open=True),
    )


with gr.Blocks() as demo:
    gr.Markdown("# Jailbreak Analysis Demo")

    with gr.Row():

        # ============================================================
        # Config Panel (Reduced Width)
        # ============================================================
        with gr.Column(scale=0.8, min_width=150):

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
                interactive=True,
            )

            dropdown_segmentation = gr.Dropdown(
                label="Segmentation",
                choices=[
                    "word-level",
                    "token-level",
                ],
                value="word-level",
                allow_custom_value=False,
                interactive=True,
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
                interactive=True,
            )

        # ============================================================
        # Chat Area
        # ============================================================
        with gr.Column(scale=3):

            chatbot = gr.Chatbot(
                label="Chat History",
                height=500,
            )

            with gr.Row():

                msg_input = gr.Textbox(
                    placeholder="Enter your prompt here...",
                    show_label=False,
                    container=False,
                    scale=8,
                )

                send_btn = gr.Button(
                    "Send",
                    scale=1,
                    variant="primary",
                )

            explain_btn = gr.Button(
                "Explain Last Response",
                variant="secondary",
                size="sm",
            )

            with gr.Accordion(
                "Explanation and Analysis",
                open=False,
                visible=False,
            ) as explanation_accordion:

                explanation_md = gr.Markdown(
                    visible=False
                )

    # ============================================================
    # Event Listeners
    # ============================================================

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


if __name__ == "__main__":
    demo.launch()