from __future__ import annotations

import gradio as gr
from transformers import pipeline

from shapiq.gui.configuration_row import ConfigurationRow

# ============================================================
# Model Cache
# ============================================================

MODEL_CACHE: dict[str, object] = {}


# ============================================================
# Load Model
# ============================================================


def load_model(model_name: str):
    """Load and cache HuggingFace pipelines."""
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]

    # --------------------------------------------------------
    # Try loading as chat/text-generation model first
    # --------------------------------------------------------

    try:
        pipe = pipeline(
            task="text-generation",
            model=model_name,
            device_map="auto",
        )

    # --------------------------------------------------------
    # Fallback: classification pipeline
    # --------------------------------------------------------

    except Exception:
        pipe = pipeline(
            task="sentiment-analysis",
            model=model_name,
        )

    MODEL_CACHE[model_name] = pipe

    return pipe


# ============================================================
# Generate Reply
# ============================================================


def generate_reply(
    message: str,
    history: list[dict],
    model_name: str,
) -> str:
    """Generate assistant reply."""
    pipe = load_model(model_name)

    # --------------------------------------------------------
    # Build Conversation Context
    # --------------------------------------------------------

    conversation = ""

    for chat in history:
        role = chat["role"]
        content = chat["content"]

        if role == "user":
            conversation += f"User: {content}\n"

        elif role == "assistant":
            conversation += f"Assistant: {content}\n"

    conversation += f"User: {message}\nAssistant:"

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    try:
        result = pipe(
            conversation,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
        )

        generated_text = result[0]["generated_text"]

        reply = generated_text.split("Assistant:")[-1].strip()

    # --------------------------------------------------------
    # Fallback for classification models
    # --------------------------------------------------------

    except Exception:
        result = pipe(message)

        label = result[0]["label"]
        score = result[0]["score"]

        reply = f"Classification Result\n\nLabel: {label}\nScore: {score:.4f}"

    return reply


# ============================================================
# Explain Button HTML
# ============================================================


def build_explain_button() -> str:
    """Small placeholder explain button."""
    return """
    <div style='display:flex; justify-content:flex-end; margin-top:8px;'>
        <button
            style='
                font-size:12px;
                padding:4px 10px;
                border-radius:8px;
                border:none;
                cursor:pointer;
            '
        >
            Explain
        </button>
    </div>
    """


# ============================================================
# Chat Function
# ============================================================


def chat_fn(
    message: str,
    history: list[dict],
    segmentation: str,
    masking_strategy: str,
    model_name: str,
):
    """Main chat callback."""
    # --------------------------------------------------------
    # Generate Reply
    # --------------------------------------------------------

    reply = generate_reply(
        message=message,
        history=history,
        model_name=model_name,
    )

    # --------------------------------------------------------
    # Add Explain Button Placeholder
    # --------------------------------------------------------

    reply += build_explain_button()

    # --------------------------------------------------------
    # Update Chat History
    # --------------------------------------------------------

    history.append(
        {
            "role": "user",
            "content": message,
        }
    )

    history.append(
        {
            "role": "assistant",
            "content": reply,
        }
    )

    return "", history


# ============================================================
# UI
# ============================================================

with gr.Blocks(title="Jailbreak Analysis Demo") as demo:
    gr.Markdown(
        """
        # Jailbreak Analysis Demo

        Chat with different models and later explain
        outputs with shapiq.
        """
    )

    # ========================================================
    # Chatbot
    # ========================================================

    chatbot = gr.Chatbot(
        height=650,
        label="Chat",
        render_markdown=True,
    )

    # ========================================================
    # Bottom Input Row
    # ========================================================

    with gr.Row():
        user_input = gr.Textbox(
            placeholder="Enter your message...",
            show_label=False,
            scale=20,
        )

        config_row = ConfigurationRow()

    # ========================================================
    # Event Listeners
    # ========================================================

    config_row.send_button.click(
        fn=chat_fn,
        inputs=[
            user_input,
            chatbot,
            *config_row.inputs,
        ],
        outputs=[
            user_input,
            chatbot,
        ],
    )

    user_input.submit(
        fn=chat_fn,
        inputs=[
            user_input,
            chatbot,
            *config_row.inputs,
        ],
        outputs=[
            user_input,
            chatbot,
        ],
    )


# ============================================================
# Launch
# ============================================================

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
