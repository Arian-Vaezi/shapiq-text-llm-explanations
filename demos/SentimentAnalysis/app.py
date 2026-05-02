import gradio as gr
from shapiq.imputer import TextImputer
from shapiq.plot import sentence_plot
from shapiq.game_theory.exact import ExactComputer
import numpy as np
import io
from PIL import Image

MODEL = "distilbert-base-uncased-finetuned-sst-2-english"


def get_prediction(imputer, text):
    result = imputer._classifier(text)[0] 
    return result["label"], result["score"]


def fig_to_numpy(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return np.array(Image.open(buf))


def chat_fn(message, history):
    # 1. Imputer
    imputer = TextImputer(MODEL, message, segmentation="word", mask_strategy="mask")

    # 2. Prediction
    label, score = get_prediction(imputer, message)
    emoji = "😊" if label == "POSITIVE" else "😠"

    # 3. Shapley values
    computer = ExactComputer(imputer)
    sv = computer(index="SV")
    token_strings = imputer._tokenizer.convert_ids_to_tokens(imputer.tokens)

    # 4. Token attribution table (replaces terminal prints)
    table = "| Token | Shapley Value |\n|---|---|\n"
    for tok, val in zip(token_strings, sv.values):
        bar = "🟩" * min(int(abs(val) * 20), 8) if val >= 0 else "🟥" * min(int(abs(val) * 20), 8)
        table += f"| `{tok}` | {val:+.4f} {bar} |\n"

    response = (
        f"### {emoji} {label}  —  confidence: `{score:.3f}`\n\n"
        f"---\n\n"
        f"**Token attributions:**\n\n{table}"
    )

    # 5. Sentence plot
    fig, ax = sentence_plot(sv, token_strings, chars_per_line=80, show=False)
    fig.set_size_inches(10, 4)
    fig.tight_layout()

    return response, fig_to_numpy(fig)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Sentiment Explainer") as demo:
    gr.Markdown("## 🔍 Sentiment Analysis · shapiq Explainer")

    # Plot hidden inside accordion — collapsed by default
    with gr.Accordion("📊 Show sentence plot", open=False):
        plot_out = gr.Image(
            label="Shapley Token Attribution",
        )

    # Chat takes full width
    gr.ChatInterface(
        fn=chat_fn,
        additional_outputs=[plot_out],
        textbox=gr.Textbox(placeholder="Enter a sentence…", lines=2),
    )

if __name__ == "__main__":
    demo.launch()