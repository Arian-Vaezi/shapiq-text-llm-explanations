from __future__ import annotations

import gradio as gr


class ExplanationConfigColumn:
    """A Gradio component group containing configuration dropdowns for the explanation.

    This component wraps a Column containing dropdowns for:
    - Model selection
    - Segmentation level
    - Masking strategy
    """

    def __init__(self) -> None:
        with gr.Column(scale=1, min_width=100):
            gr.Markdown("## Explanation Config")

            self.dropdown_model = gr.Dropdown(
                label="Model for Explanation",
                choices=[
                    "google/gemma-2-2b-it",
                    "microsoft/phi-2",
                    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    # "EleutherAI/gpt-neo-1.3B",
                ],
                value="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                allow_custom_value=False,
            )

            self.dropdown_segmentation = gr.Dropdown(
                label="Segmentation",
                choices=[
                    "word-level",
                    "token-level",
                    "sentence-level",
                ],
                value="word-level",
                allow_custom_value=False,
            )

            self.dropdown_masking = gr.Dropdown(
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
