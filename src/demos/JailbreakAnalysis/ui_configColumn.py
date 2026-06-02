from __future__ import annotations

import gradio as gr


class ExplanationConfigColumn:
    """A Gradio component group containing configuration dropdowns for the explanation.

    This component wraps a Column containing dropdowns for:
    - Model selection
    - Judge model selection (visible only when scoring_mode is "llm-as-a-judge")
    - Segmentation level
    - Masking strategy
    - Value function / scoring mode
    - Segmentation window size (visible only when segmentation is "semantic")
    - Similarity threshold (visible only when segmentation is "semantic")
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

            self.dropdown_scoring_mode = gr.Dropdown(
                label="Value Function",
                choices=[
                    "logprob",
                    "llm-as-a-judge",
                ],
                value="llm-as-a-judge",
                allow_custom_value=False,
            )

            self.dropdown_judge_model = gr.Dropdown(
                label="Judge Model",
                choices=[
                    "Qwen/Qwen2.5-1.5B-Instruct",
                ],
                value="Qwen/Qwen2.5-1.5B-Instruct",
                allow_custom_value=False,
                visible=True,  # visible by default since default scoring mode is llm-as-a-judge
            )

            self.dropdown_segmentation = gr.Dropdown(
                label="Segmentation level",
                choices=[
                    "word",
                    "token",
                    "sentence",
                    "semantic",
                ],
                value="semantic",
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

            self.text_segmentation_window = gr.Textbox(
                label="Segmentation Window Size",
                value="4",
                visible=True,  # visible by default since default segmentation is "semantic"
            )

            self.text_similarity_threshold = gr.Textbox(
                label="Similarity Threshold",
                value="0.50",
                visible=True,  # visible by default since default segmentation is "semantic"
            )

        # Wire up visibility toggles
        self.dropdown_segmentation.change(
            fn=lambda seg: (
                gr.update(visible=seg == "semantic"),
                gr.update(visible=seg == "semantic"),
            ),
            inputs=self.dropdown_segmentation,
            outputs=[self.text_segmentation_window, self.text_similarity_threshold],
        )

        self.dropdown_scoring_mode.change(
            fn=lambda mode: gr.update(visible=mode == "llm-as-a-judge"),
            inputs=self.dropdown_scoring_mode,
            outputs=self.dropdown_judge_model,
        )