from __future__ import annotations

import gradio as gr


SEGMENTATION_OPTIONS = [
    "token",
    "word",
]

MASKING_OPTIONS = [
    "mask",
    "remove",
]

MODEL_OPTIONS = [
    "google/gemma-2b-it",
    "microsoft/Phi-3-mini-4k-instruct",
    "distilbert-base-uncased-finetuned-sst-2-english",
]


class ConfigurationRow:
    """Reusable configuration row component."""

    def __init__(self) -> None:

        with gr.Row():

            self.segmentation_dropdown = gr.Dropdown(
                choices=SEGMENTATION_OPTIONS,
                value="word",
                label="Segmentation",
                scale=2,
            )

            self.masking_dropdown = gr.Dropdown(
                choices=MASKING_OPTIONS,
                value="mask",
                label="Mask Strategy",
                scale=2,
            )

            self.model_dropdown = gr.Dropdown(
                choices=MODEL_OPTIONS,
                value=MODEL_OPTIONS[0],
                label="Model",
                scale=4,
            )

            self.send_button = gr.Button(
                "➤",
                variant="primary",
                scale=1,
            )

    @property
    def inputs(self) -> list:
        """Return all interactive inputs."""

        return [
            self.segmentation_dropdown,
            self.masking_dropdown,
            self.model_dropdown,
        ]