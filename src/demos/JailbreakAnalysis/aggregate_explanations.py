"""
aggregate_explanations.py

Aggregates the Shapiq explanations from interactions_95428/ into summary_asr.json,
producing a new summary_asr_with_explanations.json.
"""

from __future__ import annotations

import json
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = THIS_DIR / "results"
INTERACTIONS_DIR = THIS_DIR / "interactions_95428"
SUMMARY_ASR_PATH = RESULTS_DIR / "summary_asr.json"
OUTPUT_PATH = RESULTS_DIR / "summary_asr_with_explanations.json"


def main():  # noqa: ANN201
    if not SUMMARY_ASR_PATH.exists():
        print(f"Error: {SUMMARY_ASR_PATH} does not exist.")
        return

    # 1. Load explanations from interactions_95428
    explanations = {}
    if INTERACTIONS_DIR.exists():
        # Explanation files are named like <model_identifier>_<prompt_id>.json
        # and we want to skip *_interaction_values.json and *.png files.
        json_files = list(INTERACTIONS_DIR.glob("*.json"))
        for file_path in json_files:
            if file_path.name.endswith("_interaction_values.json"):
                continue

            try:
                with file_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                settings = data.get("settings", {})
                model = settings.get("model")
                if not model:
                    continue

                # Extract prompt_id from the filename
                # e.g., "mistralai_Mistral-7B-Instruct-v0.3_c1_dan_malware.json"
                # The model identifier matches the model name with / replaced by _
                # We can construct the prompt_id by stripping the model identifier prefix and the extension
                model_prefix = model.replace("/", "_")
                filename_stem = file_path.stem
                if filename_stem.startswith(model_prefix + "_"):
                    prompt_id = filename_stem[len(model_prefix) + 1 :]
                else:
                    # Fallback matching
                    continue

                explanations[(model, prompt_id)] = {
                    "players": data.get("players"),
                    "player_values": data.get("player_values"),
                    "all_interactions": data.get("all_interactions"),
                    "top_interaction_pairs": data.get("top_interaction_pairs"),
                }
            except Exception as e:  # noqa: BLE001
                print(f"Failed to read explanation file {file_path}: {e}")

    print(f"Loaded {len(explanations)} explanations from {INTERACTIONS_DIR}")

    # 2. Load summary_asr.json
    with SUMMARY_ASR_PATH.open("r", encoding="utf-8") as f:
        summary_data = json.load(f)

    # 3. Merge explanations into summary data
    merged_count = 0
    for entry in summary_data:
        model = entry.get("model")
        prompt_id = entry.get("prompt_id")

        key = (model, prompt_id)
        if key in explanations:
            explanation = explanations[key]
            entry["players"] = explanation["players"]
            entry["player_values"] = explanation["player_values"]
            entry["all_interactions"] = explanation["all_interactions"]
            entry["top_interaction_pairs"] = explanation["top_interaction_pairs"]
            merged_count += 1

    print(
        f"Merged explanations into {merged_count} out of {len(summary_data)} summary configurations."
    )

    # 4. Save to summary_asr_with_explanations.json
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"Saved aggregated results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
