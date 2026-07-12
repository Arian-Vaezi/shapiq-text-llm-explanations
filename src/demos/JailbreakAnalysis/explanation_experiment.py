"""
Runs 15 (prompt/response pairs) x 3 (scoring modes) = 45 JailbreakGame
explanation experiments and saves first-order Shapley values.

Assumes jailbreak_game.py (JailbreakGame) is importable from this location.
Adjust SUMMARY_PATH / load_target_entries() to match your actual JSON schema.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import shapiq

from demos.JailbreakAnalysis.JailbreakAnalysisGame import JailbreakGame

# =====================================================
# Config
# =====================================================
SUMMARY_PATH = Path("src/demos/JailbreakAnalysis/results/summary_asr.json")
OUTPUT_PATH = Path("src/demos/JailbreakAnalysis/results/explanation_experiment_data.json")

TARGET_MODEL = "google/gemma-4-e4b-it"
TARGET_TEMPERATURE = 0.7
JUDGE_MODEL = "openai/gpt-oss-safeguard-20b"
SCORING_MODES = ["abs-logprob", "contra-logprob", "llm-as-a-judge"]
N_ENTRIES = 15


# =====================================================
# Budget (copied from your snippet)
# =====================================================
def recommended_budget(n_players: int, *, second_order: bool, multiplier: float = 1.0) -> int:
    """Pick a coalition budget that scales with players and interaction order."""
    n_coeff = n_players + n_players * (n_players - 1) // 2 if second_order else n_players
    budget = int((4 * n_coeff + 2) * multiplier)
    budget = max(budget, n_coeff + 2)
    if n_players <= 20:
        budget = min(budget, 2**n_players)
    return budget


# =====================================================
# Load the 15 target entries
# =====================================================
def load_target_entries(path: Path, n: int = N_ENTRIES) -> list[dict]:
    with path.open() as f:
        data = json.load(f)

    # TODO: adjust if summary_asr.json isn't a flat list of records.
    # Common alternatives: {"results": [...]}, {"data": [...]}, nested per-model dict.
    entries = data if isinstance(data, list) else data.get("results", data.get("data", []))

    filtered = [
        e
        for e in entries
        if e.get("model") == TARGET_MODEL and e.get("temperature") == TARGET_TEMPERATURE
    ]

    if len(filtered) < n:
        msg = (
            f"Expected at least {n} entries for model={TARGET_MODEL!r}, "
            f"temperature={TARGET_TEMPERATURE}, found {len(filtered)}."
        )
        raise ValueError(msg)

    return filtered[:n]


# =====================================================
# Run one game -> first-order Shapley values
# =====================================================
def run_single_game(prompt_text: str, response: str, scoring_mode: str) -> dict:
    game = JailbreakGame(
        model_name=TARGET_MODEL,
        input_text=prompt_text,
        scoring_mode=scoring_mode,
        judge_model_name=JUDGE_MODEL,
        model_response=response,
    )

    budget = recommended_budget(game.n_players, second_order=False)

    approx = shapiq.KernelSHAP(n=game.n_players, random_state=42)
    result = approx.approximate(budget=budget, game=game)

    # TODO: confirm attribute name for your shapiq version.
    # Common options: result.values, result.get_first_order_values(), result.dict_values
    shapley_values = np.asarray(result.values).tolist()

    return {
        "n_players": game.n_players,
        "budget": budget,
        "players": [str(p) for p in game.players],
        "shapley_values": shapley_values,
    }


# =====================================================
# Main experiment loop: 15 entries x 3 scoring modes
# =====================================================
def run_experiment(entries: list[dict]) -> list[dict]:
    results = []
    total = len(entries) * len(SCORING_MODES)
    step = 0

    for i, entry in enumerate(entries):
        prompt_text = entry["prompt_text"]
        response = entry["response"]

        for scoring_mode in SCORING_MODES:
            step += 1
            print(f"[{step}/{total}] entry={i} scoring_mode={scoring_mode}")

            game_result = run_single_game(prompt_text, response, scoring_mode)

            results.append(
                {
                    "entry_index": i,
                    "model": TARGET_MODEL,
                    "temperature": TARGET_TEMPERATURE,
                    "judge_model": JUDGE_MODEL,
                    "prompt_text": prompt_text,
                    "response": response,
                    "scoring_mode": scoring_mode,
                    **game_result,
                }
            )

    return results


def main() -> None:
    entries = load_target_entries(SUMMARY_PATH)
    results = run_experiment(entries)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} game results ({len(entries)}x{len(SCORING_MODES)}) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()