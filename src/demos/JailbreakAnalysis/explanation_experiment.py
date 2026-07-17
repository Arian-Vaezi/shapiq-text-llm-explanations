"""
Runs 15 (prompt/response pairs) x 3 (scoring modes) = 45 JailbreakGame
explanation experiments and saves first-order Shapley values, full-coalition
scores, baseline values, and pairwise (k-SII) interaction values.

Requirements:
- A CUDA GPU with enough VRAM for `google/gemma-4-e4b-it` (~16GB) AND
  `openai/gpt-oss-safeguard-20b` (~40GB+ in fp16) loaded at the same time,
  since the llm-as-a-judge scoring mode needs both. A100 (40/80GB) or similar
  is recommended; a single 16GB GPU will likely OOM once the judge model loads.
- transformers, torch, shapiq, and this project's demos/shared/* modules
  importable from the working directory.

Assumes jailbreak_game.py (JailbreakGame) is importable from this location.
Adjust SUMMARY_PATH / load_target_entries() to match your actual JSON schema.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from JailbreakAnalysisGame import JailbreakGame

import shapiq
from demos.shared.hf_model import HFModelWrapper

# =====================================================
# Config
# =====================================================
SUMMARY_PATH = Path("src/demos/JailbreakAnalysis/results/summary_asr.json")
OUTPUT_PATH = Path("src/demos/JailbreakAnalysis/results/explanation_experiment_data.json")

TARGET_MODEL = "google/gemma-4-e4b-it"
TARGET_TEMPERATURE = 0.7
JUDGE_MODEL = "openai/gpt-oss-safeguard-20b"
N_ENTRIES = 15
DEVICE = "cuda"

# Full run: all three scoring modes, including llm-as-a-judge.
RUN_JUDGE_MODE = True
SCORING_MODES = ["abs-logprob", "contra-logprob"] + (["llm-as-a-judge"] if RUN_JUDGE_MODE else [])


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
        entries: list[dict] = json.load(f)  # flat list of records

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
def run_single_game(
    prompt_text: str,
    response: str,
    scoring_mode: str,
    hf_model: HFModelWrapper,
) -> dict:
    game = JailbreakGame(
        model_name=TARGET_MODEL,
        input_text=prompt_text,
        scoring_mode=scoring_mode,
        judge_model_name=JUDGE_MODEL,
        model_response=response,
        device=DEVICE,
        hf_model=hf_model,  # reuse already-loaded model instead of reloading per game
    )

    # -------------------------
    # 1. Final (full-coalition) value function score
    # -------------------------
    full_coalition = np.ones((1, game.n_players), dtype=bool)
    full_value = float(game.value_function(full_coalition)[0])

    # -------------------------
    # 2 & 3. First-order Shapley values (+ baseline, as a byproduct)
    # -------------------------
    budget_order1 = recommended_budget(game.n_players, second_order=False)
    approx_order1 = shapiq.KernelSHAP(n=game.n_players, random_state=42)
    result_order1 = approx_order1.approximate(budget=budget_order1, game=game)

    baseline_value = float(result_order1.baseline_value)
    shapley_values = np.asarray(result_order1.values).tolist()

    # -------------------------
    # 4. Second-order (pairwise) k-SII interaction values
    # -------------------------
    budget_order2 = recommended_budget(game.n_players, second_order=True)
    approx_order2 = shapiq.KernelSHAPIQ(
        n=game.n_players,
        index="k-SII",
        max_order=2,
        random_state=42,
    )
    result_order2 = approx_order2.approximate(budget=budget_order2, game=game)

    order2_lookup = {k: v for k, v in result_order2.interaction_lookup.items() if len(k) == 2}
    interaction_values = [
        {
            "players": [str(game.players[i]) for i in player_idx],
            "value": float(result_order2.values[pos]),
        }
        for player_idx, pos in order2_lookup.items()
    ]

    return {
        "n_players": game.n_players,
        "players": [str(p) for p in game.players],
        "full_coalition_value": full_value,
        "baseline_value": baseline_value,
        "budget_order1": budget_order1,
        "shapley_values": shapley_values,
        "budget_order2": budget_order2,
        "interaction_values": interaction_values,
    }


# =====================================================
# Main experiment loop: 15 entries x 3 scoring modes
# =====================================================
def run_experiment(entries: list[dict]) -> list[dict]:
    results = []
    total = len(entries) * len(SCORING_MODES)
    step = 0

    # Load the main model once and reuse it across all 15*len(SCORING_MODES) games,
    # instead of reloading it fresh inside every JailbreakGame(...) construction.
    print(f"Loading {TARGET_MODEL} once for reuse across all games...")
    shared_model = HFModelWrapper(TARGET_MODEL, device=DEVICE)

    for i, entry in enumerate(entries):
        prompt_text = entry["prompt_text"]
        response = entry["response"]

        for scoring_mode in SCORING_MODES:
            step += 1
            print(f"[{step}/{total}] entry={i} scoring_mode={scoring_mode}")

            game_result = run_single_game(prompt_text, response, scoring_mode, shared_model)

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

    print(
        f"Saved {len(results)} game results ({len(entries)}x{len(SCORING_MODES)}) to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
