"""
build_summary_interactions.py

Merge the second-order k-SII interaction runs into one summary file for the
offline results app, mirroring build_summary_asr.py.

Two value functions are folded into the same schema so the app can show them
side by side for the same (model, prompt_id):

    logprob     mean logP(comply) - mean logP(refuse), baseline-centered.
                Deterministic and interaction-rich, but only a *proxy* for
                compliance. 30 runs (10 prompts x 3 models).

    judge_0_10  gpt-oss-safeguard grading the response 0-10. Faithful to what
                we actually care about, but near-binary in practice, so the
                payload sentence absorbs almost all the value and interactions
                collapse. 7 runs (Mistral-7B only) -- a deliberate pilot.

The raw run directories live outside the repo (they are large and not
committed); only this summary is checked in, so the app runs offline after a
clone. Pass the directories explicitly to rebuild:

    python build_summary_interactions.py --logprob-dir ... --judge-dir ...

Input:
    <repo parent>/interactions_95428/*.json   (excluding *_interaction_values.json)
    <repo parent>/judge_pilot/*.json

Output:
    results/summary_interactions.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = THIS_DIR.parents[2]

# The raw interaction runs are not committed; they sit next to the repo.
DEFAULT_LOGPROB_DIR = PROJECT_ROOT.parent / "interactions_95428"

DEFAULT_JUDGE_DIR = PROJECT_ROOT.parent / "judge_pilot"

OUTPUT_DIR = THIS_DIR / "results"

OUTPUT_FILE = OUTPUT_DIR / "summary_interactions.json"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _prompt_id_from_filename(stem: str, model: str) -> str | None:
    """Recover the prompt_id, which the logprob runs do not store inside the JSON.

    The runner wrote f"{model.replace('/', '_')}_{prompt_id}.json", so strip that
    prefix back off. Returns None if the stem does not match, so a stray file is
    skipped loudly rather than being keyed under a wrong id.
    """
    prefix = model.replace("/", "_") + "_"

    if stem.startswith(prefix):
        return stem[len(prefix) :]

    return None


def _entry(
    model: str,
    prompt_id: str,
    value_function: str,
    score: float | None,
    score_label: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    """Project one raw run onto the shared schema the app reads."""
    reconstruction = data.get("reconstruction", {}) or {}

    order1 = reconstruction.get("order1_r2")
    order2 = reconstruction.get("order2_r2")

    # The headline number: how much of the value function's behaviour only
    # shows up once pairs are allowed in. None if a run lacks reconstruction.
    delta_r2 = None

    if order1 is not None and order2 is not None:
        delta_r2 = order2 - order1

    return {
        "model": model,
        "prompt_id": prompt_id,
        "value_function": value_function,
        # Score of the FULL prompt under this value function. Deliberately not
        # called "jailbroken": the logprob score and the judge's binary label
        # disagree on 43% of runs, so these must not be conflated.
        "score": score,
        "score_label": score_label,
        "n_players": data.get("n_players"),
        "players": data.get("players", []),
        "player_values": data.get("player_values", []),
        "top_interaction_pairs": data.get("top_interaction_pairs", []),
        "reconstruction": {
            "order1_r2": order1,
            "order2_r2": order2,
            "delta_r2": delta_r2,
            "n_unique_coalitions": reconstruction.get("n_unique_coalitions"),
        },
        "budget": data.get("budget", (data.get("settings", {}) or {}).get("budget")),
        "runtime_seconds": data.get("runtime_seconds"),
    }


# ----------------------------------------------------------------------
# Readers
# ----------------------------------------------------------------------


def read_logprob_runs(input_dir: Path) -> list[dict[str, Any]]:
    """Read the logprob k-SII runs (settings.scoring_mode == 'logprob')."""
    entries = []

    if not input_dir.is_dir():
        print(f"Skipping logprob runs: {input_dir} not found")
        return entries

    # *_interaction_values.json are shapiq's raw dumps, not run summaries.
    files = sorted(
        f for f in input_dir.glob("*.json") if not f.name.endswith("_interaction_values.json")
    )

    print(f"Found {len(files)} logprob runs in {input_dir}")

    for file in files:
        try:
            with file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            settings = data.get("settings", {}) or {}

            model = settings.get("model")

            if not model:
                print(f"Skipping {file.name}: no settings.model")
                continue

            prompt_id = _prompt_id_from_filename(file.stem, model)

            if prompt_id is None:
                print(f"Skipping {file.name}: filename does not match model {model}")
                continue

            entries.append(
                _entry(
                    model=model,
                    prompt_id=prompt_id,
                    value_function="logprob",
                    score=data.get("compliance_score"),
                    score_label="compliance_score",
                    data=data,
                )
            )

        except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
            # One malformed run should not sink the whole summary.
            print(f"Failed reading {file}: {e}")

    return entries


def read_judge_runs(input_dir: Path) -> list[dict[str, Any]]:
    """Read the judge-value-function pilot runs (value_function == 'judge_0_10')."""
    entries = []

    if not input_dir.is_dir():
        print(f"Skipping judge pilot: {input_dir} not found")
        return entries

    files = sorted(
        f for f in input_dir.glob("*.json") if not f.name.endswith("_interaction_values.json")
    )

    print(f"Found {len(files)} judge-pilot runs in {input_dir}")

    for file in files:
        try:
            with file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            # These runs store both fields explicitly, unlike the logprob ones.
            model = data.get("target_model")

            prompt_id = data.get("prompt_id")

            if not model or not prompt_id:
                print(f"Skipping {file.name}: missing target_model/prompt_id")
                continue

            entries.append(
                _entry(
                    model=model,
                    prompt_id=prompt_id,
                    value_function=data.get("value_function", "judge_0_10"),
                    score=data.get("judge_score_full_prompt"),
                    score_label="judge_score_full_prompt",
                    data=data,
                )
            )

        except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
            # One malformed run should not sink the whole summary.
            print(f"Failed reading {file}: {e}")

    return entries


# ----------------------------------------------------------------------
# Build summary
# ----------------------------------------------------------------------


def build_summary(logprob_dir: Path, judge_dir: Path) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    summary = read_logprob_runs(logprob_dir) + read_judge_runs(judge_dir)

    summary.sort(key=lambda e: (e["model"], e["prompt_id"], e["value_function"]))

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    n_logprob = sum(1 for e in summary if e["value_function"] == "logprob")
    n_judge = len(summary) - n_logprob

    print(f"Saved {len(summary)} entries ({n_logprob} logprob, {n_judge} judge) to {OUTPUT_FILE}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--logprob-dir", type=Path, default=DEFAULT_LOGPROB_DIR)

    parser.add_argument("--judge-dir", type=Path, default=DEFAULT_JUDGE_DIR)

    args = parser.parse_args()

    build_summary(args.logprob_dir, args.judge_dir)


if __name__ == "__main__":
    main()
