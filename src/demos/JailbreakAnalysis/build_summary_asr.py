"""
build_summary_asr.py

Merge individual jailbreak experiment JSON files into one summary file.

Input:
    vulnerability_rerun/*.json

Output:
    results/summary_asr.json
"""


from pathlib import Path
import json


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = THIS_DIR.parents[2]

INPUT_DIR = PROJECT_ROOT / "vulnerability_rerun"

OUTPUT_DIR = THIS_DIR / "results"

OUTPUT_FILE = OUTPUT_DIR / "summary_asr.json"



# ----------------------------------------------------------------------
# Build summary
# ----------------------------------------------------------------------

def build_summary():

    OUTPUT_DIR.mkdir(exist_ok=True)


    summary = []


    files = sorted(
        INPUT_DIR.glob("*.json")
    )


    print(f"Found {len(files)} JSON files")


    for file in files:

        try:

            with file.open(
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)


            # experiment config lives under "settings" in the raw files,
            # but keep top-level fallbacks in case older files differ
            settings = data.get("settings", {})

            judge = data.get("judge", {}) or {}

            # keep only what the explorer needs
            entry = {

                "file": file.name,

                # experiment config
                "model": settings.get("model", data.get("model")),
                "temperature": settings.get("temperature", data.get("temperature")),
                "tier": settings.get("tier"),
                "device": settings.get("device"),
                "judge_model": settings.get("judge_model"),

                # jailbreak prompt
                "prompt_id": settings.get("prompt_id", data.get("prompt_id", data.get("id"))),

                "prompt_class": settings.get("class_name", data.get("class_name")),
                "class_id": settings.get("class_id", data.get("class_id")),
                "template": settings.get("template", data.get("template")),
                "source": settings.get("source", data.get("source")),
                "domain": settings.get("domain", data.get("domain")),

                "prompt_text": data.get(
                    "prompt_text",
                    data.get("goal_en", data.get("prompt"))
                ),

                # generation result
                "response": data.get(
                    "response",
                    ""
                ),

                "generation_error": data.get("generation_error"),

                # judge
                "jailbroken": data.get(
                    "jailbroken",
                    judge.get("jailbroken", False)
                ),

                "judge_raw": judge.get("judge_raw"),

                # misc metadata (handy for debugging/sorting)
                "timestamp_utc": data.get("timestamp_utc"),
                "runtime_seconds": data.get("runtime_seconds"),

            }


            summary.append(entry)


        except Exception as e:

            print(
                f"Failed reading {file}: {e}"
            )


    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False
        )


    print(
        f"Saved {len(summary)} entries to {OUTPUT_FILE}"
    )



if __name__ == "__main__":

    build_summary()