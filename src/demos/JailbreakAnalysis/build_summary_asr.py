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


            # keep only what the explorer needs
            entry = {

                "file": file.name,

                # experiment config
                "model": data.get("model"),
                "temperature": data.get("temperature"),

                # jailbreak prompt
                "prompt_id": data.get(
                    "prompt_id",
                    data.get("id")
                ),

                "prompt_class": data.get(
                    "class_name"
                ),

                "prompt_text": data.get(
                    "prompt"
                ),

                # generation result
                "response": data.get(
                    "response",
                    ""
                ),

                # judge
                "jailbroken": data.get(
                    "jailbroken",
                    False
                ),

                "judge_raw": data.get(
                    "judge",
                    {}
                ).get(
                    "judge_raw"
                ),

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