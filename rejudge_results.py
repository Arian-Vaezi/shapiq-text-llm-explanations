"""Re-judge existing vulnerability-scan results with the fixed judge.

Background
----------
The original judge ran with ``max_new_tokens=16``, which cut the Harmony/reasoning
judge (gpt-oss-safeguard) off mid-'analysis' before it ever emitted its 'final'
verdict. Every such judgment silently defaulted to 0 ("not jailbroken"), so ~54%
of results are false negatives whose verdict was never actually generated.

The saved JSONs still contain ``goal_en`` (the forbidden request) and ``response``
(the target model's output), so we can re-run *only the judge* on the stored
response — no need to regenerate target responses. This reuses the FIXED
``judge_response`` from ``run_vulnerability_scan`` (larger token budget + correct
final-channel parsing + -1 for unknown instead of silent 0).

Usage
-----
    # re-judge everything into a sibling folder (originals preserved)
    python rejudge_results.py --results-dir ../vulnerability_results

    # only re-do the suspect verdicts (0 / -1), overwriting in place
    python rejudge_results.py --results-dir ../vulnerability_results \
        --only-suspect --in-place

Run it where the judge model is available (a GPU node). The judge model loads once
and is reused across all files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_vulnerability_scan import DEFAULT_JUDGE_MODEL, judge_response


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", type=Path, required=True, help="Folder of vuln_*.json results.")
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--only-suspect",
        action="store_true",
        help="Only re-judge results whose current verdict is 0 or -1 (the broken ones). "
        "By default every result is re-judged for a consistent dataset.",
    )
    p.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the original files. Default writes to <results-dir>_rejudged/.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    src = args.results_dir
    files = sorted(src.glob("*.json"))
    if not files:
        raise SystemExit(f"No JSON files found in {src}")

    out_dir = src if args.in_place else src.with_name(src.name + "_rejudged")
    if not args.in_place:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[rejudge] {len(files)} files | judge={args.judge_model} | "
          f"{'in-place' if args.in_place else f'-> {out_dir}'} | "
          f"{'suspect-only' if args.only_suspect else 'all'}")

    changed = skipped = errors = 0
    flip_to_jb = 0  # how many flipped 0/unknown -> jailbroken
    for i, f in enumerate(files, 1):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip:unreadable] {f.name}: {exc}")
            errors += 1
            continue

        response = d.get("response")
        goal = d.get("goal_en")
        if d.get("generation_error") or not response or not goal:
            skipped += 1
            continue

        old = (d.get("judge") or {}).get("jailbroken")
        if args.only_suspect and old not in (0, -1, "0"):
            skipped += 1
            continue

        judge_result = judge_response(goal, response, args.judge_model, args.device)
        d["judge"] = judge_result
        d["jailbroken"] = judge_result["jailbroken"] == 1
        d["rejudged"] = True
        if old in (0, -1, "0") and judge_result["jailbroken"] == 1:
            flip_to_jb += 1

        (out_dir / f.name).write_text(
            json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        changed += 1
        if changed % 25 == 0:
            print(f"  ...{changed} re-judged ({i}/{len(files)})")

    print(f"[rejudge] done: rejudged={changed} skipped={skipped} unreadable={errors} "
          f"| flipped_to_jailbroken={flip_to_jb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
