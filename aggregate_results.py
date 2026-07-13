"""Aggregate all per-config vuln_*.json result files into one summary file.

Usage:
  python aggregate_results.py --results-dir results/vulnerability --out allResults.json
  python aggregate_results.py --results-dir results/vulnerability --out allResults.csv --csv

Produces a flat list of records:
  {"config_idx": 0, "model": "...", "tier": "...", "temperature": 0.0,
   "prompt_id": "...", "class_id": 1, "class_name": "role-play",
   "domain": "malware", "jailbroken": 0/1, "generation_error": null/"...",
   "judge_verdict": 0/1/-1}

Also prints a quick console summary (ASR overall, per model, per class,
per temperature) so you get a first read without opening a notebook.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_all(results_dir: Path) -> list[dict]:
    records = []
    files = sorted(results_dir.glob("vuln_*.json"))
    if not files:
        print(f"[warn] No vuln_*.json files found in {results_dir}")
        return records

    n_errors = 0
    for f in files:
        try:
            with f.open(encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            print(f"[warn] Failed to parse {f.name}: {exc}")
            continue

        settings = data.get("settings", {})
        gen_error = data.get("generation_error")
        if gen_error:
            n_errors += 1

        records.append({
            "config_idx": data.get("config_idx"),
            "model": settings.get("model"),
            "tier": settings.get("tier"),
            "temperature": settings.get("temperature"),
            "prompt_id": settings.get("prompt_id"),
            "class_id": settings.get("class_id"),
            "class_name": settings.get("class_name"),
            "domain": settings.get("domain"),
            "jailbroken": int(data.get("jailbroken", False)),
            "judge_verdict": data.get("judge", {}).get("jailbroken", -1),
            "generation_error": gen_error,
            "response_len": len(data.get("response") or ""),
            "runtime_seconds": data.get("runtime_seconds"),
            "source_file": f.name,
        })

    print(f"Loaded {len(records)} / {len(files)} result files. "
          f"{n_errors} had generation errors.")
    return records


def print_summary(records: list[dict]) -> None:
    valid = [r for r in records if r["judge_verdict"] in (0, 1)]
    errored = [r for r in records if r["generation_error"]]

    print(f"\n{'='*60}")
    print(f"SUMMARY  ({len(records)} total configs, {len(valid)} valid, {len(errored)} errored)")
    print(f"{'='*60}")

    if valid:
        overall_asr = sum(r["jailbroken"] for r in valid) / len(valid)
        print(f"\nOverall ASR (valid configs only): {overall_asr:.1%}")

        def asr_by(key: str) -> dict:
            buckets = defaultdict(list)
            for r in valid:
                buckets[r[key]].append(r["jailbroken"])
            return {k: (sum(v) / len(v), len(v)) for k, v in buckets.items()}

        print("\nASR by model:")
        for model, (asr, n) in sorted(asr_by("model").items(), key=lambda x: -x[1][0]):
            print(f"  {asr:5.1%}  (n={n:3d})  {model}")

        print("\nASR by class:")
        for cls, (asr, n) in sorted(asr_by("class_name").items(), key=lambda x: -x[1][0]):
            print(f"  {asr:5.1%}  (n={n:3d})  {cls}")

        print("\nASR by temperature:")
        for temp, (asr, n) in sorted(asr_by("temperature").items()):
            print(f"  T={temp:<4}  {asr:5.1%}  (n={n:3d})")

        print("\nASR by domain:")
        for domain, (asr, n) in sorted(asr_by("domain").items(), key=lambda x: -x[1][0]):
            print(f"  {asr:5.1%}  (n={n:3d})  {domain}")

    if errored:
        print(f"\n{len(errored)} configs had generation errors. First 5:")
        for r in errored[:5]:
            print(f"  [{r['config_idx']}] {r['model']} T={r['temperature']} {r['prompt_id']}")
        err_models = defaultdict(int)
        for r in errored:
            err_models[r["model"]] += 1
        print("\nErrors by model:")
        for model, n in sorted(err_models.items(), key=lambda x: -x[1]):
            print(f"  {n:3d}  {model}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate vulnerability scan results.")
    parser.add_argument("--results-dir", type=Path, default=Path("jailbreakResults/vulnerability"))
    parser.add_argument("--out", type=Path, default=Path("jailbreakResults/allResults.json"))
    parser.add_argument("--csv", action="store_true", help="Write CSV instead of JSON.")
    args = parser.parse_args()

    records = load_all(args.results_dir)
    if not records:
        return 1

    print_summary(records)

    if args.csv or args.out.suffix == ".csv":
        out_path = args.out if args.out.suffix == ".csv" else args.out.with_suffix(".csv")
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        print(f"\nWritten: {out_path}")
    else:
        with args.out.open("w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"\nWritten: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
