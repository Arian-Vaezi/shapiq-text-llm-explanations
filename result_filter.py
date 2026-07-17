from __future__ import annotations

import json
from pathlib import Path

folder = Path("vulnerability_rerun")

deleted = 0
kept = 0

for file in folder.glob("*.json"):
    try:
        with open(file, encoding="utf-8") as f:
            data = json.load(f)

        # Delete if generation_error is NOT null
        if data.get("generation_error") is not None:
            file.unlink()
            deleted += 1
        else:
            kept += 1

    except Exception as e:  # noqa: BLE001 - one malformed file must not sink the filter
        print(f"Skipping {file}: {e}")

print(f"Deleted: {deleted}")
print(f"Kept: {kept}")
