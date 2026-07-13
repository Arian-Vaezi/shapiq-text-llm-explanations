import json
from pathlib import Path

folder = Path("vulnerability_rerun")

deleted = 0
kept = 0

for file in folder.glob("*.json"):
    try:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Delete if generation_error is NOT null
        if data.get("generation_error") is not None:
            file.unlink()
            deleted += 1
        else:
            kept += 1

    except Exception as e:
        print(f"Skipping {file}: {e}")

print(f"Deleted: {deleted}")
print(f"Kept: {kept}")