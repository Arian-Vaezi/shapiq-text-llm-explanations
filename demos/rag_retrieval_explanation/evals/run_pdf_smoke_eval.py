"""Backward-compatible shim — delegates to evals/runners/run_pdf_smoke_eval.py."""

from __future__ import annotations

import sys
from pathlib import Path

_DEMO_DIR = Path(__file__).resolve().parents[1]
if str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

from evals.runners.run_pdf_smoke_eval import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
