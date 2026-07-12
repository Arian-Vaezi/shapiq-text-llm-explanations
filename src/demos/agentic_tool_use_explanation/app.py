"""Streamlit entrypoint for the agentic tool-use explanation demo."""

from __future__ import annotations

import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from _app_impl import *  # noqa: E402, F403
from _app_impl import main  # noqa: E402

if __name__ == "__main__":
    main()
