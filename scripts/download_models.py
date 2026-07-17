"""Download GGUF model files for the RAG Shapley demo.

Usage
-----
List all registered models and their local availability::

    uv run python scripts/download_models.py --list

Download a single model::

    uv run python scripts/download_models.py --model qwen3-8b

Download several models::

    uv run python scripts/download_models.py --models qwen3-1.5b qwen3-4b qwen3-8b

Download an entire family::

    uv run python scripts/download_models.py --family qwen3
    uv run python scripts/download_models.py --family gemma4

Download everything (large — check sizes with --list first)::

    uv run python scripts/download_models.py --all

Force re-download of already-present files::

    uv run python scripts/download_models.py --model gemma4-e4b --force

Models are downloaded to models/llm/ at the project root.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: make sure the backend package is importable without an editable
# install — we only need model_registry so no heavy deps are required.
# ---------------------------------------------------------------------------
_DEMO_DIR = Path(__file__).resolve().parents[1] / "demos" / "rag_retrieval_explanation"
sys.path.insert(0, str(_DEMO_DIR))

from backend.model_registry import (  # noqa: E402
    MODEL_REGISTRY,
    MODELS_DIR,
    list_models,
    list_models_by_family,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _format_size(path: Path) -> str:
    """Return a human-readable file size string."""
    if not path.exists():
        return ""
    size = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024  # type: ignore[assignment]
    return f"{size:.1f} TB"


def _print_model_table(models: list[dict]) -> None:
    """Print a table of model IDs, availability, and sizes."""
    col_id = max(len(m["id"]) for m in models) + 2
    col_name = max(len(m["display_name"]) for m in models) + 2
    header = f"{'ID':<{col_id}}{'Name':<{col_name}}{'Family':<14}{'Available':<12}Size"
    print(header)
    print("─" * (len(header) + 8))
    for m in models:
        avail = "✓ yes" if m["available"] else "✗ missing"
        size = _format_size(Path(str(m["path"]))) if m["available"] else ""
        print(
            f"{m['id']:<{col_id}}{m['display_name']:<{col_name}}{m['family']:<14}{avail:<12}{size}"
        )


def _download_model(model_id: str, *, force: bool = False) -> bool:
    """Download one model. Returns True on success, False if skipped or failed."""
    if model_id not in MODEL_REGISTRY:
        print(f"  ERROR: '{model_id}' is not a registered model ID.", file=sys.stderr)
        print("  Run --list to see valid IDs.", file=sys.stderr)
        return False

    entry = MODEL_REGISTRY[model_id]
    dest = MODELS_DIR / entry["filename"]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if dest.is_file() and not force:
        size = _format_size(dest)
        print(f"  {model_id}: already present ({size}) — skipping. Use --force to re-download.")
        return True

    repo_id = entry.get("repo_id", "")
    filename = entry["filename"]

    print(f"  {model_id}: downloading {filename}")
    print(f"    repo : {repo_id}")
    print(f"    dest : {dest}")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "  ERROR: huggingface_hub is not installed. Run:\n"
            "    uv pip install huggingface_hub\n"
            "  or add it to the demo requirements.txt.",
            file=sys.stderr,
        )
        return False

    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(MODELS_DIR),
            force_download=force,
        )
    except Exception as exc:
        print(f"  ERROR downloading {model_id}: {exc}", file=sys.stderr)
        return False

    size = _format_size(dest)
    print(f"  {model_id}: done ({size})")
    return True


# ── CLI ───────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download GGUF model files for the RAG Shapley demo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--list",
        action="store_true",
        help="List all registered models and their local availability.",
    )
    group.add_argument(
        "--model",
        metavar="MODEL_ID",
        help="Download a single model by ID (e.g. qwen3-8b).",
    )
    group.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL_ID",
        help="Download one or more model IDs.",
    )
    group.add_argument(
        "--family",
        metavar="FAMILY",
        help="Download all models in a family (e.g. qwen3, gemma4).",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Download every registered model (large — check sizes with --list first).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download files that already exist locally.",
    )
    return p


def main() -> int:
    """Run the model registry download command-line interface."""
    parser = _build_parser()
    args = parser.parse_args()

    # ── --list ──────────────────────────────────────────────────────────────
    if args.list:
        models = list_models()
        print(f"\nRegistered models  (stored in: {MODELS_DIR})\n")
        _print_model_table(models)
        available = [m for m in models if m["available"]]
        missing = [m for m in models if not m["available"]]
        print(f"\n{len(available)} available, {len(missing)} missing")
        if missing:
            print("\nTo download missing models:")
            for m in missing:
                print(f"  uv run python scripts/download_models.py --model {m['id']}")
        return 0

    # ── collect target IDs ───────────────────────────────────────────────────
    target_ids: list[str] = []

    if args.model:
        target_ids = [args.model]
    elif args.models:
        target_ids = list(args.models)
    elif args.family:
        family_models = list_models_by_family(args.family)
        if not family_models:
            print(
                f"ERROR: no models found for family '{args.family}'. "
                f"Known families: {sorted({e['family'] for e in MODEL_REGISTRY.values()})}",
                file=sys.stderr,
            )
            return 1
        target_ids = [m["id"] for m in family_models]
    elif args.all:
        target_ids = list(MODEL_REGISTRY.keys())

    # ── download loop ────────────────────────────────────────────────────────
    print(f"\nTarget model(s): {', '.join(target_ids)}\n")
    results = {mid: _download_model(mid, force=args.force) for mid in target_ids}
    failed = [mid for mid, ok in results.items() if not ok]

    print()
    if failed:
        print(f"FAILED: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("All done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
