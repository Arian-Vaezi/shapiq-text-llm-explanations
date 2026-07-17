from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parents[1]  # src/
sys.path.insert(0, str(SRC_DIR))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

import shapiq
from shapiq.plot import sentence_interaction_heatmap, sentence_plot

SUMMARY_PATH = THIS_DIR / "results" / "summary.json"
FIGURES_DIR = THIS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


# ── Reconstruct InteractionValues from saved JSON dicts ───────────────────────


def rebuild_sv(order1: dict[str, float], words: list[str]) -> shapiq.InteractionValues:
    """Rebuild a first-order InteractionValues object from a saved order1 dict.

    Args:
        order1: {word: value} dict, as saved by interaction_values_to_summary().
        words: Ordered list of player names (must match the original run).

    Returns:
        shapiq.InteractionValues containing only singleton coalitions.
    """
    n = len(words)
    lookup: dict[tuple[int, ...], int] = {}
    values = []
    for i, w in enumerate(words):
        lookup[(i,)] = len(values)
        values.append(order1[w])

    return shapiq.InteractionValues(
        values=np.array(values, dtype=float),
        index="SV",
        max_order=1,
        min_order=1,
        n_players=n,
        interaction_lookup=lookup,
        baseline_value=0.0,
        estimated=False,
    )


def rebuild_sii(
    order1: dict[str, float],
    order2: dict[str, float],
    words: list[str],
) -> shapiq.InteractionValues:
    """Rebuild a k-SII InteractionValues object (order 1 + order 2) from saved dicts.

    Args:
        order1: {word: value} dict for singleton coalitions.
        order2: {"word1__word2": value} dict for pairwise coalitions.
        words: Ordered list of player names (must match the original run).

    Returns:
        shapiq.InteractionValues containing singleton + pairwise coalitions.
    """
    n = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}

    lookup: dict[tuple[int, ...], int] = {}
    values: list[float] = []

    # order 1
    for i, w in enumerate(words):
        lookup[(i,)] = len(values)
        values.append(order1[w])

    # order 2 — keys are "word_i__word_j" in original (i<j) order
    for i, j in combinations(range(n), 2):
        key = f"{words[i]}__{words[j]}"
        lookup[(i, j)] = len(values)
        values.append(order2[key])

    return shapiq.InteractionValues(
        values=np.array(values, dtype=float),
        index="k-SII",
        max_order=2,
        min_order=1,
        n_players=n,
        interaction_lookup=lookup,
        baseline_value=0.0,
        estimated=False,
    )


# ── Plotting ───────────────────────────────────────────────────────────────────


def make_sentence_plot(sv: shapiq.InteractionValues, words: list[str], out_path: Path) -> None:
    """Save the sentence attribution plot."""
    try:
        fig, _ = sentence_plot(sv, words, show=False)
        fig.patch.set_facecolor("white")
        fig.set_size_inches(10, 2.5)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {out_path.name}")
    except Exception as e:  # noqa: BLE001
        print(f"  ✗ sentence plot failed: {e}")


def make_network_plot(sii: shapiq.InteractionValues, words: list[str], out_path: Path) -> None:
    """Save the interaction network plot."""
    try:
        result = sii.plot_network(feature_names=words, show=False)
        if result is None:
            print("  ✗ network plot returned None")
            return
        fig = result[0] if isinstance(result, tuple) else result
        fig.patch.set_facecolor("white")
        fig.set_size_inches(7, 7)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {out_path.name}")
    except Exception as e:  # noqa: BLE001
        print(f"  ✗ network plot failed: {e}")


def make_heatmap_plot(sii: shapiq.InteractionValues, words: list[str], out_path: Path) -> None:
    """Save the interaction heatmap plot."""
    try:
        fig, _ = sentence_interaction_heatmap(sii, words, show=False)
        fig.patch.set_facecolor("white")
        fig.set_size_inches(7, 6)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ {out_path.name}")
    except Exception as e:  # noqa: BLE001
        print(f"  ✗ heatmap plot failed: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Build shapiq diagrams from saved results.")
    parser.add_argument(
        "--run_key", default=None, help="Plot only this run_key (e.g. sar_film_en_decoder)."
    )
    parser.add_argument(
        "--phenomenon", default=None, choices=["negative", "positive", "negation", "sarcasm"]
    )
    parser.add_argument("--lang", default=None, choices=["en", "fr", "de"])
    parser.add_argument("--model_type", default=None, choices=["encoder", "decoder"])
    parser.add_argument(
        "--index",
        default="approx",
        choices=["approx", "exact"],
        help="Use approximate (KernelSHAP/IQ) or exact values for plotting. Defaults to approx.",
    )
    args = parser.parse_args()

    with SUMMARY_PATH.open("r") as f:
        all_results = json.load(f)

    # Filter
    filtered = all_results
    if args.run_key:
        filtered = [r for r in filtered if r["run_key"] == args.run_key]
    if args.phenomenon:
        filtered = [r for r in filtered if r["phenomenon"] == args.phenomenon]
    if args.lang:
        filtered = [r for r in filtered if r["lang"] == args.lang]
    if args.model_type:
        filtered = [r for r in filtered if r["model_type"] == args.model_type]

    if not filtered:
        print("No matching results found. Check your filters.")
        return

    print(f"Building diagrams for {len(filtered)} run(s), index={args.index}...")

    for r in filtered:
        run_key = r["run_key"]
        words = r["words"]

        sv_dict = r["sv_exact"] if args.index == "exact" else r["sv_approx"]
        sii_dict = r["sii_exact"] if args.index == "exact" else r["sii_approx"]

        if sv_dict is None or sii_dict is None:
            print(
                f"[{run_key}] {args.index} values not available (exact_skipped={r.get('exact_skipped')}), skipping."
            )
            continue

        print(f"[{run_key}]  text='{r['text']}'  label={r['label']}")

        sv = rebuild_sv(sv_dict["order1"], words)
        sii = rebuild_sii(sii_dict["order1"], sii_dict["order2"], words)

        make_sentence_plot(sv, words, FIGURES_DIR / f"{run_key}_sentence.png")
        make_network_plot(sii, words, FIGURES_DIR / f"{run_key}_network.png")
        make_heatmap_plot(sii, words, FIGURES_DIR / f"{run_key}_heatmap.png")

    print(f"\nDone. Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
