"""experiments.py — Standalone SLURM experiment runner.

PURPOSE
-------
This script is NOT the Streamlit app. It runs OFFLINE as a SLURM batch job:
    - No UI, no interactivity, no st.spinner
    - Loads models once, runs through a fixed sentence dataset
    - Computes SV + k-SII, both APPROXIMATE (KernelSHAP/KernelSHAPIQ) and
      EXACT (brute-force via shapiq's ExactComputer) for every sentence
    - Saves results incrementally to disk after EVERY sentence (checkpointing)
    - Resume-safe: if the job is killed/times out, rerunning skips completed work

RESEARCH QUESTION
------------------
Do Shapley interactions reveal the same failure signature for difficult
sentiment phenomena (negation, sarcasm) across English, French, and German —
or does the model's interaction structure (and accuracy) degrade differently
per language?

DESIGN
------
8 base items (sentence + topic), each translated into EN/FR/DE = 24 sentences:

    phenomenon   topic     EN / FR / DE
    ─────────────────────────────────────
    negative     film      ✓
    negative     service   ✓
    positive     film      ✓
    positive     service   ✓
    negation     film      ✓
    negation     service   ✓
    sarcasm      film      ✓
    sarcasm      service   ✓

MODELS
------
    Encoder : cardiffnlp/twitter-xlm-roberta-base-sentiment
              3-class (negative/neutral/positive), EN/FR/DE covered.
              v(S) = P(positive|S) - P(negative|S)

    Decoder : Qwen/Qwen3.5-9B-Instruct
              Strong multilingual coverage (201 languages).
              v(S) = mean log P(T+|S) - mean log P(T-|S)

USAGE
-----
This script is split into two runs to fit the 8-hour SLURM window:

    python experiments.py --half 1   # sentences 1-12
    python experiments.py --half 2   # sentences 13-24

Or run everything in one go (only if your SLURM allocation allows it):

    python experiments.py --half all

OUTPUT
------
    results/
        raw/
            {sentence_id}_{model_type}.pkl   <- full InteractionValues objects
        summary.json                          <- JSON-safe summary, all runs
        progress.log                          <- human-readable progress log
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# ── Path setup ─────────────────────────────────────────────────────────────────
# Ensure shapiq and the demo modules are importable regardless of cwd.
THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR.parents[1]  # src/
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(THIS_DIR))

# Redirect HF cache if HF_HOME is not already set (adjust path for your CIP setup)
os.environ.setdefault("HF_HOME", str(Path.home() / "hf_cache"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import shapiq  # noqa: E402

# ── Output paths ──────────────────────────────────────────────────────────────
RESULTS_DIR = THIS_DIR / "results"
RAW_DIR = RESULTS_DIR / "raw"
SUMMARY_PATH = RESULTS_DIR / "summary.json"
LOG_PATH = RESULTS_DIR / "progress.log"

RESULTS_DIR.mkdir(exist_ok=True)
RAW_DIR.mkdir(exist_ok=True)

# ── Model configuration ──────────────────────────────────────────────────────
ENCODER_MODEL_ID = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
ENCODER_POSITIVE_LABEL = "positive"
ENCODER_NEGATIVE_LABEL = "negative"

DECODER_MODEL_ID = "Qwen/Qwen3.5-4B"
DECODER_REGISTER = "formal"  # "formal" or "informal" — set once for the whole run

RANDOM_STATE = 42
BUDGET_ENCODER = 512
BUDGET_DECODER = 512

# Exact computation is only safe for n_players below this threshold.
# 2^EXACT_MAX_PLAYERS coalitions are evaluated directly — no sampling.
# At n=13 (longest sarcasm sentence) this is 8192 calls; still feasible
# for the encoder but expensive for the decoder. We compute exact values
# for ALL sentences on the encoder, and only for n<=10 on the decoder,
# flagging longer ones as "exact_skipped" in the results.
EXACT_MAX_PLAYERS_DECODER = 20


# ── Sentence dataset ──────────────────────────────────────────────────────────
# 8 items x 3 languages = 24 sentences.
# Each item shares an id across languages so cross-language comparison is easy.

SENTENCES: list[dict] = [
    # ── Negative ──────────────────────────────────────────────────────────────
    {"id": "neg_film", "phenomenon": "negative", "topic": "film", "lang": "en",
     "text": "This film was absolutely terrible."},
    {"id": "neg_film", "phenomenon": "negative", "topic": "film", "lang": "fr",
     "text": "Ce film était absolument terrible."},
    {"id": "neg_film", "phenomenon": "negative", "topic": "film", "lang": "de",
     "text": "Dieser Film war absolut furchtbar."},

    {"id": "neg_service", "phenomenon": "negative", "topic": "service", "lang": "en",
     "text": "The service at this restaurant was really bad."},
    {"id": "neg_service", "phenomenon": "negative", "topic": "service", "lang": "fr",
     "text": "Le service dans ce restaurant était vraiment mauvais."},
    {"id": "neg_service", "phenomenon": "negative", "topic": "service", "lang": "de",
     "text": "Der Service in diesem Restaurant war wirklich schlecht."},

    # ── Positive ──────────────────────────────────────────────────────────────
    {"id": "pos_film", "phenomenon": "positive", "topic": "film", "lang": "en",
     "text": "This film was absolutely wonderful."},
    {"id": "pos_film", "phenomenon": "positive", "topic": "film", "lang": "fr",
     "text": "Ce film était absolument merveilleux."},
    {"id": "pos_film", "phenomenon": "positive", "topic": "film", "lang": "de",
     "text": "Dieser Film war absolut wunderbar."},

    {"id": "pos_service", "phenomenon": "positive", "topic": "service", "lang": "en",
     "text": "The service at this restaurant was truly amazing."},
    {"id": "pos_service", "phenomenon": "positive", "topic": "service", "lang": "fr",
     "text": "Le service dans ce restaurant était vraiment incroyable."},
    {"id": "pos_service", "phenomenon": "positive", "topic": "service", "lang": "de",
     "text": "Der Service in diesem Restaurant war wirklich erstaunlich."},

    # ── Negation ──────────────────────────────────────────────────────────────
    {"id": "neg8_film", "phenomenon": "negation", "topic": "film", "lang": "en",
     "text": "This film is not bad at all."},
    {"id": "neg8_film", "phenomenon": "negation", "topic": "film", "lang": "fr",
     "text": "Ce film n'est pas mauvais du tout."},
    {"id": "neg8_film", "phenomenon": "negation", "topic": "film", "lang": "de",
     "text": "Dieser Film ist überhaupt nicht schlecht."},

    {"id": "neg8_service", "phenomenon": "negation", "topic": "service", "lang": "en",
     "text": "The service at this restaurant is not terrible at all."},
    {"id": "neg8_service", "phenomenon": "negation", "topic": "service", "lang": "fr",
     "text": "Le service dans ce restaurant n'est pas terrible du tout."},
    {"id": "neg8_service", "phenomenon": "negation", "topic": "service", "lang": "de",
     "text": "Der Service in diesem Restaurant ist überhaupt nicht furchtbar."},

    # ── Sarcasm ───────────────────────────────────────────────────────────────
    {"id": "sar_film", "phenomenon": "sarcasm", "topic": "film", "lang": "en",
     "text": "Oh great, another boring film, exactly what I was hoping for."},
    {"id": "sar_film", "phenomenon": "sarcasm", "topic": "film", "lang": "fr",
     "text": "Oh super, encore un film ennuyeux, exactement ce que j'espérais."},
    {"id": "sar_film", "phenomenon": "sarcasm", "topic": "film", "lang": "de",
     "text": "Oh toll, noch ein langweiliger Film, genau das, worauf ich gehofft hatte."},

    {"id": "sar_service", "phenomenon": "sarcasm", "topic": "service", "lang": "en",
     "text": "Oh wonderful, another slow service, just what I needed."},
    {"id": "sar_service", "phenomenon": "sarcasm", "topic": "service", "lang": "fr",
     "text": "Oh merveilleux, encore un service lent, juste ce qu'il me fallait."},
    {"id": "sar_service", "phenomenon": "sarcasm", "topic": "service", "lang": "de",
     "text": "Oh wunderbar, noch ein langsamer Service, genau das, was ich brauchte."},
]

assert len(SENTENCES) == 24, f"Expected 24 sentences, got {len(SENTENCES)}"

# Stable run id per sentence — used for filenames and resume checks.
for s in SENTENCES:
    s["run_id"] = f"{s['id']}_{s['lang']}"


# ── Logging ───────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    """Print and append a timestamped message to the progress log."""
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a") as f:
        f.write(line + "\n")


# ── Model loading ─────────────────────────────────────────────────────────────

def load_encoder():
    """Load the multilingual encoder model + tokenizer once.

    Returns:
        Tuple of (model, tokenizer) on the correct device.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Loading encoder {ENCODER_MODEL_ID} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(ENCODER_MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(ENCODER_MODEL_ID)
    model.to(device)
    model.eval()

    log("Encoder ready.")
    return model, tokenizer


def load_decoder():
    """Load the decoder model wrapper once.

    Uses 4-bit quantization — Qwen/Qwen3.5-9B needs ~18GB in fp16,
    too large for an 8GB GPU. 4-bit quantization reduces this to ~5-6GB.

    Returns:
        HFModelWrapper instance.
    """
    from HFModelWrapper import HFModelWrapper

    log(f"Loading decoder {DECODER_MODEL_ID} (4-bit quantized)...")
    wrapper = HFModelWrapper(model_name=DECODER_MODEL_ID, load_in_4bit=True)
    log("Decoder ready.")
    return wrapper


# ── Game construction ─────────────────────────────────────────────────────────

def build_encoder_game(text: str, model, tokenizer):
    """Build an EncoderTextImputer for the given text using preloaded weights."""
    from EncoderTextImputer import EncoderTextImputer

    return EncoderTextImputer(
        model_name=ENCODER_MODEL_ID,
        input_text=text,
        preloaded_model=model,
        preloaded_tokenizer=tokenizer,
        positive_label=ENCODER_POSITIVE_LABEL,
        negative_label=ENCODER_NEGATIVE_LABEL,
    )


def build_decoder_game(text: str, wrapper, language: str = "en", register: str = "formal"):
    """Build a SentimentDecoderGame for the given text using a preloaded wrapper.

    Args:
        text: The sentence to explain.
        wrapper: Preloaded HFModelWrapper.
        language: Template language ("en", "fr", "de") — MUST match the
            sentence's actual language to avoid a code-switching confound.
        register: Template phrasing style ("formal" or "informal").

    Returns:
        SentimentDecoderGame instance.
    """
    from SentimentDecoderGame import SentimentDecoderGame

    return SentimentDecoderGame(
        model_name=DECODER_MODEL_ID,
        input_text=text,
        hf_model=wrapper,
        language=language,
        register=register,
    )


# ── Core computation ──────────────────────────────────────────────────────────

def compute_all(game, budget: int, *, compute_exact: bool) -> dict:
    """Run approximate and (optionally) exact SV + k-SII for one game.

    Args:
        game: A shapiq Game instance (EncoderTextImputer or SentimentDecoderGame).
        budget: Sample budget for the approximate computation.
        compute_exact: Whether to also compute exact_values() ground truth.

    Returns:
        dict with keys:
            sv_approx, sii_approx   — InteractionValues
            sv_exact, sii_exact     — InteractionValues or None
            exact_skipped           — bool, True if exact was skipped (n too large)
            timing                  — dict of stage -> seconds
    """
    timing: dict[str, float] = {}

    t0 = time.time()
    sv_approx = shapiq.KernelSHAP(
        n=game.n_players, random_state=RANDOM_STATE
    ).approximate(budget=budget, game=game)
    timing["sv_approx"] = time.time() - t0

    t0 = time.time()
    sii_approx = shapiq.KernelSHAPIQ(
        n=game.n_players, index="k-SII", max_order=2, random_state=RANDOM_STATE
    ).approximate(budget=budget, game=game)
    timing["sii_approx"] = time.time() - t0

    sv_exact = None
    sii_exact = None
    exact_skipped = False

    if compute_exact:
        t0 = time.time()
        sv_exact = game.exact_values(index="SV", order=1)
        timing["sv_exact"] = time.time() - t0

        t0 = time.time()
        sii_exact = game.exact_values(index="k-SII", order=2)
        timing["sii_exact"] = time.time() - t0
    else:
        exact_skipped = True

    return {
        "sv_approx": sv_approx,
        "sii_approx": sii_approx,
        "sv_exact": sv_exact,
        "sii_exact": sii_exact,
        "exact_skipped": exact_skipped,
        "timing": timing,
    }


def interaction_values_to_summary(
    iv: shapiq.InteractionValues | None,
    words: list[str],
) -> dict | None:
    """Convert an InteractionValues object into a JSON-safe summary dict.

    Args:
        iv: InteractionValues object, or None if not computed.
        words: List of player names for labeling.

    Returns:
        JSON-safe dict, or None.
    """
    if iv is None:
        return None

    order1 = {}
    order2 = {}
    for coalition, idx in iv.interaction_lookup.items():
        value = float(iv.values[idx])
        if len(coalition) == 1:
            order1[words[coalition[0]]] = value
        elif len(coalition) == 2:
            i, j = coalition
            order2[f"{words[i]}__{words[j]}"] = value

    return {"order1": order1, "order2": order2}


# ── Single sentence run ────────────────────────────────────────────────────────

def run_one(sentence: dict, model_type: str, models: dict) -> dict | None:
    """Run the full pipeline for one sentence on one model type.

    Args:
        sentence: One entry from SENTENCES.
        model_type: "encoder" or "decoder".
        models: dict with preloaded "encoder" -> (model, tokenizer) and
                "decoder" -> wrapper.

    Returns:
        Summary dict (JSON-safe), or None if the run failed.
    """
    run_key = f"{sentence['run_id']}_{model_type}"
    raw_path = RAW_DIR / f"{run_key}.pkl"

    # ── Resume check ──────────────────────────────────────────────────────────
    if raw_path.exists():
        log(f"SKIP (already done): {run_key}")
        with raw_path.open("rb") as f:
            cached = pickle.load(f)
        return cached["summary"]

    log(f"RUN: {run_key}  |  text='{sentence['text']}'")
    t_start = time.time()

    try:
        if model_type == "encoder":
            model, tokenizer = models["encoder"]
            game = build_encoder_game(sentence["text"], model, tokenizer)
            budget = BUDGET_ENCODER
            compute_exact = True  # always cheap enough for the encoder
        else:
            wrapper = models["decoder"]
            game = build_decoder_game(
                sentence["text"], wrapper,
                language=sentence["lang"], register=DECODER_REGISTER,
            )
            budget = BUDGET_DECODER
            compute_exact = game.n_players <= EXACT_MAX_PLAYERS_DECODER

        words = game.players.tolist() if hasattr(game.players, "tolist") else list(game.players)
        n_players = game.n_players

        # Raw prediction on full sentence
        grand = np.ones((1, n_players), dtype=bool)
        v_full = float(game(grand)[0]) + game.normalization_value
        label = "POSITIVE" if v_full >= 0 else "NEGATIVE"

        result = compute_all(game, budget=budget, compute_exact=compute_exact)

        summary = {
            "run_id": sentence["run_id"],
            "run_key": run_key,
            "phenomenon": sentence["phenomenon"],
            "topic": sentence["topic"],
            "lang": sentence["lang"],
            "text": sentence["text"],
            "model_type": model_type,
            "model_id": ENCODER_MODEL_ID if model_type == "encoder" else DECODER_MODEL_ID,
            "register": DECODER_REGISTER if model_type == "decoder" else None,
            "words": words,
            "n_players": n_players,
            "label": label,
            "v_full": v_full,
            "v_empty": game.normalization_value,
            "budget": budget,
            "exact_skipped": result["exact_skipped"],
            "timing": result["timing"],
            "total_runtime_seconds": time.time() - t_start,
            "sv_approx": interaction_values_to_summary(result["sv_approx"], words),
            "sii_approx": interaction_values_to_summary(result["sii_approx"], words),
            "sv_exact": interaction_values_to_summary(result["sv_exact"], words),
            "sii_exact": interaction_values_to_summary(result["sii_exact"], words),
        }

        # ── Save raw objects (pkl) — full InteractionValues preserved ──────────
        with raw_path.open("wb") as f:
            pickle.dump({"summary": summary, "raw": result}, f)

        log(f"DONE: {run_key}  |  {time.time() - t_start:.1f}s  |  label={label}")
        return summary

    except Exception as e:  # noqa: BLE001
        log(f"FAILED: {run_key}  |  {type(e).__name__}: {e}")
        log(traceback.format_exc())
        return None


# ── Summary persistence ────────────────────────────────────────────────────────

def load_summary() -> list[dict]:
    """Load existing summary.json, or return an empty list."""
    if SUMMARY_PATH.exists():
        with SUMMARY_PATH.open("r") as f:
            return json.load(f)
    return []


def save_summary(all_results: list[dict]) -> None:
    """Write the full summary list to summary.json (overwrite)."""
    with SUMMARY_PATH.open("w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


def append_to_summary(entry: dict) -> None:
    """Append one result to summary.json, replacing any existing entry
    with the same run_key (idempotent on resume)."""
    all_results = load_summary()
    all_results = [r for r in all_results if r.get("run_key") != entry["run_key"]]
    all_results.append(entry)
    save_summary(all_results)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run shapiq sentiment experiments offline.")
    parser.add_argument(
        "--half",
        choices=["1", "2", "all"],
        default="all",
        help="Which half of the 24 sentences to run: '1' (first 12), '2' (last 12), 'all'.",
    )
    parser.add_argument(
        "--model",
        choices=["encoder", "decoder", "both"],
        default="both",
        help="Which model type(s) to run.",
    )
    args = parser.parse_args()

    if args.half == "1":
        sentences = SENTENCES[:12]
    elif args.half == "2":
        sentences = SENTENCES[12:]
    else:
        sentences = SENTENCES

    model_types = ["encoder", "decoder"] if args.model == "both" else [args.model]

    log("=" * 70)
    log(f"Starting experiment run: half={args.half}  model={args.model}")
    log(f"Sentences to process: {len(sentences)}  x  models: {model_types}")
    log("=" * 70)

    # ── Load only the models we need ──────────────────────────────────────────
    models: dict = {}
    if "encoder" in model_types:
        models["encoder"] = load_encoder()
    if "decoder" in model_types:
        models["decoder"] = load_decoder()

    # ── Main loop ──────────────────────────────────────────────────────────────
    total_runs = len(sentences) * len(model_types)
    done = 0

    for sentence in sentences:
        for model_type in model_types:
            done += 1
            log(f"--- Progress: {done}/{total_runs} ---")
            summary = run_one(sentence, model_type, models)
            if summary is not None:
                append_to_summary(summary)

    log("=" * 70)
    log(f"Finished. {done}/{total_runs} runs attempted.")
    log(f"Results saved to: {SUMMARY_PATH}")
    log("=" * 70)


if __name__ == "__main__":
    main()