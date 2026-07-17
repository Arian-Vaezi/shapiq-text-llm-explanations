# Jailbreak Analysis: Shapley Interactions over Adversarial Prompts

Which prompt sentences — and which sentence *pairs* — make an LLM comply with a jailbreak?

A jailbreak prompt is rarely one magic string: it is a benign pretext, a harmful request, and a
"no warnings" instruction working *together*. This demo treats the prompt as a cooperative game —
its **sentences are the players** — and uses `shapiq` to attribute compliance to sentences
(Shapley values) and to sentence pairs (second-order **k-SII** interactions).

## Quick start

Run these commands from the repository root. The offline explorer needs only the project
dependencies; the live app additionally needs a local model/GPU or configured API credentials.

```bash
uv sync

# Offline results explorer — no GPU, no API key; reads the committed summaries:
uv run streamlit run src/demos/JailbreakAnalysis/results_app.py

# Live app — type a prompt, pick a model/value function, explain it:
uv run streamlit run src/demos/JailbreakAnalysis/app.py
```

For the live app, put `GROQ_API_KEY` or `GEMINI_API_KEY` in a local `.env` for an API backend.
For gated Hugging Face models, export `HF_TOKEN` (or authenticate with Hugging Face). Never commit
credentials.

The results app has three pages: the experimental setup, a **Result Explorer** (per-config
response, judge verdict, Shapley/k-SII values, and the order-1 vs order-1+2 reconstruction),
and an **Explanation Explorer** (sentence plots, interaction heatmaps, network views).

## The game

- **Players (reported experiments)** — the sentences of one adversarial prompt (3–6 per prompt;
  split on sentence-ending punctuation in `JailbreakAnalysisGame._build_players`). The live app
  also supports word-, token-, and semantic-level segmentation.
- **Coalition** — the same prompt with only those sentences kept; absent sentences are
  *removed*, not mask-tokened (`coalition_to_prompt`, `mask_strategy="remove"`).
- **Budget** — `recommended_budget` caps at `2^n` for small games, so every coalition is
  evaluated and the k-SII values are **exact** — no sampling variance.

Two value-function variants appear in the committed results:

| | Logprob contrast *(30-run sweep)* | 0–10 LLM judge *(7-run pilot)* |
|---|---|---|
| `v(S)` | `mean log P(comply) − mean log P(refuse)`, baseline-centered so `v(∅)=0` | generate a response, then grade the response from 0 to 10 |
| Cost | several next-token scores; no response generation | target generation plus judge generation per coalition |
| Character | deterministic and interaction-rich, but a **verbalized-compliance proxy** | a more direct behavioral proxy, but near-binary in this pilot |

That trade-off is a central finding: on the same prompt with identical coalitions, the logprob
gap is **+57.3 pp** where the judge's is **+1.0 pp**. The result shows that the chosen value
function strongly affects the interaction structure; neither value function is ground truth.

The live app's current `llm-as-a-judge` path is not identical to the committed pilot. The live
`JailbreakAnalysisGame.py` implementation computes an expected score over the next-token digits
0–9 and normalizes it to `[0, 1]`; `run_judge_interaction_pilot.py` produced the committed 0–10
pilot artifacts.

## The vulnerability scan

`run_vulnerability_scan.py` (repo root) defines a **10 models × 6 temperatures × 15 adversarial
prompts** grid. The committed results cover the five successfully completed models: 450
configurations, each with a generated response scored by a **binary** `gpt-oss-safeguard-20b`
judge (449 returned a parseable verdict; one is recorded as `-1`, unknown). Slurm scripts
(`run_vulnerability_scan.sbatch` and friends) ran these experiments on an A100.

```bash
uv run python run_vulnerability_scan.py --list-grid
```

Direct GPU runs read `HF_TOKEN` from the process environment for gated models; the provided Slurm
script can source it from a local `.env` file.

Two judge details that matter:

- gpt-oss-safeguard is a **Harmony reasoning model**: it writes an `analysis` channel before its
  `final` verdict. The judge therefore gets 512 tokens and only the `final` channel is parsed.
- An unparseable verdict returns **`-1` (unknown)**, never a silent `0`.

Headline numbers (committed in `results/summary_asr*.json`): 449 parseable verdicts, **178
jailbroken (39.6 %)**, rates from **77.8 %** (Mistral-7B-Instruct) down to **17.8 %**
(`google/gemma-4-e4b-it`). Model choice dominates; temperature is flat or *declining* for 4 of 5
models.

![Jailbreak rate by model and temperature — the rows (models) differ by 4×, the columns
(temperature) are nearly flat](jailbreak_by_model_temperature.png)

## The interaction sweep

`run_experiment.py` is the headless k-SII runner (order 2, sentence players, logprob value
function): 10 multi-sentence jailbroken prompts × 3 models. Each run records every evaluated
coalition and fits an order-1 and an order-1+2 least-squares reconstruction — the gap **ΔR²**
measures how much of the value function only appears once pairs are allowed in
(range **+2.8 to +59.8 pp**, mean **+24.4 pp** across the 30 runs).

![Pairwise k-SII over prompt sentences for three runs: the same phishing prompt jailbreaks both
Mistral-7B and Qwen2.5-7B, but the same sentence pair flips from synergy (+4.3) on Mistral to
redundancy (−5.5) on Qwen; a third prompt is near-additive](assets/ksii_across_models.png)

*Same attack, same outcome, different structure: the identical prompt jailbreaks both models
(left, middle), yet the S2–S3 pair flips from **+4.3** on Mistral to **−5.5** on Qwen. On the
right, the same method reports a near-additive prompt (+3 pp) — it can also say "there is nothing
here". Red = synergy, blue = redundancy; the sign measures deviation from additivity, not
direction of the jailbreak. Per-run heatmap and network PNGs for all 30 runs are committed under
[`interactions_95428/`](interactions_95428/).*

## Reproducibility limitation

The result artifacts record model IDs and experiment settings, and `uv.lock` pins Python
dependencies. The current Hugging Face loaders do not pin model revisions, however, so an exact
rerun may drift if an upstream model repository changes.

## Reading the numbers — three caveats

1. **The k-SII sign is about additivity, not outcome direction.** A `+` (synergy) does *not*
   mean the pair pushes toward a jailbreak; across the sweep the pairs mostly counteract the
   main effects. Redundancy with a large ΔR² is not a contradiction — saturation *is* structure.
2. **`compliance_score` is not the jailbreak label.** It measures *verbalized* compliance and
   disagrees with the judge on ~43 % of runs (a prompt-injection response can comply fully while
   reciting refusal phrases). The judge verdict is the label.
3. At order 2, per-player values are **2-SII main effects** (`result[(i,)]`), not plain Shapley
   values.

## Files

| File | Purpose |
|---|---|
| `JailbreakAnalysisGame.py` | live-app game: segmentation, masking, logprob scoring, and normalized expected-digit judge scoring |
| `app.py` | live Streamlit demo |
| `results_app.py` | offline results explorer (three pages) |
| `run_experiment.py` | headless k-SII runner + reconstruction diagnostic |
| `build_summary_asr.py` / `build_summary_interactions.py` | fold raw runs into the committed `results/*.json` |
| `interaction_prompts/` | the 10 multi-sentence prompts used in the sweep |
| `run_judge_interaction_pilot.py` | the committed 0–10 judge-value-function pilot (at the repo root) |
| `run_vulnerability_scan.py` + `*.sbatch` | the scan and its Slurm scripts (at the repo root) |
