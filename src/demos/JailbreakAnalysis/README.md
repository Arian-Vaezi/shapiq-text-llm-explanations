# Jailbreak Analysis: Shapley Interactions over Adversarial Prompts

Which prompt sentences — and which sentence *pairs* — make an LLM comply with a jailbreak?

A jailbreak prompt is rarely one magic string: it is a benign pretext, a harmful request, and a
"no warnings" instruction working *together*. This demo treats the prompt as a cooperative game —
its **sentences are the players** — and uses `shapiq` to attribute compliance to sentences
(Shapley values) and to sentence pairs (second-order **k-SII** interactions).

## Quick start

```bash
# Offline results explorer — no GPU, no API key; reads the committed summaries:
uv run streamlit run src/demos/JailbreakAnalysis/results_app.py

# Live app — type a prompt, pick a model/value function, explain it:
uv run streamlit run src/demos/JailbreakAnalysis/app.py
```

The results app has three pages: the experimental setup, a **Result Explorer** (per-config
response, judge verdict, Shapley/k-SII values, and the order-1 vs order-1+2 reconstruction),
and an **Explanation Explorer** (sentence plots, interaction heatmaps, network views).

## The game

- **Players** — the sentences of one adversarial prompt (3–6 per prompt; split on
  sentence-ending punctuation in `JailbreakAnalysisGame._build_players`).
- **Coalition** — the same prompt with only those sentences kept; absent sentences are
  *removed*, not mask-tokened (`coalition_to_prompt`, `mask_strategy="remove"`).
- **Budget** — `recommended_budget` caps at `2^n` for small games, so every coalition is
  evaluated and the k-SII values are **exact** — no sampling variance.

Two value functions turn the model into `v : 2^N → ℝ`:

| | Logprob contrast *(the sweep)* | LLM-as-a-judge *(7-prompt pilot)* |
|---|---|---|
| `v(S)` | `mean log P(comply) − mean log P(refuse)`, baseline-centered so `v(∅)=0` | generate a response, judge grades it 0–10 |
| Cost | one forward pass, no generation | a generation + a judging pass per coalition (~6×) |
| Character | deterministic, interaction-rich, but a **proxy** | faithful, but **near-binary** — interactions vanish |

That trade-off is a central finding: on the same prompt with identical coalitions, the logprob
gap is **+57.3 pp** where the judge's is **+1.0 pp**. A faithful value function has almost no
interactions to find; an interaction-rich one is only a proxy.

## The vulnerability scan

`run_vulnerability_scan.py` (repo root) runs the grid — **5 working models × 6 temperatures ×
15 adversarial prompts** — generating a response per config and labeling it with a **binary**
`gpt-oss-safeguard-20b` judge. Slurm scripts (`run_vulnerability_scan.sbatch` and friends) ran
this on an A100.

Two judge details that matter:

- gpt-oss-safeguard is a **Harmony reasoning model**: it writes an `analysis` channel before its
  `final` verdict. The judge therefore gets 512 tokens and only the `final` channel is parsed.
- An unparseable verdict returns **`-1` (unknown)**, never a silent `0`.

Headline numbers (committed in `results/summary_asr*.json`): 449 valid verdicts, **178
jailbroken (39.6%)**, rates from **77.8 %** (Mistral-7B-Instruct) down to **17.8 %**
(gemma-4-e4b). Model choice dominates; temperature is flat or *declining* for 4 of 5 models.

## The interaction sweep

`run_experiment.py` is the headless k-SII runner (order 2, sentence players, logprob value
function): 10 multi-sentence jailbroken prompts × 3 models. Each run records every evaluated
coalition and fits an order-1 and an order-1+2 least-squares reconstruction — the gap **ΔR²**
measures how much of the value function only appears once pairs are allowed in
(range **+2.8 to +59.8 pp**, mean **+24.4 pp** across the 30 runs).

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
| `JailbreakAnalysisGame.py` | the `shapiq.Game`: players, masking, both value functions |
| `app.py` | live Streamlit demo |
| `results_app.py` | offline results explorer (three pages) |
| `run_experiment.py` | headless k-SII runner + reconstruction diagnostic |
| `build_summary_asr.py` / `build_summary_interactions.py` | fold raw runs into the committed `results/*.json` |
| `interaction_prompts/` | the 10 multi-sentence prompts used in the sweep |
| `run_vulnerability_scan.py` + `*.sbatch` | the scan and its Slurm scripts (at the repo root) |
| `PROJECT_NOTES.md` | deeper design notes (segmentation, masking, budget, papers) |
