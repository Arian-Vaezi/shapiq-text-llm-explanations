# Representative Agent + XAI results

This directory contains the authoritative results of a fixed 20-case Agent + XAI
comparison. The experiment evaluates tool-routing behavior and explains each model's
actual selected tool using Shapley values (SV) and pairwise k-SII interactions. The
cases were selected before XAI from the existing 40-prompt holdout for interpretability
and category coverage; this is not a new accuracy benchmark.

## Reproducibility

- Repository commit: `c5ac64c0f71cfe881d6e57cecd8942a98b965858`
- Device: Apple PyTorch MPS with `PYTORCH_ENABLE_MPS_FALLBACK=1`
- Dtype: `auto`
- Quantization: `none`
- Maximum new tokens: `512`
- Maximum pairs per batch: `1`
- First-order index: SV
- Pairwise index: k-SII, maximum order 2
- Exact-computation threshold: 10 linguistic players

The fixed case IDs, in execution order, were:

- Weather: `w01`, `w03`, `w04`, `w07`, `w10`
- Calculator: `c01`, `c05`, `c06`, `c07`, `c08`
- Web search: `s03`, `s04`, `s07`, `s08`, `s09`
- No tool: `n01`, `n03`, `n04`, `n08`, `n10`

The runs were invoked from `src/demos/agentic_tool_use_explanation` with:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 \
uv run python run_representative_xai_20.py \
  --model-name Qwen/Qwen2.5-3B-Instruct \
  --device mps \
  --dtype auto \
  --quantization none \
  --max-new-tokens 512 \
  --max-pairs-per-batch 1 \
  --output-dir outputs/representative_xai_20
```

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 \
uv run python run_representative_xai_20.py \
  --model-name Qwen/Qwen3-4B-Instruct-2507 \
  --device mps \
  --dtype auto \
  --quantization none \
  --max-new-tokens 512 \
  --max-pairs-per-batch 1 \
  --output-dir outputs/representative_xai_20_qwen3_4b
```

## Results overview

| Model | Routing accuracy | Completed | Failed | Exact SV | Approximate SV | Exact k-SII | Approximate k-SII |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Qwen/Qwen2.5-3B-Instruct` | 19/20 (95%) | 20 | 0 | 20 | 0 | 20 | 0 |
| `Qwen/Qwen3-4B-Instruct-2507` | 14/20 (70%) | 20 | 0 | 20 | 0 | 20 | 0 |

The exact/approximate counts above count completed cases for each index. The JSON
records the exact coalition-evaluation count for every individual explanation.

This is a cross-model comparison, not a pure scaling experiment. The models come from
different Qwen generations and instruction-tuned releases, so differences cannot be
attributed only to parameter count.

## Authoritative artifacts

- `qwen2_5_3b/representative_xai_results.json`: complete structured 3B run record.
- `qwen2_5_3b/representative_xai_summary.csv`: concise 3B case summary.
- `qwen3_4b/representative_xai_results.json`: complete structured 4B run record.
- `qwen3_4b/representative_xai_summary.csv`: concise 4B case summary.

Raw logs, smoke-test outputs, duplicate four-case outputs, and failed or partial runs
are intentionally excluded. They are not authoritative results and are unnecessary for
interpreting or reproducing this comparison.
