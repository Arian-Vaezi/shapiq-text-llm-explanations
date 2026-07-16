#!/bin/bash
#SBATCH --job-name=ctrl50-lex
#SBATCH --partition=NvidiaAll
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

export TMPDIR="$PWD/tmp"
export UV_CACHE_DIR="$PWD/.uv_cache"
export HF_HOME="$PWD/.hf_cache"
export TRANSFORMERS_CACHE="$PWD/.hf_cache/transformers"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$HF_HOME"

RUN_ID="$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}_controlled_50_bge_lexical"
OUT_DIR="demos/rag_retrieval_explanation/evals/runs/${RUN_ID}"

if [ -e "$OUT_DIR" ]; then
  echo "Refusing to overwrite existing run dir: $OUT_DIR" >&2
  exit 1
fi

echo "out: $OUT_DIR"

uv run python -m demos.rag_retrieval_explanation.evals.experiments.run_controlled_eval \
  --method "Dense embeddings" \
  --embedding-model models/embedding/bge-base-en-v1.5 \
  --embedding-device cuda \
  --value-function lexical \
  --output-dir "$OUT_DIR"

echo "DONE"
cat "$OUT_DIR/report.md"
