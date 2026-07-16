#!/bin/bash
#SBATCH --job-name=ctrl50-cu4
#SBATCH --partition=NvidiaAll
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
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

RUN_ID="$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}_controlled_50_bge_contrastive_ll_e4b_cuda4"
OUT_DIR="demos/rag_retrieval_explanation/evals/runs/${RUN_ID}"

if [ -e "$OUT_DIR" ]; then
  echo "Refusing to overwrite existing run dir: $OUT_DIR" >&2
  exit 1
fi

echo "host: $(hostname)"
echo "out: $OUT_DIR"
nvidia-smi || true

uv run python -m demos.rag_retrieval_explanation.evals.experiments.run_controlled_eval \
  --method "Dense embeddings" \
  --embedding-model models/embedding/bge-base-en-v1.5 \
  --embedding-device cuda \
  --value-function contrastive_likelihood \
  --model-path models/llm/gemma-4-E4B-it-Q4_K_M.gguf \
  --n-gpu-layers 4 \
  --output-dir "$OUT_DIR"

echo "DONE"
cat "$OUT_DIR/report.md"
