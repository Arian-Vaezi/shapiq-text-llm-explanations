#!/usr/bin/env bash
#SBATCH --job-name=qasper50-bge-ll
#SBATCH --partition=NvidiaAll
#SBATCH --cpus-per-task=14
#SBATCH --time=12:00:00
#SBATCH --output=logs/qasper50-bge-ll-%j.out
#SBATCH --error=logs/qasper50-bge-ll-%j.err

set -euo pipefail

cd "$HOME/projects/shapiq-text-llm-explanations"

JOB_ID="${SLURM_JOB_ID:-manual}"
mkdir -p logs .tmp .uv-cache .hf-cache

export TMPDIR="$PWD/.tmp/slurm-${JOB_ID}"
mkdir -p "$TMPDIR"
export UV_CACHE_DIR="$PWD/.uv-cache"
export HF_HOME="$PWD/.hf-cache"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-14}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

echo "host: $(hostname)"
date
nvidia-smi || true

test -f models/llm/gemma-4-E4B-it-Q4_K_M.gguf
test -d models/embedding/bge-base-en-v1.5
test -f demos/rag_retrieval_explanation/evals/cases/qasper_validation_50.json

uv run python - <<'PY'
import torch
import llama_cpp

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("llama_cpp:", llama_cpp.__version__)
PY

STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUT_DIR="demos/rag_retrieval_explanation/evals/runs/${STAMP}_${JOB_ID}_qasper_50_bge_contrastive_ll_e4b_cuda4"
if [[ -e "$OUT_DIR" ]]; then
  echo "Refusing to overwrite existing output directory: $OUT_DIR" >&2
  exit 2
fi

echo "out: $OUT_DIR"

uv run python -m demos.rag_retrieval_explanation.evals.experiments.run_qasper_eval \
  --output-dir "$OUT_DIR" \
  --model-path models/llm/gemma-4-E4B-it-Q4_K_M.gguf \
  --n-ctx 4096 \
  --n-gpu-layers 4 \
  --embedding-model models/embedding/bge-base-en-v1.5 \
  --embedding-device cuda
