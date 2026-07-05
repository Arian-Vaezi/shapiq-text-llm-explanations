#!/bin/bash
# Second-order k-SII interaction analysis on the jailbroken multi-sentence prompts.
# Submit:  sbatch run_interaction_sweep.sbatch
# Monitor: squeue -M hlai -u $USER ; tail -f logs/jb-interactions-<jobid>.out
# Retrieve (laptop): scp -r <user>@cool.hpc.lrz.de:.../results/interactions_<jobid> ./
#
#SBATCH --clusters=hlai
#SBATCH --job-name=jb-interactions
#SBATCH --output=logs/jb-interactions-%j.out
#SBATCH --error=logs/jb-interactions-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00

export SLURM_CONF="/lrz/sys/share/CMUC4/etc/slurm_hlai.conf"
# All caches on SCRATCH, never $HOME (small, shared quota).
: "${SCRATCH:=/dss/lxclscratch/04/$USER}"
export HF_HOME="$SCRATCH/hf_cache"
export UV_CACHE_DIR="$SCRATCH/uv_cache"
export TMPDIR="$SCRATCH/tmp"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$TMPDIR"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
OUTDIR="results/interactions_${SLURM_JOB_ID:-manual}"
mkdir -p logs "$OUTDIR"
source .venv/bin/activate

echo "Job ${SLURM_JOB_ID:-<none>} on $(hostname) -> $OUTDIR"
nvidia-smi || true

set -uo pipefail

# Value function = logprob (comply vs refuse): deterministic and continuous, which
# gives clean k-SII interactions. Run on the strong local causal models that the
# vulnerability scan showed actually get jailbroken. (logprob needs a local causal
# model, so no API/Groq models here.)
MODELS=(
  "mistralai/Mistral-7B-Instruct-v0.3"   # highest jailbreak rate (78%)
  "Qwen/Qwen2.5-7B-Instruct"             # 43%
  "TinyLlama/TinyLlama-1.1B-Chat-v1.0"   # weak-model baseline (37%)
)
PROMPT_DIR="src/demos/JailbreakAnalysis/interaction_prompts"

run() {  # args: model prompt_file
  local model="$1" pf="$2"
  local pid; pid="$(basename "$pf" .txt)"
  local tag; tag="$(echo "$model" | tr '/' '_')_${pid}"
  echo ">>> $(date +%H:%M:%S)  $tag"
  uv run python src/demos/JailbreakAnalysis/run_experiment.py \
      --model "$model" --prompt-file "$pf" \
      --segmentation sentence --order 2 --scoring-mode logprob --quality Thorough \
      --output "${OUTDIR}/${tag}.json" \
    || echo "!!! FAILED: $tag"
}

for m in "${MODELS[@]}"; do
  for pf in "$PROMPT_DIR"/*.txt; do
    run "$m" "$pf"
  done
done

echo "Interaction sweep complete: $OUTDIR"
ls -1 "$OUTDIR"/*.json 2>/dev/null | wc -l
