#!/bin/bash
# Submit with:  sbatch run_jailbreak_experiment.sbatch
# Monitor with: squeue -M hlai -u $USER   and   tail -f logs/jailbreak-<jobid>.out
#
# This runs the HEADLESS runner (src/demos/JailbreakAnalysis/run_experiment.py) on a GPU
# node. Edit the python flags at the bottom to change model / segmentation / order.
#
#SBATCH --clusters=hlai
#SBATCH --job-name=jailbreak-exp
#SBATCH --output=logs/jailbreak-%j.out
#SBATCH --error=logs/jailbreak-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=04:00:00

set -euo pipefail

export SLURM_CONF="/lrz/sys/share/CMUC4/etc/slurm_hlai.conf"
# ALL caches on SCRATCH (huge), never $HOME (small, shared quota): HF model cache,
# uv build cache, and tempfiles. Fall back to the LRZ scratch path if not inherited.
: "${SCRATCH:=/dss/lxclscratch/04/$USER}"
export HF_HOME="$SCRATCH/hf_cache"
export UV_CACHE_DIR="$SCRATCH/uv_cache"
export TMPDIR="$SCRATCH/tmp"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$TMPDIR"

# Run in the submit directory (works from your own clone on scratch), not a hardcoded path.
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs results
source .venv/bin/activate

echo "Job ID:  ${SLURM_JOB_ID:-<none>}"
echo "Node:    $(hostname)"
echo "Branch:  $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '<git unavailable on node>')"
nvidia-smi || true

# First-order Shapley on TinyLlama, sentence segmentation, logprob scoring (cheap, local).
# Output filename includes the job id so reruns don't overwrite each other.
uv run python src/demos/JailbreakAnalysis/run_experiment.py \
    --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
    --segmentation sentence \
    --mask-strategy remove \
    --scoring-mode logprob \
    --order 1 \
    --quality Auto \
    --example 0 \
    --output "results/jailbreak_${SLURM_JOB_ID}.json"

# --- Second-order k-SII (uncomment to also produce heatmap + network plots) ---
# uv run python src/demos/JailbreakAnalysis/run_experiment.py \
#     --model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
#     --segmentation sentence --mask-strategy remove --scoring-mode logprob \
#     --order 2 --quality Thorough \
#     --example 0 \
#     --output "results/jailbreak_${SLURM_JOB_ID}_o2.json"

echo "Experiment finished."
