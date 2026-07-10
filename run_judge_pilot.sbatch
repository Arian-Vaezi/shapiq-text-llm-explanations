#!/bin/bash
# Pilot: interaction analysis with a JUDGE value function (gpt-oss-safeguard, graded 0-10).
# Submit:  sbatch run_judge_pilot.sbatch
# Monitor: squeue -M hlai -u $USER ; tail -f logs/jb-judge-pilot-<jobid>.out
# Retrieve (laptop): scp -r <user>@cool.hpc.lrz.de:.../results/judge_pilot ./
#
#SBATCH --clusters=hlai
#SBATCH --job-name=jb-judge-pilot
#SBATCH --output=logs/jb-judge-pilot-%j.out
#SBATCH --error=logs/jb-judge-pilot-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00

export SLURM_CONF="/lrz/sys/share/CMUC4/etc/slurm_hlai.conf"
# All caches on SCRATCH, never $HOME (small quota).
: "${SCRATCH:=/dss/lxclscratch/04/$USER}"
export HF_HOME="$SCRATCH/hf_cache"
export UV_CACHE_DIR="$SCRATCH/uv_cache"
export TMPDIR="$SCRATCH/tmp"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$TMPDIR"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs results
source .venv/bin/activate

# HF_TOKEN for gated models if ever needed (Mistral-7B and the judge are ungated).
if [ -f .env ]; then set -a; source .env; set +a; fi

echo "Job ${SLURM_JOB_ID:-<none>} on $(hostname)"
nvidia-smi || true

# Target (7B) + judge (20B) load together on one A100-80GB. Deterministic (greedy).
uv run python run_judge_interaction_pilot.py \
    --target-model "mistralai/Mistral-7B-Instruct-v0.3" \
    --judge-model "openai/gpt-oss-safeguard-20b" \
    --gen-tokens 256 \
    --output-dir results/judge_pilot

echo "Pilot finished."
