#!/bin/bash
# Batch sweep of the headless jailbreak runner across models x examples x segmentations.
# Submit:  sbatch run_jailbreak_sweep.sbatch
# Monitor: squeue -M hlai -u $USER ;  tail -f logs/jb-sweep-<jobid>.out
# Retrieve (from your LAPTOP): scp -r <user>@cool.hpc.lrz.de:~/projects/shapiq-text-llm-explanations/results/sweep_<jobid> ./
#
#SBATCH --clusters=hlai
#SBATCH --job-name=jb-sweep
#SBATCH --output=logs/jb-sweep-%j.out
#SBATCH --error=logs/jb-sweep-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00

export SLURM_CONF="/lrz/sys/share/CMUC4/etc/slurm_hlai.conf"
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

cd "$HOME/projects/shapiq-text-llm-explanations"
OUTDIR="results/sweep_${SLURM_JOB_ID:-manual}"
mkdir -p logs "$OUTDIR"
source .venv/bin/activate

echo "Job ${SLURM_JOB_ID:-<none>} on $(hostname) -> $OUTDIR"
nvidia-smi || true

# No `set -e`: one failed config must not abort the whole sweep.
set -uo pipefail

# Trimmed to the two small models so the HF cache (~5 GB) fits the home-directory
# disk quota. Re-enable Phi-3 (+7 GB) and Mistral-7B (+14 GB) only after pointing
# HF_HOME at a large scratch/work filesystem (set HF_HOME above), or home will fill up
# again ("Disk quota exceeded"). All are ungated and recognized by CausalModelWrapper.
MODELS=(
  "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  "Qwen/Qwen2.5-1.5B-Instruct"
  # "microsoft/Phi-3-mini-4k-instruct"      # +7 GB  -> needs scratch HF_HOME
  # "mistralai/Mistral-7B-Instruct-v0.3"    # +14 GB -> needs scratch HF_HOME
)
EXAMPLES=(0 1 2 3)

run() {  # args: model example segmentation order
  local model="$1" ex="$2" seg="$3" order="$4"
  local tag; tag="$(echo "$model" | tr '/' '_')_ex${ex}_${seg}_o${order}"
  echo ">>> $(date +%H:%M:%S)  $tag"
  uv run python src/demos/JailbreakAnalysis/run_experiment.py \
      --model "$model" --example "$ex" \
      --segmentation "$seg" --order "$order" \
      --scoring-mode logprob --quality Auto \
      --output "${OUTDIR}/${tag}.json" \
    || echo "!!! FAILED: $tag"
}

for m in "${MODELS[@]}"; do
  for ex in "${EXAMPLES[@]}"; do
    # sentence & semantic: both orders (few enough players for readable order-2 plots)
    run "$m" "$ex" sentence 1
    run "$m" "$ex" sentence 2
    run "$m" "$ex" semantic 1
    run "$m" "$ex" semantic 2
    # word: order-1 only — long JBB prompts give 60-80+ players, so order-2 is both
    # very expensive (quadratic in players) and unreadable as a heatmap. Add an
    # explicit `run "$m" "$ex" word 2` here if you specifically want it.
    run "$m" "$ex" word 1
  done
done

echo "Sweep complete: $OUTDIR"
ls -1 "$OUTDIR" | wc -l
