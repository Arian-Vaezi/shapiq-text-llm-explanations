#!/bin/bash
#SBATCH -J shapiq_sentiment_half1
#SBATCH -p NvidiaAll
#SBATCH -t 08:00:00

# ── Logs inside project ──────────────────────────────────────────────────────
LOG_DIR="$HOME/shapiq-text-llm-explanations/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID:-noid}.log"
exec > >(tee -a "$LOG") 2>&1

set -Eeuo pipefail
trap 'st=$?; echo "ERR on line $LINENO: $BASH_COMMAND (exit $st)"; exit $st' ERR

echo "==== START $(date) node=$(hostname) job=${SLURM_JOB_ID:-noid} ===="

# ── Activate uv virtual environment ──────────────────────────────────────────
PROJECT_DIR="$HOME/shapiq-text-llm-explanations"
cd "$PROJECT_DIR"

set +u
source "$PROJECT_DIR/.venv/bin/activate"
set -u

python -V || true
which python || true

# ── GPU sanity check ──────────────────────────────────────────────────────────
nvidia-smi || true
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" || true

# ── Temp dir (optional but good) ─────────────────────────────────────────────
JOB_LOCAL_BASE="$HOME/tmp_jobs"
JOB_TMP="$JOB_LOCAL_BASE/shapiq_${SLURM_JOB_ID:-manual}"
mkdir -p "$JOB_TMP"
chmod 700 "$JOB_TMP"
export TMPDIR="$JOB_TMP"
echo "TMPDIR=$TMPDIR"

# ── HuggingFace cache (avoid filling home quota, adjust if needed) ───────────
export HF_HOME="$HOME/hf_cache"
mkdir -p "$HF_HOME"
echo "HF_HOME=$HF_HOME"

# ── Thread limits (avoid CPU oversubscription on shared nodes) ───────────────
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# ── Go to script location ────────────────────────────────────────────────────
SCRIPT_DIR="$PROJECT_DIR/src/demos/SentimentAnalysis"
SCRIPT_PATH="$SCRIPT_DIR/Experiments.py"
cd "$SCRIPT_DIR"

echo "Running: $SCRIPT_PATH --half 1 --model both"

# ── RUN EXPERIMENT — sentences 1-12, both encoder and decoder ───────────────
python "$SCRIPT_PATH" --half 1 --model both

echo "==== END $(date) ===="
