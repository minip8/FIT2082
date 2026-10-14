#!/bin/bash
# Run one command from this repo as a Slurm job on Monash's M3 (see M3.md).
#
# Submit from the repo root on M3. Everything after the script name is the
# command, run under `uv run --no-sync` so that a job never downloads (the venv
# is synced on purpose, in a CPU job; see M3.md). `sbatch` flags placed before
# the script override the defaults below:
#
#     sbatch scripts/m3_job.sh python -m fit2082.boost.experiment \
#         --dataset Pedestrian --seeds 1 --compile --variants baseline
#     sbatch --time=04:00:00 --mem=64G -J lendb scripts/m3_job.sh \
#         python scripts/stream_full.py --dataset LenDB --model hashboost --epochs 5
#     sbatch --gres=gpu:A100:1 scripts/m3_job.sh python scripts/ucr_benchmark.py --compile
#
# The defaults name an L40S. A bare `--gres=gpu:1` landed on a T4 in the smoke
# test, which trained Pedestrian's baseline in 32.5 s against 8-14 s on the
# local RTX 4060. A job prints its node, GPU, commit and command first, and
# its GPU utilisation last. M3 cancels jobs that hold an idle GPU, and its own
# `jobstats` crashed on our jobs, so the log carries the evidence instead.
#
#SBATCH --job-name=fit2082
#SBATCH --partition=gpu
#SBATCH --gres=gpu:L40S:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch2/jt76/myan0101/logs/%x-%j.out

set -euo pipefail

if [ $# -eq 0 ]; then
    echo "usage: sbatch [sbatch flags] scripts/m3_job.sh <command> [args...]" >&2
    exit 2
fi

# == environment ===============================================================

# uv's cache, venv and Python live on /scratch2 (M3.md, "Python environment").
source ~/.bashrc.d/fit2082.sh
# Compiled kernels persist across jobs here, instead of in the node-local /tmp
# that every new job starts without.
export TORCHINDUCTOR_CACHE_DIR=/scratch2/jt76/myan0101/torchinductor
cd "$SLURM_SUBMIT_DIR"

# == header ====================================================================

echo "job      $SLURM_JOB_ID ($SLURM_JOB_NAME) on $(hostname), $(date '+%F %T')"
echo "gpu      $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
echo "commit   $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- || echo ' (dirty)')"
echo "command  $*"
echo

# == run =======================================================================

# Sample GPU utilisation every 10 s for the summary below.
gpu_log="/scratch2/jt76/myan0101/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.gpu.csv"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits \
    -l 10 > "$gpu_log" &
sampler=$!

start=$SECONDS
status=0
uv run --no-sync "$@" || status=$?
kill "$sampler" 2>/dev/null || true

# == summary ===================================================================

echo
echo "exit     $status after $((SECONDS - start)) s"
awk -F', ' '{ u += $1; m = ($2 > m ? $2 : m); n++ }
    END { if (n) printf "gpu      mean %.0f%% busy over %d samples, peak %d MiB\n", u / n, n, m }' \
    "$gpu_log"
exit "$status"
