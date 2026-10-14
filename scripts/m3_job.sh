#!/bin/bash
# Run commands from this repo as a Slurm job on Monash's M3 (see M3.md).
#
# Submit from the repo root on M3. Everything after the script name (and its
# own options) is the command, run under `uv run --no-sync` so that a job
# never downloads (the venv is synced on purpose, in a CPU job; see M3.md).
# `sbatch` flags placed before the script override the defaults below:
#
#     sbatch scripts/m3_job.sh python -m fit2082.boost.experiment \
#         --dataset Pedestrian --seeds 1 --compile --variants baseline
#     sbatch --time=04:00:00 --mem=64G -J lendb scripts/m3_job.sh \
#         python scripts/stream_full.py --dataset LenDB --model hashboost --epochs 5
#     sbatch --gres=gpu:A100:1 scripts/m3_job.sh python scripts/ucr_benchmark.py --compile
#
# Options, before the command:
#
#   --stage A,B      copy data/A and data/B to the job's node-local /tmp first
#                    and export its path as $DATA. Reading LenDB's series off
#                    Lustre left the GPU waiting 36 s of a 79 s stream. The copy
#                    stays in the page cache, so it counts against --mem. Write
#                    '$DATA' in single quotes so it reaches the job unexpanded:
#                    ... --stage LenDB python scripts/stream_full.py --path '$DATA'
#   --parallel FILE  run each line of FILE (blank lines and # comments skipped)
#                    as its own process on the one GPU, with bash expanding
#                    $DATA. One LenDB stream used 2 GB of the L40S's 46 GB and
#                    kept it 25% busy, so several arms fill each other's stalls.
#                    Give each line its own --out: the processes write at once.
#                    Sharing slowed each run (QUANT 32 s alone, 99 s shared), so
#                    use it for accuracy, not for timings.
#   --serial FILE    run the lines one after another instead: one queue wait
#                    and one allocation, and each run's timings are its own.
#
# Every result file written in a job records it in its model entries (`context`,
# from `fit2082.results.run_context`): the job, host and GPU, `gpu_runs` (how
# many runs shared the GPU) and `shared_gpu`.
#
# The defaults name an L40S, which beat an A100 on every stage of the LenDB
# stream and waited a fifth as long to start (M3.md). A job prints its node,
# GPU, commit and command(s) first, and its GPU utilisation last. M3 cancels
# jobs that hold an idle GPU, and its own `jobstats` crashed on our jobs, so the
# log carries the evidence instead.
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

usage="usage: sbatch [sbatch flags] scripts/m3_job.sh [--stage A,B] (--parallel FILE | --serial FILE | <command> [args...])"
stage=""
mode=single
file=""
while [ $# -gt 0 ]; do
    case "$1" in
        --stage) stage=$2; shift 2 ;;
        --parallel) mode=parallel; file=$2; shift 2 ;;
        --serial) mode=serial; file=$2; shift 2 ;;
        *) break ;;
    esac
done
if [ "$mode" = single ] && [ $# -eq 0 ]; then
    echo "$usage" >&2
    exit 2
fi

# == environment ===============================================================

# uv's cache, venv and Python live on /scratch2 (M3.md, "Python environment").
source ~/.bashrc.d/fit2082.sh
# Compiled kernels persist across jobs here, instead of in the node-local /tmp
# that every new job starts without.
export TORCHINDUCTOR_CACHE_DIR=/scratch2/jt76/myan0101/torchinductor
cd "$SLURM_SUBMIT_DIR"
logs=/scratch2/jt76/myan0101/logs
base="$logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"

# == header ====================================================================

echo "job      $SLURM_JOB_ID ($SLURM_JOB_NAME) on $(hostname), $(date '+%F %T')"
echo "gpu      $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"
echo "commit   $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- || echo ' (dirty)')"

# == staging ===================================================================

export DATA=data
if [ -n "$stage" ]; then
    # Slurm gives each job its own /tmp and deletes it when the job ends; the
    # trap covers a run outside such a job.
    DATA="/tmp/fit2082-$SLURM_JOB_ID"
    trap 'rm -rf "$DATA"' EXIT
    mkdir -p "$DATA"
    t0=$SECONDS
    for name in ${stage//,/ }; do
        cp -rL "data/$name" "$DATA/"
    done
    echo "staged   $stage to $DATA ($(du -sh "$DATA" | cut -f1) in $((SECONDS - t0)) s)"
fi

# == run =======================================================================

# Sample GPU utilisation every 2 s for the summary below.
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits \
    -l 2 > "$base.gpu.csv" &
sampler=$!

start=$SECONDS
status=0
# one log per line, and the end of each copied into the job's own log
report() {
    echo
    echo "run $1    exit $2 after $((SECONDS - start)) s, ending:"
    tail -n 3 "$base.$1.out" | sed 's/^/         /'
}

if [ "$mode" = single ]; then
    export FIT2082_GPU_RUNS=1
    # a literal $DATA in the arguments becomes the staged path
    set -- "${@//\$DATA/$DATA}"
    echo "command  $*"
    echo
    uv run --no-sync "$@" || status=$?
else
    mapfile -t lines < <(grep -v -e '^\s*#' -e '^\s*$' "$file")
    if [ "$mode" = serial ]; then
        export FIT2082_GPU_RUNS=1
        echo "serial   ${#lines[@]} runs from $file"
        for i in "${!lines[@]}"; do
            echo "run $i    ${lines[$i]}  > $base.$i.out"
            run_status=0
            uv run --no-sync bash -c "${lines[$i]}" > "$base.$i.out" 2>&1 || run_status=$?
            report "$i" "$run_status"
            [ "$run_status" -eq 0 ] || status=$run_status
        done
    else
        export FIT2082_GPU_RUNS=${#lines[@]}
        # split the CPUs between the runs, or each torch would start a thread
        # per CPU and they would fight over them
        export OMP_NUM_THREADS=$(( SLURM_CPUS_PER_TASK / ${#lines[@]} > 0 ? SLURM_CPUS_PER_TASK / ${#lines[@]} : 1 ))
        echo "parallel ${#lines[@]} runs from $file, $OMP_NUM_THREADS threads each"
        pids=()
        for i in "${!lines[@]}"; do
            echo "run $i    ${lines[$i]}  > $base.$i.out"
            uv run --no-sync bash -c "${lines[$i]}" > "$base.$i.out" 2>&1 &
            pids+=($!)
        done
        for i in "${!pids[@]}"; do
            run_status=0
            wait "${pids[$i]}" || run_status=$?
            report "$i" "$run_status"
            [ "$run_status" -eq 0 ] || status=$run_status
        done
    fi
fi
kill "$sampler" 2>/dev/null || true

# == summary ===================================================================

echo
echo "exit     $status after $((SECONDS - start)) s"
awk -F', ' '{ u += $1; m = ($2 > m ? $2 : m); n++ }
    END { if (n) printf "gpu      mean %.0f%% busy over %d samples, peak %d MiB\n", u / n, n, m }' \
    "$base.gpu.csv"
exit "$status"
