#!/usr/bin/env bash
# Run N independent GA executions with consecutive seeds and collect the
# final binomial-IC performance of the best rule of each run.
#
# Usage: scripts/run_experiment.sh [-n runs] [-s first_seed] [-j jobs] [-b binary] [-i ics] [-o outdir]
#   -n runs        number of executions (default 100)
#   -s first_seed  seeds used are first_seed .. first_seed+runs-1 (default 1)
#   -j jobs        executions run concurrently (default 4)
#   -b binary      cuCga binary (default build/cuCga, built with CMake if missing)
#   -i ics         binomial ICs for the final evaluation (default 10000)
#   -o outdir      output directory (default results/<timestamp>)
#
# Produces <outdir>/results.csv (one row per run) and <outdir>/runs/seed_NNNN/
# holding each run's logs/output.log and stdout.
set -euo pipefail
export LC_ALL=C

RUNS=100
FIRST_SEED=1
JOBS=4
BIN=build/cuCga
ICS=10000
OUT=""

while getopts "n:s:j:b:i:o:h" opt; do
    case "$opt" in
        n) RUNS=$OPTARG ;;
        s) FIRST_SEED=$OPTARG ;;
        j) JOBS=$OPTARG ;;
        b) BIN=$OPTARG ;;
        i) ICS=$OPTARG ;;
        o) OUT=$OPTARG ;;
        *) sed -n '2,15p' "$0"; exit 1 ;;
    esac
done

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
[ -n "$OUT" ] || OUT="results/$(date +%Y%m%d-%H%M%S)"
if [ "$BIN" = "build/cuCga" ] && [ ! -x build/cuCga ]; then
    cmake -S . -B build && cmake --build build
fi
BIN=$(realpath "$BIN")
mkdir -p "$OUT/runs"
OUT=$(realpath "$OUT")

{
    echo "date: $(date -Is)"
    echo "git: $(git rev-parse HEAD 2>/dev/null || echo n/a)$(git diff --quiet 2>/dev/null || echo ' (dirty)')"
    echo "binary: $BIN ($(sha256sum "$BIN" | cut -c1-16))"
    echo "runs: $RUNS  seeds: $FIRST_SEED..$((FIRST_SEED+RUNS-1))  ics: $ICS  jobs: $JOBS"
    echo "gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo n/a)"
    echo "nvcc: $(nvcc --version 2>/dev/null | tail -1 || echo n/a)"
} > "$OUT/manifest.txt"

run_one() {
    local seed=$1 dir
    dir=$(printf "%s/runs/seed_%04d" "$OUT" "$seed")
    mkdir -p "$dir/logs"
    (cd "$dir" && "$BIN" -s "$seed" -n "$ICS" > stdout.txt 2> /dev/null)
}
export -f run_one
export OUT BIN ICS

start=$(date +%s)
seq "$FIRST_SEED" $((FIRST_SEED+RUNS-1)) | xargs -P "$JOBS" -I{} bash -c 'run_one {} && echo -n "." >&2'
echo >&2

echo "seed,train_best,rule,nics,perf,perf_strict" > "$OUT/results.csv"
for f in "$OUT"/runs/seed_*/stdout.txt; do
    # seed=1 train_best=98 rule=... nics=10000 perf=0.6512 perf_strict=0.6512
    sed -E 's/^seed=([0-9]+) train_best=([0-9]+) rule=([0-9a-f]+) nics=([0-9]+) perf=([0-9.]+) perf_strict=([0-9.]+)$/\1,\2,\3,\4,\5,\6/' "$f"
done | sort -t, -k1,1n >> "$OUT/results.csv"

n=$(($(wc -l < "$OUT/results.csv") - 1))
echo "elapsed: $(( $(date +%s) - start ))s" >> "$OUT/manifest.txt"
echo "$n/$RUNS runs collected in $OUT/results.csv" >&2
[ "$n" -eq "$RUNS" ]
