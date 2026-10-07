#!/usr/bin/env bash
# Run N independent GA executions with consecutive seeds and collect the
# final binomial-IC performance of the best rule of each run.
#
# Usage: scripts/run_experiment.sh [-n runs] [-s first_seed] [-j jobs] [-b binary] [-i ics] [-o outdir] [-a args] [-x]
#   -n runs        number of executions (default 100)
#   -s first_seed  seeds used are first_seed .. first_seed+runs-1 (default 1)
#   -j jobs        executions run concurrently (default: CPU threads - 2, at
#                  least 1, which leaves two threads free for the desktop)
#   -b binary      GA binary (default build/cuCga or build/cga, whichever this
#                  repository builds; built with CMake if missing)
#   -i ics         binomial ICs for the final evaluation (default 10000)
#   -o outdir      output directory (default results/<timestamp>)
#   -a args        extra GA options passed to every run, as one quoted string,
#                  e.g. -a "-r single -g 200"; the seed, ICs and CSV are set by
#                  this script, so -s/-n/-o/-v (and long forms) are not allowed
#   -x             do not draw the evolution plots (drawn only when local
#                  plotting tools are present)
#
# Produces <outdir>/results.csv (one row per run), <outdir>/evolution.png (all
# runs), <outdir>/manifest.txt (environment, plus the effective GA parameters
# reported by the binary, training ICs and whether they are shared included) and
# <outdir>/runs/seed_NNNN/ holding each run's logs/output.log, stdout, stderr,
# evolution.csv and evolution.png. The plots need local plotting tools (.venv
# and scripts/plot_evolution.py), which are not part of this repository;
# without them they are skipped with a warning.
set -euo pipefail
export LC_ALL=C

RUNS=100
FIRST_SEED=1
THREADS=$(nproc 2>/dev/null || echo 4)
JOBS=$(( THREADS > 2 ? THREADS - 2 : 1 ))
BIN=""
ICS=10000
OUT=""
PLOT=1
GA_ARGS=""

while getopts "n:s:j:b:i:o:a:xh" opt; do
    case "$opt" in
        n) RUNS=$OPTARG ;;
        s) FIRST_SEED=$OPTARG ;;
        j) JOBS=$OPTARG ;;
        b) BIN=$OPTARG ;;
        i) ICS=$OPTARG ;;
        o) OUT=$OPTARG ;;
        a) GA_ARGS=$OPTARG ;;
        x) PLOT=0 ;;
        *) sed -n '2,23p' "$0"; exit 1 ;;
    esac
done

for a in $GA_ARGS; do
    case "$a" in
        -s*|--seed*|-n*|--ics*|-o*|--csv*|-v*|--validate*|-h|--help)
            echo "error: -a must not contain '$a' (seed, ICs, CSV and validation are set by this script)" >&2
            exit 1 ;;
    esac
done

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
[ -n "$OUT" ] || OUT="results/$(date +%Y%m%d-%H%M%S)"
if [ -z "$BIN" ]; then
    # cuCGA has the CUDA backend, cga the CPU one; both build with CMake into build/
    if [ -f src/kernel.cu ]; then BIN=build/cuCga; else BIN=build/cga; fi
    [ -x "$BIN" ] || { cmake -S . -B build && cmake --build build; }
fi
BIN=$(realpath "$BIN")
PY="$ROOT/.venv/bin/python"
if [ "$PLOT" -eq 1 ] && { [ ! -x "$PY" ] || [ ! -f "$ROOT/scripts/plot_evolution.py" ]; }; then
    echo "warning: local plotting tools not found (.venv, scripts/plot_evolution.py), no plots" >&2
    PLOT=0
fi
mkdir -p "$OUT/runs"
OUT=$(realpath "$OUT")

{
    echo "date: $(date -Is)"
    echo "git: $(git rev-parse HEAD 2>/dev/null || echo n/a)$(git diff --quiet 2>/dev/null || echo ' (dirty)')"
    echo "binary: $BIN ($(sha256sum "$BIN" | cut -c1-16))"
    echo "runs: $RUNS  seeds: $FIRST_SEED..$((FIRST_SEED+RUNS-1))  ics: $ICS  jobs: $JOBS"
    echo "ga args: ${GA_ARGS:-(defaults)}"
    echo "gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo n/a)"
    echo "nvcc: $(nvcc --version 2>/dev/null | tail -1 || echo n/a)"
} > "$OUT/manifest.txt"

# Training ICs reported by the binary in its parameter line (first line of stderr), so
# that -I/--train-ics in any spelling, or its default, is what gets recorded and plotted.
# Empty for binaries that predate -I.
train_ics() {
    sed -nE '1s/.* train-ics=([0-9]+) .*/\1/p' "$1"
}

run_one() {
    local seed=$1 dir n
    dir=$(printf "%s/runs/seed_%04d" "$OUT" "$seed")
    mkdir -p "$dir/logs"
    # shellcheck disable=SC2086 # GA_ARGS is split into options on purpose
    (cd "$dir" && "$BIN" $GA_ARGS -s "$seed" -n "$ICS" --csv evolution.csv > stdout.txt 2> stderr.txt)
    if [ "$PLOT" -eq 1 ]; then
        n=$(train_ics "$dir/stderr.txt")
        "$PY" "$ROOT/scripts/plot_evolution.py" run "$dir/evolution.csv" -o "$dir/evolution.png" ${n:+--training-ics "$n"} \
            --stdout "$dir/stdout.txt" --stderr "$dir/stderr.txt" --title "Evolution, seed $seed" > /dev/null
    fi
}
export -f run_one train_ics
export OUT BIN ICS PLOT PY ROOT GA_ARGS

start=$(date +%s)
seq "$FIRST_SEED" $((FIRST_SEED+RUNS-1)) | xargs -P "$JOBS" -I{} bash -c 'run_one {} && echo -n "." >&2'
echo >&2

echo "seed,train_best,rule,nics,perf,perf_strict" > "$OUT/results.csv"
for f in "$OUT"/runs/seed_*/stdout.txt; do
    # seed=1 train_best=98 rule=... nics=10000 perf=0.6512 perf_strict=0.6512
    sed -E 's/^seed=([0-9]+) train_best=([0-9]+) rule=([0-9a-f]+) nics=([0-9]+) perf=([0-9.]+) perf_strict=([0-9.]+)$/\1,\2,\3,\4,\5,\6/' "$f"
done | sort -t, -k1,1n >> "$OUT/results.csv"

# Effective GA parameters, as printed by the binary (identical across runs but for the seed)
first=$(printf "%s/runs/seed_%04d/stderr.txt" "$OUT" "$FIRST_SEED")
params=$(head -1 "$first" 2>/dev/null | sed -E 's/ seed=[0-9]+//' || true)
N=$(train_ics "$first" 2>/dev/null || true)
case "$params" in
    *"(shared)"*)         SHARED=yes ;;
    *"(per individual)"*) SHARED=no ;;
    *)                    SHARED=n/a ;;
esac
{
    echo "train ics: ${N:-n/a}  shared ics: $SHARED"
    echo "ga params: ${params:-n/a}"
} >> "$OUT/manifest.txt"

if [ "$PLOT" -eq 1 ]; then
    "$PY" scripts/plot_evolution.py experiment "$OUT" -o "$OUT/evolution.png" ${N:+--training-ics "$N"} \
        --title "Evolution across runs ($(basename "$OUT"))${GA_ARGS:+: $GA_ARGS}" >&2
fi

n=$(($(wc -l < "$OUT/results.csv") - 1))
echo "elapsed: $(( $(date +%s) - start ))s" >> "$OUT/manifest.txt"
echo "$n/$RUNS runs collected in $OUT/results.csv" >&2
[ "$n" -eq "$RUNS" ]
