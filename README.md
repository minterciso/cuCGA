# Introduction

Here you'll  find the source code used to test a GA for the CA problem known as DCT.

This is the source code for the IEEE paper [Ternary representation improves the search for binary, one-dimensional density classifier cellular automata](https://ieeexplore.ieee.org/document/5949850) developed by myself and my masters teacher and coleague Pedro Paulo Balbi de Oliveira.

More documentation to follow

# Building and running

Requires CMake >= 3.24, the CUDA toolkit and a C compiler. The GPU architecture defaults to the one(s) present on the machine (`-DCMAKE_CUDA_ARCHITECTURES=89` to override).

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build           # template decoding, and GPU vs CPU CA results
mkdir -p logs && ./build/cuCga --seed 42
./build/cuCga --representation single --seed 42
./build/cuCga --generations 500 --seed 42
./build/cuCga --seed 42 --csv evolution.csv   # per-generation fitness statistics, for plotting
./build/cuCga --validate 005f005f005f005f005fff5f005fff5f   # evaluate a rule (GKL) on 10^4 binomial ICs
./build/cuCga --help
```

The host code in `src/` is the same as in [cga](https://github.com/minterciso/cga), the CPU version of this GA; only the CA backend differs (`src/kernel.cu` here, `src/backend_cpu.c` there, behind `src/backend.h`). For the same seed and options both programs produce the same run.

## Plots

The plotting tools run in a project-local Python virtual environment, so nothing is installed in the system Python:

```sh
scripts/setup_venv.sh                       # creates .venv and installs requirements.txt
.venv/bin/python scripts/plot_evolution.py run evolution.csv -o evolution.png \
    --stdout stdout.txt --stderr stderr.txt   # one run (CSV from --csv)
.venv/bin/python scripts/plot_evolution.py experiment results/<dir>   # several runs
```

`scripts/run_experiment.sh -n RUNS [-a "GA options"]` runs RUNS seeds (e.g. `-a "-r single -g 200"`) and, when `.venv` exists, draws `evolution.png` for every run and one for the whole experiment (median and quartiles across runs, plus the final binomial performance in the bins of Table V of the paper); `-x` skips the plots. The curves are **training** fitness (each individual on its own random ICs), which predicts the final binomial performance poorly; that performance is shown separately (★ on the per-run plot).
