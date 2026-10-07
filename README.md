# Introduction

Here you'll  find the source code used to test a GA for the CA problem known as DCT.

This is the source code for the IEEE paper [Ternary representation improves the search for binary, one-dimensional density classifier cellular automata](https://ieeexplore.ieee.org/document/5949850) developed by myself and my masters teacher and coleague Pedro Paulo Balbi de Oliveira. As well as improvements on the same paper and GA.

> **Paper version.** The code published for the paper is tagged [`cec2011-paper`](https://github.com/minterciso/cuCGA/tree/cec2011-paper) (commit `6390aa3`). Later commits fix bugs found in that version, move the build to CMake and add options, so results produced with later commits are not directly comparable with the paper's.

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

## Running experiments

`scripts/run_experiment.sh -n RUNS [-s FIRST_SEED] [-a "GA options"]` runs RUNS consecutive seeds (e.g. `-a "-r single -g 200"`), one process per seed, and collects each run's final line in `results.csv`, its per-generation statistics (`--csv`) and its logs in its own folder, with a `manifest.txt` recording the binary, the environment and the effective GA parameters. Analysis and plotting tools are not part of this repository.
