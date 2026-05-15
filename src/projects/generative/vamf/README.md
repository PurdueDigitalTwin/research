# Variance-Aware Mean Flows

This directory contains the code for running the experiments for the variance-aware mean flows project. The code is organized as follows:

```text
src/projects/generative/vamf
├── BUILD
├── experiments                 # entry points for running experiments
│   ├── BUILD
│   ├── run_diagnostic.py
│   └── run_toy.py
├── figures                     # entry points for plotting figures
│   ├── BUILD
│   ├── plot_diagnostics.py
│   ├── plot_illustration.py
│   └── plot_toy.py
├── model                       # code for the model and training
│   ├── BUILD
│   ├── tests
│   │   └── test_trace.py
│   └── trace.py
├── README.md
└── scripts                     # shell scripts for running experiments
    └── run_toy_experiment.sh
```

## Getting Started

In this paper, our main results consists of a sweep toy experiments where we compare the performance of variance-aware mean flows to the baseline mean flows, a diagnostic experiment where we inspect:

- the existence of Jacobian Variance Amplification in original Mean Flows,
- the curvature gap vs. the interval length, and
- the growth of the Jacobian norm.

### Toy Experiments

To run the toy experiments, use the `run_toy_experiment` shell script. For example, to reproduce the sweep experiment on the four datasets in our paper, you can use the following command:

```bash
chmod +x src/projects/generative/vamf/scripts/run_toy_experiment.sh

DATASETS="checkerboard eight_gaussians two_moons swiss_roll"
METHODS="meanflow vamf_l2 vamf_tw"
PLATFORM="cuda"
SEEDS="42 0 1"

WORK_DIR=$(pwd)/logs/vamf/toy_exp/200k \
    DATASETS="$DATASETS" METHODS="$METHODS" SEEDS="$SEEDS" \
    ./src/projects/generative/vamf/scripts/run_toy_experiment.sh
```

This will kick start the sweep and save the results in `logs/vamf/toy_exp/200k`. You can change the `WORK_DIR` to save the results to a different location.

To run the sweep experiment on high-dimensional Gaussians, you can use the following command:

```bash
DATASETS="dgmm_2 dgmm_4 dgmm_8 dgmm_16 dgmm_32 dgmm_64"
METHODS="meanflow vamf_l2 vamf_tw"
PLATFORM="cuda"
SEEDS="42 0 1"

WORK_DIR=$(pwd)/logs/vamf/toy_exp/dgmm_200k \
    DATASETS="$DATASETS" METHODS="$METHODS" SEEDS="$SEEDS" \
    ./src/projects/generative/vamf/scripts/run_toy_experiment.sh
```

### Diagnostic Experiments
