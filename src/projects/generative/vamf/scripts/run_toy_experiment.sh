#!/usr/bin/bash
# Run the toy experiment for Variance-Aware MeanFlows
#
# Author: Juanwu Lu
# Date: 2026-03-27
#
# Sweeps {toy_datasets} x {methods} x {seeds} through `run_toy.py` on CUDA.
# Skips runs whose output JSON already exists (idempotent).
#
# This script is meant to be invoked via `bazel run`:
#
#   bazelisk run --config=cuda //src/projects/generative/meanflow/scripts:run_toy_experiment
#   bazelisk run --config=cpu  //src/projects/generative/meanflow/scripts:run_toy_experiment -- ...
#
# The bazel build pre-builds the `run_toy` py_binary as a data dep, so the
# script invokes its launcher directly — no nested `bazelisk run`, no per-
# iteration rebuild, no server-lock contention. The build configuration
# (cuda / cpu / mps) is selected by the outer `--config=...` flag.
#
# Override the sweep grid via env vars:
#   WORK_DIR=/path/to/out STEPS=100000 \
#       bazelisk run --config=cuda //...:run_toy_experiment
#   DATASETS="eight_gaussians swiss_roll" METHODS="vamf_tw" SEEDS="0 1 2" \
#       bazelisk run --config=cuda //...:run_toy_experiment

set -euo pipefail

# --- runfiles bootstrap (bazel boilerplate) -----------------------------------
# Ref: https://github.com/bazelbuild/bazel/blob/master/tools/bash/runfiles/runfiles.bash
# --- begin runfiles.bash initialization v3 ---
set +e
f=bazel_tools/tools/bash/runfiles/runfiles.bash
source "${RUNFILES_DIR:-/dev/null}/$f" 2>/dev/null \
  || source "$(grep -sm1 "^$f " "${RUNFILES_MANIFEST_FILE:-/dev/null}" | cut -f 2- -d ' ')" 2>/dev/null \
  || source "$0.runfiles/$f" 2>/dev/null \
  || source "$(grep -sm1 "^$f " "$0.runfiles_manifest" | cut -f 2- -d ' ')" 2>/dev/null \
  || { echo >&2 "ERROR: cannot find $f"; exit 1; }
set -e
# --- end runfiles.bash initialization v3 ---

RUN_TOY="$(rlocation _main/src/projects/generative/vamf/experiments/run_toy)"
if [[ -z "$RUN_TOY" || ! -x "$RUN_TOY" ]]; then
  echo "ERROR: could not locate run_toy launcher in runfiles." >&2
  exit 1
fi

# Resolve workspace root: BUILD_WORKSPACE_DIRECTORY is set by `bazel run`.
WORKSPACE_DIR="${BUILD_WORKSPACE_DIRECTORY:-$(git rev-parse --show-toplevel)}"

WORK_DIR="${WORK_DIR:-${WORKSPACE_DIR}/logs/vamf/toy_exp/200k_fixed}"
STEPS="${STEPS:-200000}"
DATASETS=( "${DATASETS:-checkerboard eight_gaussians two_moons swiss_roll}" )
METHODS=( "${METHODS:-meanflow vamf_l2 vamf_tw}" )
SEEDS=( "${SEEDS:-42 0 1}" )

mkdir -p "$WORK_DIR"

echo "Workspace : $WORKSPACE_DIR"
echo "Output    : $WORK_DIR"
echo "run_toy   : $RUN_TOY"
echo "Steps     : $STEPS"
echo "Datasets  : ${DATASETS[*]}"
echo "Methods   : ${METHODS[*]}"
echo "Seeds     : ${SEEDS[*]}"
echo

for seed in "${SEEDS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
      out="${WORK_DIR}/${ds}_${method}_${seed}.json"
      if [[ -f "$out" ]]; then
        echo "[skip] ${out} already exists"
        continue
      fi

      echo "=== Running ${ds} / ${method} / seed=${seed} ==="
      "$RUN_TOY" \
        --dataset="$ds" \
        --method="$method" \
        --steps="$STEPS" \
        --seed="$seed" \
        --exact_trace=true \
        --work_dir="$WORK_DIR" \
        2>&1 | grep -E 'loss=|Training finished|Saved'
      echo "--- Done: ${ds} / ${method} / seed=${seed} ---"
      echo
    done
  done
done

echo "All sweeps complete."
