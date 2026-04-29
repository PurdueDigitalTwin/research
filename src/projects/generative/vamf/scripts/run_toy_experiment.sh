#!/usr/bin/bash
# Run the toy experiment for Variance-Aware MeanFlows
#
# Author: Juanwu Lu
# Date: 2026-03-27
#
# Sweeps {toy_datasets} x {methods} x {seeds} through `run_toy.py`.
# Skips runs whose output JSON already exists (idempotent).
#
# Each iteration invokes `bazelisk run` for the run_toy target. The build
# configuration (cuda / cpu / mps / tpu) is picked via the PLATFORM env var.
#
# Usage:
#   ./run_toy_experiment.sh
#   PLATFORM=cpu  WORK_DIR=/path/to/out STEPS=100000 ./run_toy_experiment.sh
#   DATASETS="eight_gaussians swiss_roll" METHODS="vamf_tw" SEEDS="0 1 2" \
#       ./run_toy_experiment.sh

set -euo pipefail

# Resolve workspace root from git so the script works no matter where it's run.
WORKSPACE_DIR="$(git rev-parse --show-toplevel)"
cd "$WORKSPACE_DIR"

WORK_DIR="${WORK_DIR:-${WORKSPACE_DIR}/logs/vamf/toy_exp/200k}"
STEPS="${STEPS:-200000}"
# Use `read -ra` to split the space-separated strings into arrays.
# Refer to https://www.shellcheck.net/wiki/SC2206 for details.
read -ra DATASETS <<< "${DATASETS:-checkerboard eight_gaussians two_moons swiss_roll}"
read -ra METHODS  <<< "${METHODS:-meanflow vamf_l2 vamf_tw}"
read -ra SEEDS    <<< "${SEEDS:-42 0 1}"
PLATFORM="${PLATFORM:-cuda}"

mkdir -p "$WORK_DIR"

echo "Workspace : $WORKSPACE_DIR"
echo "Output    : $WORK_DIR"
echo "Platform  : $PLATFORM"
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
      bazelisk run "--config=${PLATFORM}" \
        //src/projects/generative/vamf/experiments:run_toy -- \
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
