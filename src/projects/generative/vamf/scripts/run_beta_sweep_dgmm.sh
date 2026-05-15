#!/usr/bin/bash
# DGMM beta-sweep for high-dimensional theory validation.
#
# Author: Juanwu Lu
# Date: 2026-05-01
#
# Same structure as run_beta_sweep.sh but on dgmm_{2,4,8,16,32,64}.
# Tests Theorem 4's prediction that the optimal beta → 0 as ‖b‖² shrinks
# toward the isotropic-data limit.
#
# Usage:
#   ./run_beta_sweep_dgmm.sh
#   PLATFORM=cuda BETAS="0.0 0.5 1.0" ./run_beta_sweep_dgmm.sh

set -euo pipefail

WORKSPACE_DIR="$(git rev-parse --show-toplevel)"
cd "$WORKSPACE_DIR"

WORK_DIR="${WORK_DIR:-${WORKSPACE_DIR}/logs/vamf/beta_sweep_dgmm_200k}"
STEPS="${STEPS:-200000}"

read -ra DATASETS <<< "${DATASETS:-dgmm_2 dgmm_4 dgmm_8 dgmm_16 dgmm_32 dgmm_64}"
read -ra SEEDS    <<< "${SEEDS:-42 0 1}"
read -ra BETAS    <<< "${BETAS:-0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0}"
PLATFORM="${PLATFORM:-cuda}"

echo "Workspace : $WORKSPACE_DIR"
echo "Work dir  : $WORK_DIR"
echo "Platform  : $PLATFORM"
echo "Steps     : $STEPS"
echo "Datasets  : ${DATASETS[*]}"
echo "Seeds     : ${SEEDS[*]}"
echo "Betas     : ${BETAS[*]}"
total=$((${#DATASETS[@]} * ${#SEEDS[@]} * ${#BETAS[@]}))
echo "Total runs: $total"
echo

count=0
for beta in "${BETAS[@]}"; do
  beta_dir="${WORK_DIR}/beta_${beta}"
  mkdir -p "$beta_dir"
  for seed in "${SEEDS[@]}"; do
    for ds in "${DATASETS[@]}"; do
      count=$((count + 1))
      out="${beta_dir}/${ds}_vamf_tmix_${seed}.json"
      if [[ -f "$out" ]]; then
        echo "[skip $count/$total] β=${beta} ${ds} seed=${seed}"
        continue
      fi

      echo "=== [$count/$total] β=${beta} ${ds} seed=${seed} ==="
      bazelisk run "--config=${PLATFORM}" \
        //src/projects/generative/vamf/experiments:run_toy -- \
        --dataset="$ds" \
        --method="vamf_tmix" \
        --tangent_beta="$beta" \
        --steps="$STEPS" \
        --seed="$seed" \
        --exact_trace=true \
        --work_dir="$beta_dir" \
        2>&1 | grep -E 'loss=|Training finished|Saved'
      echo "--- Done [$count/$total]: β=${beta} ${ds} seed=${seed} ---"
      echo
    done
  done
done

echo "All runs complete. Outputs in: $WORK_DIR/beta_*/"
