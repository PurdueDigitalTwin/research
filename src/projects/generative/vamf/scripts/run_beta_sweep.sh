#!/usr/bin/bash
# Full beta-sweep for theory validation of Theorem 4.
#
# Author: Juanwu Lu
# Date: 2026-05-01
#
# Sweeps {datasets} x {beta values} x {seeds} through `run_toy.py` using
# method=vamf_tmix with the --tangent_beta flag. beta=0 recovers vanilla
# MeanFlow; beta=1 recovers the EMA-tangent instantiation. Interior values
# trace the empirical M(beta) curve for direct comparison against the
# closed-form prediction of Theorem 4.
#
# Each run writes to a beta-namespaced subdirectory so existing beta=0
# and beta=1 results from `run_toy_experiment.sh` are not overwritten.
#
# Usage:
#   ./run_beta_sweep.sh
#   PLATFORM=cpu STEPS=100000 ./run_beta_sweep.sh
#   DATASETS="swiss_roll pinwheel" SEEDS="0 1 2" ./run_beta_sweep.sh
#   BETAS="0.0 0.5 1.0" ./run_beta_sweep.sh
#   MEASURE_GRAD_VAR_EVERY=2000 ./run_beta_sweep.sh   # enables Tr(Cov[g]) probing
#
# Outputs: ${WORK_DIR}/beta_${BETA}/${dataset}_vamf_tmix_${seed}.json

set -euo pipefail

WORKSPACE_DIR="$(git rev-parse --show-toplevel)"
cd "$WORKSPACE_DIR"

WORK_DIR="${WORK_DIR:-${WORKSPACE_DIR}/logs/vamf/beta_sweep_200k}"
STEPS="${STEPS:-200000}"

read -ra DATASETS <<< "${DATASETS:-checkerboard eight_gaussians two_moons swiss_roll two_spirals pinwheel}"
read -ra SEEDS    <<< "${SEEDS:-42 0 1}"
read -ra BETAS    <<< "${BETAS:-0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0}"
PLATFORM="${PLATFORM:-cuda}"
MEASURE_GRAD_VAR_EVERY="${MEASURE_GRAD_VAR_EVERY:-0}"
MEASURE_GRAD_VAR_N_BATCHES="${MEASURE_GRAD_VAR_N_BATCHES:-8}"

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
        --measure_grad_var_every="$MEASURE_GRAD_VAR_EVERY" \
        --measure_grad_var_n_batches="$MEASURE_GRAD_VAR_N_BATCHES" \
        --work_dir="$beta_dir" \
        2>&1 | grep -E 'loss=|Training finished|Saved'
      echo "--- Done [$count/$total]: β=${beta} ${ds} seed=${seed} ---"
      echo
    done
  done
done

echo "All runs complete. Outputs in: $WORK_DIR/beta_*/"
