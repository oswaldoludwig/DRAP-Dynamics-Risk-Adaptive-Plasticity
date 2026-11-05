#!/usr/bin/env bash
if [[ -z "${BASH_VERSION:-}" ]]; then
  exec bash "$0" "$@"
fi
# Sequential KappaTune + dynamics-risk-weighted adjustment suppression.
# Run this script inside an existing GPU allocation.
set -Eeuo pipefail
trap 'status=$?; echo "ERROR: ${BASH_SOURCE[0]}:${LINENO}: command failed (${status}): ${BASH_COMMAND}" >&2; exit ${status}' ERR
export PYTHONUNBUFFERED=1
export SINGULARITYENV_PYTHONUNBUFFERED=1

PROJECT_DIR="${PROJECT_DIR:-     <include here the path for your project>    }"
CONTAINER="${CONTAINER:-${PROJECT_DIR}/kappatune_vla.sif}"
TRAIN_SCRIPT="${PROJECT_DIR}/train_continual_kappa_risk_suppression.py"
RISK_SCRIPT="${RISK_SCRIPT:-${PROJECT_DIR}/compute_libero_risk_scores.py}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs/kappa_risk_v1}"



# RISK_ROOT="${RISK_ROOT:-${PROJECT_DIR}/risk_scores}"


RISK_ROOT="${RISK_ROOT:-${PROJECT_DIR}/risk_scores_uniform}"




CACHE_ROOT="${CACHE_ROOT:-${PROJECT_DIR}/cache}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-openvla/openvla-7b}"
SUITES_CSV="${SUITES:-libero_spatial,libero_object,libero_goal,libero_long}"

EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-10}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LR="${LR:-5e-5}"
NUM_DEMOS="${NUM_DEMOS:-0}"

# Original KappaTune-only trainable budget.
BUDGET_VISION="${BUDGET_VISION:-50}"
BUDGET_LANGUAGE="${BUDGET_LANGUAGE:-90}"
BUDGET_ACTION_HEAD="${BUDGET_ACTION_HEAD:-1}"

# For training without KappaTune:
# BUDGET_VISION="${BUDGET_VISION:-999999}"
# BUDGET_LANGUAGE="${BUDGET_LANGUAGE:-999999}"
# BUDGET_ACTION_HEAD="${BUDGET_ACTION_HEAD:-999999}"

# Risk-weighted tensor consolidation.
# The penalty uses summed squared drift over every selected parameter.
# Start conservatively; sweep 1e-9, 1e-8, and 1e-7 after validating acquisition.
# Dynamics-risk-weighted adjustment suppression.  Threshold 0 is fully
# relaxed, isolating the importance-denominator effect in the first study.
SUPPRESSION_BETA="${SUPPRESSION_BETA:-1.0}"
SUPPRESSION_POWER="${SUPPRESSION_POWER:-2.0}"
UPDATE_THRESHOLD="${UPDATE_THRESHOLD:-0.0}"

# To obtain the no-DRAP control (remove it after the control experiment):
# SUPPRESSION_BETA="${SUPPRESSION_BETA:-0.0}"


IMPORTANCE_DECAY="${IMPORTANCE_DECAY:-1.0}"
IMPORTANCE_MAX_BATCHES="${IMPORTANCE_MAX_BATCHES:-256}"
# One frame per importance backward pass avoids cross-example cancellation.
IMPORTANCE_BATCH_SIZE="${IMPORTANCE_BATCH_SIZE:-1}"

# Offline dynamics-risk model.
RISK_WINDOW="${RISK_WINDOW:-32}"
RISK_MIN_TRANSITIONS="${RISK_MIN_TRANSITIONS:-16}"
RISK_RIDGE="${RISK_RIDGE:-1e-3}"
RISK_CLIP_LOW="${RISK_CLIP_LOW:-0.25}"
RISK_CLIP_HIGH="${RISK_CLIP_HIGH:-4.0}"
RECOMPUTE_RISK="${RECOMPUTE_RISK:-0}"

mkdir -p \
  "${RUN_ROOT}" "${RISK_ROOT}" \
  "${CACHE_ROOT}/hf" "${CACHE_ROOT}/torch" "${CACHE_ROOT}/tmp"

for file in \
  "${CONTAINER}" "${TRAIN_SCRIPT}" "${RISK_SCRIPT}" \
  "${PROJECT_DIR}/risk_regularization.py" \
  "${PROJECT_DIR}/risk_weighted_tensor_importance.py"; do
  [[ -f "${file}" ]] || { echo "ERROR: missing ${file}" >&2; exit 2; }
done
command -v singularity >/dev/null 2>&1 || {
  echo "ERROR: singularity is not on PATH" >&2
  exit 2
}

export SINGULARITYENV_HF_HOME="${CACHE_ROOT}/hf"
export SINGULARITYENV_HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/hf/hub"
export SINGULARITYENV_TORCH_HOME="${CACHE_ROOT}/torch"
export SINGULARITYENV_TMPDIR="${CACHE_ROOT}/tmp"
export SINGULARITYENV_PYTHONNOUSERSITE=1
export SINGULARITYENV_TOKENIZERS_PARALLELISM=false
[[ -n "${HF_TOKEN:-}" ]] && export SINGULARITYENV_HF_TOKEN="${HF_TOKEN}"

THREADS="${SLURM_CPUS_PER_TASK:-32}"
export SINGULARITYENV_OMP_NUM_THREADS="${THREADS}"
export SINGULARITYENV_MKL_NUM_THREADS="${THREADS}"
export SINGULARITYENV_OPENBLAS_NUM_THREADS="${THREADS}"
export SINGULARITYENV_NUMEXPR_NUM_THREADS="${THREADS}"

run_container() {
  singularity exec --nv --cleanenv \
    --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
    --pwd "${PROJECT_DIR}" \
    "${CONTAINER}" "$@"
}

echo "Node:          $(hostname)"
echo "Project:       ${PROJECT_DIR}"
echo "Container:     ${CONTAINER}"
echo "Trainer:       ${TRAIN_SCRIPT}"
echo "Risk scorer:   ${RISK_SCRIPT}"
[[ -f "${TRAIN_SCRIPT}" ]] || { echo "ERROR: missing Python trainer: ${TRAIN_SCRIPT}" >&2; exit 2; }
[[ -f "${PROJECT_DIR}/dynamics_risk_adjustment_suppression.py" ]] || {
  echo "ERROR: missing suppression optimizer module" >&2; exit 2;
}
echo "Run root:      ${RUN_ROOT}"
echo "Risk root:     ${RISK_ROOT}"
nvidia-smi -L || true

# Import the complete stack before downloading data or scoring tensors. Any
# version mismatch now prints a full traceback instead of an empty Slurm exit.
run_container python3 -u -c '
import train_continual
import train_continual_kappa_risk_suppression
import dynamics_risk_adjustment_suppression
import compute_libero_risk_scores
import risk_regularization
import risk_weighted_tensor_importance
print("Python suppression-framework import preflight OK")
'

help_text="$(run_container python3 -u "${TRAIN_SCRIPT}" --help 2>&1)"
for option in risk-scores suppression-beta suppression-power update-threshold importance-max-batches importance-batch-size; do
  grep -q "${option}" <<<"${help_text}" || {
    echo "ERROR: trainer does not support --${option}" >&2
    exit 2
  }
done

IFS=',' read -r -a SUITES_ARRAY <<<"${SUITES_CSV}"

# ---------------------------------------------------------------------------
# Phase 1: precompute policy-independent per-frame risk once per requested suite
# ---------------------------------------------------------------------------
for suite in "${SUITES_ARRAY[@]}"; do
  risk_file="${RISK_ROOT}/${suite}_risk.npz"
  if [[ "${RECOMPUTE_RISK}" == "1" || ! -f "${risk_file}" ]]; then
    echo "Computing demonstration risk: ${suite} -> ${risk_file}"
    run_container python3 -u "${RISK_SCRIPT}" \
      --suite "${suite}" \
      --window "${RISK_WINDOW}" \
      --min-transitions "${RISK_MIN_TRANSITIONS}" \
      --ridge "${RISK_RIDGE}" \
      --clip-low "${RISK_CLIP_LOW}" \
      --clip-high "${RISK_CLIP_HIGH}" \
      --output "${risk_file}" \
      --hf-home "${CACHE_ROOT}/hf"
  else
    echo "Using existing risk scores: ${risk_file}"
  fi
  [[ -f "${risk_file}" ]] || { echo "ERROR: risk file not created" >&2; exit 3; }
done

# ---------------------------------------------------------------------------
# Phase 2: sequential training and risk-weighted consolidation
# ---------------------------------------------------------------------------
previous_checkpoint="${BASE_CHECKPOINT}"
for stage_idx in "${!SUITES_ARRAY[@]}"; do
  suite="${SUITES_ARRAY[$stage_idx]}"
  stage_dir="${RUN_ROOT}/stage${stage_idx}_${suite#libero_}"
  risk_file="${RISK_ROOT}/${suite}_risk.npz"
  mkdir -p "${stage_dir}"

  if [[ -f "${stage_dir}/TRAINING_COMPLETE" ]]; then
    echo "Skipping completed stage: ${stage_dir}"
    previous_checkpoint="${stage_dir}"
    continue
  fi

  args=(
    --suite "${suite}"
    --init-checkpoint "${previous_checkpoint}"
    --out-dir "${stage_dir}"
    --risk-scores "${risk_file}"
    --budget-vision "${BUDGET_VISION}"
    --budget-language "${BUDGET_LANGUAGE}"
    --budget-action-head "${BUDGET_ACTION_HEAD}"
    --suppression-beta "${SUPPRESSION_BETA}"
    --suppression-power "${SUPPRESSION_POWER}"
    --update-threshold "${UPDATE_THRESHOLD}"
    --importance-decay "${IMPORTANCE_DECAY}"
    --importance-max-batches "${IMPORTANCE_MAX_BATCHES}"
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --grad-accum "${GRAD_ACCUM}"
    --lr "${LR}"
  )
  [[ "${NUM_DEMOS}" != "0" ]] && args+=(--num-demos "${NUM_DEMOS}")

  printf '\n============================================================\n'
  echo "Stage:                  ${stage_idx}"
  echo "Suite:                  ${suite}"
  echo "Input checkpoint:       ${previous_checkpoint}"
  echo "Output:                 ${stage_dir}"
  echo "Risk file:              ${risk_file}"
  echo "KappaTune budget:       ${BUDGET_VISION}/${BUDGET_LANGUAGE}/${BUDGET_ACTION_HEAD}"
  echo "AdamW LR:               ${LR}"
  echo "Suppression beta:       ${SUPPRESSION_BETA}"
  echo "Suppression power:      ${SUPPRESSION_POWER}"
  echo "Update threshold:       ${UPDATE_THRESHOLD} (0 = relaxed)"
  echo "Importance decay:       ${IMPORTANCE_DECAY}"
  echo "Importance max batches: ${IMPORTANCE_MAX_BATCHES}"
  printf '============================================================\n\n'

  run_container python3 -u "${TRAIN_SCRIPT}" "${args[@]}"

  for artifact in \
    selection.json action_stats.json loss_history.jsonl \
    risk_tensor_importance.pt risk_tensor_importance_report.json; do
    [[ -f "${stage_dir}/${artifact}" ]] || {
      echo "ERROR: missing ${stage_dir}/${artifact}" >&2
      exit 4
    }
  done

  touch "${stage_dir}/TRAINING_COMPLETE"
  previous_checkpoint="${stage_dir}"
done

echo "All requested KappaTune + dynamics-risk adjustment-suppression stages completed."
