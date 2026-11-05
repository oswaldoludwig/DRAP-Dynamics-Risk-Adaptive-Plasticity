#!/usr/bin/env bash
#SBATCH --job-name=kt-vla-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --output=/include/here/the/path/for/your/project/slurm_logs/%x-%j.out
#SBATCH --error=/include/here/the/path/for/your/project/slurm_logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/include/here/the/path/for/your/project}"
CONTAINER="${CONTAINER:-${PROJECT_DIR}/kappatune_vla_eval.sif}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${PROJECT_DIR}/eval_forgetting.py}"
METHOD="${METHOD:-kappatune}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs/${METHOD}}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_DIR}/results/${METHOD}}"
CACHE_ROOT="${CACHE_ROOT:-${PROJECT_DIR}/cache}"
PYDEPS_ROOT="${PYDEPS_ROOT:-${PROJECT_DIR}/eval_pydeps}"

EPISODES_PER_TASK="${EPISODES_PER_TASK:-5}"

SEED="${SEED:-11}"

SAVE_VIDEOS="${SAVE_VIDEOS:-0}"
TASK_IDS="${TASK_IDS:-}"
MAX_STEPS="${MAX_STEPS:-}"
NO_FLIP="${NO_FLIP:-0}"
RUN_SMOKE_TEST="${RUN_SMOKE_TEST:-1}"
RESULTS_JSON="${RESULTS_JSON:-${RESULT_ROOT}/forgetting_matrix.json}"
VIDEO_DIR="${VIDEO_DIR:-${RESULT_ROOT}/videos}"
SANITY_DIR="${SANITY_DIR:-${RESULT_ROOT}/sanity}"

# Out-of-the-box (zero-shot, pre-fine-tuning) reference row. Evaluated with
# NO action_stats.json of its own, so eval_forgetting.py computes each
# suite's stats fresh from that suite's own data -- the same correct
# per-suite normalization every fine-tuned stage gets, so this is a fair
# baseline rather than one skewed by reused/mismatched stats. Its stage name
# must contain "baseline" (checked by eval_forgetting.py's stage_sort_key /
# is_continual_stage) so it sorts first and is excluded from forgetting math.
RUN_BASELINE="${RUN_BASELINE:-0}"
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-openvla/openvla-7b}"
BASELINE_STAGE_NAME="${BASELINE_STAGE_NAME:-baseline_zeroshot}"

mkdir -p "${PROJECT_DIR}/slurm_logs" "${RESULT_ROOT}" "${VIDEO_DIR}" \
    "${SANITY_DIR}" "${CACHE_ROOT}/hf" "${CACHE_ROOT}/torch" \
    "${CACHE_ROOT}/tmp" "${CACHE_ROOT}/libero_config" "${PYDEPS_ROOT}"

[[ -f "${CONTAINER}" ]] || { echo "ERROR: missing container: ${CONTAINER}" >&2; exit 2; }
[[ -f "${EVAL_SCRIPT}" ]] || { echo "ERROR: missing evaluator: ${EVAL_SCRIPT}" >&2; exit 2; }
command -v singularity >/dev/null 2>&1 || { echo "ERROR: singularity not found" >&2; exit 2; }

export SINGULARITYENV_HF_HOME="${CACHE_ROOT}/hf"
export SINGULARITYENV_HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/hf/hub"
export SINGULARITYENV_TORCH_HOME="${CACHE_ROOT}/torch"
export SINGULARITYENV_TMPDIR="${CACHE_ROOT}/tmp"
export SINGULARITYENV_LIBERO_CONFIG_PATH="${CACHE_ROOT}/libero_config"
export SINGULARITYENV_KT_PYDEPS="${PYDEPS_ROOT}"
export SINGULARITYENV_PYTHONNOUSERSITE=1
export SINGULARITYENV_TOKENIZERS_PARALLELISM=false
export SINGULARITYENV_MUJOCO_GL=osmesa
export SINGULARITYENV_PYOPENGL_PLATFORM=osmesa
[[ -n "${HF_TOKEN:-}" ]] && export SINGULARITYENV_HF_TOKEN="${HF_TOKEN}"

run_in_container() {
    srun --gres=gpu:1 -p h200.141gb,h100.80gb --mem=100G --cpus-per-task=12 singularity exec --nv --cleanenv \
        --bind "${PROJECT_DIR}:${PROJECT_DIR}" \
        --pwd "${PROJECT_DIR}" \
        "${CONTAINER}" "$@"
}

# eval_forgetting.py appends KT_PYDEPS to sys.path. Bare dependency probes must
# do the same explicitly.
probe_module() {
    local module="$1"
    run_in_container python3 -c \
        "import os,sys; p=os.environ.get('KT_PYDEPS'); p and sys.path.append(p); import ${module}; print('${module} import OK')"
}

ensure_eval_dependency() {
    local module="$1"
    local package="$2"
    if ! probe_module "${module}" >/dev/null 2>&1; then
        echo "Installing missing evaluation dependency: ${package}"
        run_in_container python3 -m pip install \
            --disable-pip-version-check --no-cache-dir --upgrade \
            --target "${PYDEPS_ROOT}" "${package}"
    fi
    probe_module "${module}"
}

echo "=== CUDA preflight ==="
run_in_container python3 - <<'PY'
import torch, transformers
print("PyTorch:", torch.__version__)
print("Compiled CUDA:", torch.version.cuda)
print("Transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable inside container")
print("GPU:", torch.cuda.get_device_name(0))
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB")
PY

echo "=== Evaluation dependencies ==="
ensure_eval_dependency bddl bddl
ensure_eval_dependency easydict easydict
ensure_eval_dependency matplotlib matplotlib
ensure_eval_dependency gym "gym==0.26.2"

echo "=== LIBERO import preflight ==="
run_in_container python3 "${EVAL_SCRIPT}" --check-imports

if [[ "${RUN_SMOKE_TEST}" == "1" ]]; then
    echo "=== LIBERO rendering smoke test ==="
    run_in_container python3 "${EVAL_SCRIPT}" --smoke-test \
        --suites libero_spatial --sanity-dir "${SANITY_DIR}" --seed "${SEED}"
fi

DEFAULT_STAGE_DIRS="${RUN_ROOT}/stage0_spatial,${RUN_ROOT}/stage1_object,${RUN_ROOT}/stage2_goal,${RUN_ROOT}/stage3_long"
IFS=',' read -r -a STAGES <<< "${STAGE_DIRS:-${DEFAULT_STAGE_DIRS}}"
ALL_SUITES=(libero_spatial libero_object libero_goal libero_long)

if [[ "${RUN_BASELINE}" == "1" ]]; then
    # Evaluated on every suite any stage below will cover, so the baseline
    # column lines up against the full stage0..stageN comparison, not just
    # whichever suite stage0 happens to train on first.
    baseline_suites=("${ALL_SUITES[@]:0:${#STAGES[@]}}")
    args=(--checkpoint "${BASELINE_CHECKPOINT}" --stage-name "${BASELINE_STAGE_NAME}"
          --suites "${baseline_suites[@]}" --episodes-per-task "${EPISODES_PER_TASK}"
          --results-json "${RESULTS_JSON}" --save-videos "${SAVE_VIDEOS}"
          --video-dir "${VIDEO_DIR}" --sanity-dir "${SANITY_DIR}" --seed "${SEED}")
    [[ -n "${TASK_IDS}" ]] && args+=(--task-ids "${TASK_IDS}")
    [[ -n "${MAX_STEPS}" ]] && args+=(--max-steps "${MAX_STEPS}")
    [[ "${NO_FLIP}" == "1" ]] && args+=(--no-flip)

    echo "=== Evaluating ${BASELINE_STAGE_NAME} (out-of-the-box): ${baseline_suites[*]} ==="
    run_in_container python3 "${EVAL_SCRIPT}" "${args[@]}" \
        2>&1 | tee "${RESULT_ROOT}/${BASELINE_STAGE_NAME}_eval.log"
fi

for stage_idx in "${!STAGES[@]}"; do
    checkpoint="${STAGES[$stage_idx]}"
    [[ -d "${checkpoint}" ]] || { echo "ERROR: missing checkpoint: ${checkpoint}" >&2; exit 3; }
    suffix="$(basename "${checkpoint}" | sed -E 's/^stage[0-9]+_//')"
    stage_name="stage${stage_idx}_${suffix}"
    suites=("${ALL_SUITES[@]:0:$((stage_idx + 1))}")

    args=(--checkpoint "${checkpoint}" --stage-name "${stage_name}"
          --suites "${suites[@]}" --episodes-per-task "${EPISODES_PER_TASK}"
          --results-json "${RESULTS_JSON}" --save-videos "${SAVE_VIDEOS}"
          --video-dir "${VIDEO_DIR}" --sanity-dir "${SANITY_DIR}" --seed "${SEED}")
    [[ -n "${TASK_IDS}" ]] && args+=(--task-ids "${TASK_IDS}")
    [[ -n "${MAX_STEPS}" ]] && args+=(--max-steps "${MAX_STEPS}")
    [[ "${NO_FLIP}" == "1" ]] && args+=(--no-flip)

    echo "=== Evaluating ${stage_name}: ${suites[*]} ==="
    run_in_container python3 "${EVAL_SCRIPT}" "${args[@]}" \
        2>&1 | tee "${RESULT_ROOT}/${stage_name}_eval.log"
done

run_in_container python3 "${EVAL_SCRIPT}" --show-results "${RESULTS_JSON}"
echo "Evaluation completed: ${RESULTS_JSON}"
