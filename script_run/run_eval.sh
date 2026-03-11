#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONDA_BASE="${CONDA_BASE:-$(conda info --base)}"
CONDA_ENV="${CONDA_ENV:-MolFORM}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

RESULT_PATH="${RESULT_PATH:-./outputs_sampling/sample_default}"
PROTEIN_ROOT="${PROTEIN_ROOT:-./data/test_set}"
DOCKING_MODE="${DOCKING_MODE:-vina_score}"
EXHAUSTIVENESS="${EXHAUSTIVENESS:-16}"
MULTIPROCESS="${MULTIPROCESS:-True}"
SAVE_PICKLE="${SAVE_PICKLE:-True}"
SAVE="${SAVE:-True}"
EVAL_NUM_EXAMPLES="${EVAL_NUM_EXAMPLES:-}"
ATOM_ENC_MODE="${ATOM_ENC_MODE:-add_aromatic}"

if [[ ! -d "${RESULT_PATH}" ]]; then
  echo "[ERROR] RESULT_PATH does not exist: ${RESULT_PATH}" >&2
  exit 1
fi

echo "Start time: $(date)"
echo "Project root: ${PROJECT_ROOT}"
echo "Conda env: ${CONDA_ENV}"
echo "RESULT_PATH: ${RESULT_PATH}"
echo "PROTEIN_ROOT: ${PROTEIN_ROOT}"
echo "DOCKING_MODE: ${DOCKING_MODE}"
echo "EXHAUSTIVENESS: ${EXHAUSTIVENESS}"
echo "MULTIPROCESS: ${MULTIPROCESS}"
echo "SAVE_PICKLE: ${SAVE_PICKLE}"
echo "SAVE: ${SAVE}"
echo "ATOM_ENC_MODE: ${ATOM_ENC_MODE}"

cmd=(
  python -m scripts.evaluate_diffusion_multiprocess
  "${RESULT_PATH}"
  --protein_root "${PROTEIN_ROOT}"
  --docking_mode "${DOCKING_MODE}"
  --exhaustiveness "${EXHAUSTIVENESS}"
  --multiprocess "${MULTIPROCESS}"
  --save_pickle "${SAVE_PICKLE}"
  --save "${SAVE}"
  --atom_enc_mode "${ATOM_ENC_MODE}"
)

if [[ -n "${EVAL_NUM_EXAMPLES}" ]]; then
  cmd+=(--eval_num_examples "${EVAL_NUM_EXAMPLES}")
fi

"${cmd[@]}"

echo "End time: $(date)"
