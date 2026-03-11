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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

CONFIG_FILE="${CONFIG_FILE:-configs/sampling_kdd_confidence_49000.yml}"
RESULT_PATH="${RESULT_PATH:-./outputs_sampling/sample_default}"
START_IDX="${START_IDX:-0}"
DATA_PATH="${DATA_PATH:-}"
SPLIT_PATH="${SPLIT_PATH:-}"

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "[ERROR] CONFIG_FILE not found: ${CONFIG_FILE}" >&2
  exit 1
fi

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
GPU_COUNT="${#GPU_IDS[@]}"
if [[ "${GPU_COUNT}" -lt 1 ]]; then
  GPU_IDS=(0)
  GPU_COUNT=1
fi

if [[ -n "${NODE_ALL:-}" ]]; then
  NODE_ALL="${NODE_ALL}"
else
  NODE_ALL="${GPU_COUNT}"
fi

mkdir -p "${RESULT_PATH}"

echo "Start time: $(date)"
echo "Project root: ${PROJECT_ROOT}"
echo "Conda env: ${CONDA_ENV}"
echo "CONFIG_FILE: ${CONFIG_FILE}"
echo "RESULT_PATH: ${RESULT_PATH}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "NODE_ALL: ${NODE_ALL}"
echo "START_IDX: ${START_IDX}"
echo "DATA_PATH: ${DATA_PATH:-<from config/checkpoint>}"
echo "SPLIT_PATH: ${SPLIT_PATH:-<from config/checkpoint>}"

pids=()
for ((NODE_RANK=0; NODE_RANK<NODE_ALL; NODE_RANK++)); do
  DEVICE="${GPU_IDS[$((NODE_RANK % GPU_COUNT))]}"
  echo "Launching NODE_RANK=${NODE_RANK} on GPU ${DEVICE}"
  CUDA_VISIBLE_DEVICES="${DEVICE}" \
    bash scripts/batch_sample_diffusion.sh \
      "${CONFIG_FILE}" \
      "${RESULT_PATH}" \
      "${NODE_ALL}" \
      "${NODE_RANK}" \
      "${START_IDX}" &
  pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    rc=1
  fi
done

echo "End time: $(date)"
exit "${rc}"
