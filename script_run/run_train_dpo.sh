#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONDA_BASE="${CONDA_BASE:-$(conda info --base)}"
CONDA_ENV="${CONDA_ENV:-MolFORM}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DPO_DATA="${DPO_DATA:-./data/dpo_data/dpo_idx_sort_new.pkl}"

if [ ! -f "${DPO_DATA}" ]; then
  echo "DPO data not found: ${DPO_DATA}"
  exit 1
fi

echo "Starting DPO training at $(date)"
python scripts/train_diffusion.py \
  configs/training_dpo.yml \
  --tag dpo_training \
  --dpo_data "${DPO_DATA}" \
  --name "DPO Training"

echo "DPO training completed at $(date)"
