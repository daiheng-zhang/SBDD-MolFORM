#!/usr/bin/env bash
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

echo "Starting standard training at $(date)"
python scripts/train_diffusion.py \
    configs/training_standard.yml \
    --tag standard_training \
    --name "Standard Training"

echo "Standard training completed at $(date)"
