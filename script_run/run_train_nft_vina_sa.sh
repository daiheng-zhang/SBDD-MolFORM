#!/bin/bash
#SBATCH --job-name=kdd_molform_nft_vina_sa
#SBATCH --output=log/kdd_molform_nft_vina_sa_%j.log
#SBATCH --error=log/kdd_molform_nft_vina_sa_%j.log
#SBATCH --partition=ghx4
#SBATCH --account=bdrw-dtai-gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=2
#SBATCH --mem=256G
#SBATCH --time=48:00:00

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Running on host: $(hostname)"
echo "Start time: $(date)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
echo "Working directory: $(pwd)"

CONDA_BASE="${CONDA_BASE:-$(conda info --base)}"
CONDA_ENV="${CONDA_ENV:-MolFORM}"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
echo "Activated environment: $CONDA_DEFAULT_ENV"

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export WANDB_DIR="${WANDB_DIR:-./wandb}"

CKPT="${CKPT:-./ckpt/molform_base_model.pt}"
CONFIG="${CONFIG:-configs/training_nft_vina_sa.yml}"
LOGDIR="${LOGDIR:-./logs_diffusion}"
RUN_NAME="${RUN_NAME:-KDD-Molform-NFT-VinaSA}"
TAG="${TAG:-nft_vina_sa}"
RESET_ITERATION="${RESET_ITERATION:-1}"

mkdir -p log "${LOGDIR}" "${WANDB_DIR}"

if [ ! -f "${CKPT}" ]; then
  echo "Checkpoint not found: ${CKPT}"
  exit 1
fi

echo "Starting Vina-SA NFT training from ckpt: ${CKPT}"
echo "Start training time: $(date)"

CMD=(python -m scripts.train_diffusion "${CONFIG}" --train_report_iter 10 --tag "${TAG}" --logdir "${LOGDIR}" --name "${RUN_NAME}" --checkpoint "${CKPT}" --no_optimizer_state --non_strict_load)
if [ "${RESET_ITERATION}" = "1" ]; then
  CMD+=(--reset_iteration)
fi
"${CMD[@]}"

echo "Job completed at $(date)"
