#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm-%A_%a.out

# Usage:
#      Generate configs:                python configs/generate_sweep.py
#   A. Submit a range of the manifest:  sbatch --array=0-10%2 run.sbatch
#   B. Single-config submit:            CONFIG=configs/sweep/foo.json sbatch run.sbatch

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" configs/sweep/_manifest.txt)
fi
: "${CONFIG:?CONFIG must be set (either via array task id or via env)}"
echo "CONFIG=$CONFIG"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-?} ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-?}"

torchrun --standalone --nnodes=1 --nproc_per_node=4 src/train.py --config "$CONFIG"
