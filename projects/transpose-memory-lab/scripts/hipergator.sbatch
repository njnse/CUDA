#!/usr/bin/env bash
#SBATCH --job-name=transpose-memory-lab
#SBATCH --account=eel4930
#SBATCH --qos=eel4930
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --constraint=l4
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=transpose-%j.log
set -euo pipefail
cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
module load "${CUDA_MODULE:-cuda/12.4.1}"
export CUDA_ARCH="${CUDA_ARCH:-sm_89}"
export RESULTS_DIR="$PWD/projects/transpose-memory-lab/results/local/slurm-$SLURM_JOB_ID"
bash projects/transpose-memory-lab/scripts/run.sh
