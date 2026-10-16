#!/usr/bin/env bash
#SBATCH --job-name=transpose-profile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --constraint=l4
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=transpose-profile-%j.log
set -euo pipefail
cd "${REPO_DIR:-${SLURM_SUBMIT_DIR:?}}"
if [[ -n "${CUDA_MODULE:-}" ]]; then module load "$CUDA_MODULE"; fi
root="$PWD/projects/transpose-memory-lab"
r="$root/results/local/profile-$SLURM_JOB_ID"
mkdir -p "$r"
git rev-parse HEAD > "$r/source-commit.txt"
sha256sum "$root/src/transpose.cu" "$root/build/transpose_lab" > "$r/source-and-binary-sha256.txt"
nvidia-smi > "$r/nvidia-smi.txt"
ncu --version > "$r/ncu-version.txt"
set +e
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum --kernel-name 'regex:.*tiled_kernel.*' --launch-count 2 --csv --log-file "$r/ncu-metrics.csv" "$root/build/transpose_lab" --benchmark 1024 1024 1 > "$r/application.txt" 2>&1
rc=$?
set -e
printf '%s\n' "$rc" > "$r/profiler-exit-code.txt"
exit "$rc"
