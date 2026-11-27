#!/usr/bin/env bash
set -euo pipefail
: "${RESULTS_DIR:?Set an absolute output path}"
[[ "$RESULTS_DIR" == /* && ${CUDA_ARCH:-sm_100} == sm_100 && -n ${SLURM_JOB_ID:-} ]]
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p "$RESULTS_DIR" build
[[ -z $(find "$RESULTS_DIR" -mindepth 1 -maxdepth 1 -print -quit) ]]
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$RESULTS_DIR/device.txt"
[[ $(wc -l < "$RESULTS_DIR/device.txt") -eq 1 ]]
grep -q '^NVIDIA B200,' "$RESULTS_DIR/device.txt"
nvcc --version > "$RESULTS_DIR/nvcc.txt"
sha256sum batch.cu measure.py scripts/run.sh > "$RESULTS_DIR/source-sha256.txt"
nvcc -std=c++17 -O3 -lineinfo -arch=sm_100 batch.cu -lcublas -o build/batch > "$RESULTS_DIR/build.txt" 2>&1
python3 measure.py verify > "$RESULTS_DIR/correctness.txt"
python3 measure.py "$RESULTS_DIR"
for mode in loop strided; do
    for tool in memcheck initcheck racecheck synccheck; do
        compute-sanitizer --tool "$tool" --error-exitcode 7 build/batch "$mode" verify 17 19 23 3 1 \
            > "$RESULTS_DIR/$tool-$mode.txt" 2>&1
    done
    ncu --csv --launch-count 1 --metrics gpu__time_duration.sum \
        build/batch "$mode" sample 32 32 32 64 1 > "$RESULTS_DIR/ncu-$mode.csv" 2> "$RESULTS_DIR/ncu-$mode.stderr"
    grep -q 'gpu__time_duration.sum' "$RESULTS_DIR/ncu-$mode.csv"
done
printf '%s\n' '{"correctness":"passed","benchmark":"passed","sanitizers":{"memcheck":"passed","initcheck":"passed","racecheck":"passed","synccheck":"passed"},"profiler":"passed"}' > "$RESULTS_DIR/checks.json"
