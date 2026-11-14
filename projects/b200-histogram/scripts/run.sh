#!/usr/bin/env bash
set -euo pipefail
: "${RESULTS_DIR:?Set RESULTS_DIR to an absolute path}"
case "$RESULTS_DIR" in /*) ;; *) exit 2;; esac
[[ ${CUDA_ARCH:-sm_100} == sm_100 && -n ${SLURM_JOB_ID:-} ]]
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p "$RESULTS_DIR" build
[[ -z $(find "$RESULTS_DIR" -mindepth 1 -maxdepth 1 -print -quit) ]]
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$RESULTS_DIR/device.txt"
[[ $(wc -l < "$RESULTS_DIR/device.txt") -eq 1 ]]
grep -q 'NVIDIA B200' "$RESULTS_DIR/device.txt"
nvcc --version > "$RESULTS_DIR/nvcc.txt"
sha256sum histogram.cu benchmark.py scripts/run.sh > "$RESULTS_DIR/source-sha256.txt"
python3 -m unittest -v benchmark > "$RESULTS_DIR/cpu.txt" 2>&1
nvcc -std=c++17 -O3 -lineinfo -arch=sm_100 histogram.cu -o build/histogram > "$RESULTS_DIR/build.txt" 2>&1
python3 benchmark.py verify > "$RESULTS_DIR/correctness.txt"
python3 benchmark.py "$RESULTS_DIR"
for mode in global block; do
    for tool in memcheck initcheck racecheck synccheck; do
        compute-sanitizer --tool "$tool" --error-exitcode 7 build/histogram "$mode" hot 257 1 \
            > "$RESULTS_DIR/$tool-$mode.txt" 2>&1
    done
    for distribution in uniform hot; do
        ncu --csv --metrics gpu__time_duration.sum --launch-skip 2 --launch-count 1 \
            build/histogram "$mode" "$distribution" 1048576 1 \
            > "$RESULTS_DIR/ncu-$mode-$distribution.csv" 2> "$RESULTS_DIR/ncu-$mode-$distribution.stderr"
        grep -q 'gpu__time_duration.sum' "$RESULTS_DIR/ncu-$mode-$distribution.csv"
    done
done
printf '%s\n' '{"correctness":"passed","benchmark":"passed","sanitizers":{"memcheck":"passed","initcheck":"passed","racecheck":"passed","synccheck":"passed"},"profiler":"passed"}' > "$RESULTS_DIR/checks.json"
