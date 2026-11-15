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
sha256sum fusion.cu measure.py scripts/run.sh > "$RESULTS_DIR/source-sha256.txt"
nvcc -std=c++17 -O3 -lineinfo -arch=sm_100 fusion.cu -o build/fusion > "$RESULTS_DIR/build.txt" 2>&1
python3 measure.py verify > "$RESULTS_DIR/correctness.txt"
python3 measure.py "$RESULTS_DIR"
for mode in split fused; do
    for tool in memcheck initcheck racecheck synccheck; do
        compute-sanitizer --tool "$tool" --error-exitcode 7 build/fusion "$mode" 257 1 \
            > "$RESULTS_DIR/$tool-$mode.txt" 2>&1
    done
    if [[ $mode == split ]]; then skip=6; count=2; else skip=3; count=1; fi
    ncu --csv --launch-skip "$skip" --launch-count "$count" \
        --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum \
        build/fusion "$mode" 67108864 1 > "$RESULTS_DIR/ncu-$mode.csv" 2> "$RESULTS_DIR/ncu-$mode.stderr"
    grep -q 'dram__bytes_read.sum' "$RESULTS_DIR/ncu-$mode.csv"
done
printf '%s\n' '{"correctness":"passed","benchmark":"passed","sanitizers":{"memcheck":"passed","initcheck":"passed","racecheck":"passed","synccheck":"passed"},"profiler":"passed"}' > "$RESULTS_DIR/checks.json"
