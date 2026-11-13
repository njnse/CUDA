#!/usr/bin/env bash
set -euo pipefail
: "${RESULTS_DIR:?Set RESULTS_DIR to an absolute path}"
case "$RESULTS_DIR" in /*) ;; *) exit 2;; esac
[[ ${CUDA_ARCH:-sm_100} == sm_100 ]]
[[ -n ${SLURM_JOB_ID:-} ]]
project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$RESULTS_DIR"
if [[ -n $(find "$RESULTS_DIR" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
    echo 'Use an empty results directory' >&2
    exit 2
fi
cd "$project"
mkdir -p build
export CUDA_CACHE_DISABLE=1 CUDA_MODULE_LOADING=EAGER
unset CUDA_FORCE_PTX_JIT CUDA_FORCE_JIT CUDA_DISABLE_PTX_JIT CUDA_DISABLE_JIT
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$RESULTS_DIR/device.txt"
[[ $(wc -l < "$RESULTS_DIR/device.txt") -eq 1 ]]
grep -q 'NVIDIA B200' "$RESULTS_DIR/device.txt"
nvcc --version > "$RESULTS_DIR/nvcc.txt"
sha256sum kernel.cu probe.py benchmark.py test_probe.py scripts/run.sh > "$RESULTS_DIR/source-sha256.txt"
python3 -m unittest -v test_probe > "$RESULTS_DIR/cpu.txt" 2>&1
{
    nvcc -O3 -lineinfo -arch=sm_100 --cubin kernel.cu -o build/kernel.cubin
    nvcc -O3 -lineinfo -arch=compute_100 --ptx kernel.cu -o build/kernel.ptx
} > "$RESULTS_DIR/build.txt" 2>&1
sha256sum build/kernel.cubin build/kernel.ptx > "$RESULTS_DIR/image-sha256.txt"
for image in cubin ptx; do
    for n in 1 255 256 257 65537; do
        python3 probe.py "build/kernel.$image" "$n" --negative
    done
done > "$RESULTS_DIR/correctness.txt"
python3 benchmark.py "$RESULTS_DIR"
for tool in memcheck initcheck racecheck synccheck; do
    for image in cubin ptx; do
        compute-sanitizer --tool "$tool" --error-exitcode 7 python3 probe.py "build/kernel.$image" 257 \
            > "$RESULTS_DIR/$tool-$image.txt" 2>&1
    done
done
for image in cubin ptx; do
    ncu --csv --metrics gpu__time_duration.sum --launch-count 1 python3 probe.py "build/kernel.$image" 65537 \
        > "$RESULTS_DIR/ncu-$image.csv" 2> "$RESULTS_DIR/ncu-$image.stderr"
    grep -q 'gpu__time_duration.sum' "$RESULTS_DIR/ncu-$image.csv"
done
cat > "$RESULTS_DIR/checks.json" <<'JSON'
{"correctness":"passed","benchmark":"passed","sanitizers":{"memcheck":"passed","initcheck":"passed","racecheck":"passed","synccheck":"passed"},"profiler":"passed"}
JSON
