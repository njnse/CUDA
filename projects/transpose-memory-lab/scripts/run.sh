#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
results="${RESULTS_DIR:-$root/results/local/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$results"
git -C "$root" rev-parse HEAD > "$results/source-commit.txt"
git -C "$root" status --porcelain > "$results/source-status.txt"
sha256sum "$root/src/transpose.cu" "$root/include/reference.hpp" > "$results/source-sha256.txt"
nvidia-smi > "$results/nvidia-smi.txt"
nvcc --version > "$results/nvcc.txt"
bash "$root/scripts/build.sh" > "$results/build.txt" 2>&1
"$root/build/transpose_lab" --verify > "$results/verify.txt" 2>&1
for shape in '31 33' '1024 1024' '4096 4096' '4093 2051'; do
    read -r rows cols <<< "$shape"
    "$root/build/transpose_lab" --benchmark "$rows" "$cols" 15 > "$results/${rows}x${cols}.csv" 2> "$results/${rows}x${cols}.stderr"
done
for tool in memcheck racecheck synccheck initcheck; do
    compute-sanitizer --tool "$tool" --error-exitcode 1 "$root/build/transpose_lab" --verify > "$results/$tool.txt" 2>&1
done
python3 "$root/scripts/analyze.py" "$results"/*.csv > "$results/summary.csv"
printf 'Results: %s\n' "$results"
