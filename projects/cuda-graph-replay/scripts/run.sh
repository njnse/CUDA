#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
out="${RESULTS_DIR:-$root/results/local/run}"
mkdir -p "$out"
git -C "$root" rev-parse HEAD > "$out/source-commit.txt"
(cd "$root"; sha256sum src/main.cu include/reference.hpp scripts/build.sh) > "$out/source-sha256.txt"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > "$out/device.csv"
nvcc --version > "$out/compiler.txt"
bash "$root/scripts/build.sh" > "$out/build.txt" 2>&1
python3 -m unittest discover -s "$root/tests" -p 'test_*.py' -v > "$out/analysis-tests.txt" 2>&1
"$root/build/graph_lab" --verify > "$out/verify.txt" 2>&1
"$root/build/graph_lab" --benchmark > "$out/benchmark.csv" 2> "$out/setup-and-checks.txt"
for tool in memcheck racecheck initcheck synccheck; do
    compute-sanitizer --tool "$tool" --error-exitcode 1 "$root/build/graph_lab" --verify > "$out/$tool.txt" 2>&1
done
python3 "$root/scripts/analyze.py" "$out/benchmark.csv" > "$out/summary.csv"
echo "All checks passed"
