#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$root/build"
"${CXX:-g++}" -std=c++17 -O2 -Wall -Wextra -Werror -I"$root/include" "$root/tests/reference_test.cpp" -o "$root/build/reference_test"
"$root/build/reference_test"
"${CUDACXX:-nvcc}" -std=c++17 -O3 -lineinfo -arch="${CUDA_ARCH:-sm_89}" -I"$root/include" "$root/src/transpose.cu" -o "$root/build/transpose_lab"
