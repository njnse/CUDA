#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build
"${CXX:-g++}" -std=c++17 -O3 -Wall -Wextra -Werror -Iinclude tests/reference_test.cpp -o build/reference_test
./build/reference_test
python3 -B -m unittest discover -s tests -p 'test_*.py' -v
