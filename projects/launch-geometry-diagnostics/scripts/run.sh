#!/usr/bin/env bash
set -euo pipefail
: "${RESULTS_DIR:?Set RESULTS_DIR to an absolute output directory}"
case "$RESULTS_DIR" in /*) ;; *) echo 'RESULTS_DIR must be absolute' >&2; exit 2;; esac
project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$project/scripts/run_worker.py"
