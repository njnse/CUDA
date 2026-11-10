#!/usr/bin/env bash
set -euo pipefail
: "${RESULTS_DIR:?Set absolute RESULTS_DIR}"
case "$RESULTS_DIR" in /*) ;; *) exit 2;; esac
project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$project/scripts/run.py"
