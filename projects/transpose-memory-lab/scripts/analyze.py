"""Summarize real CUDA-event samples; never supply synthetic fallback timings."""
import csv
import math
import statistics
import sys
from collections import defaultdict


def summarize(paths):
    groups = defaultdict(list)
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                value = float(row["ms_per_launch"])
                if not math.isfinite(value) or value <= 0:
                    raise ValueError("timings must be finite and positive")
                groups[(row["variant"], int(row["rows"]), int(row["cols"]))].append(value)
    if not groups:
        raise ValueError("no benchmark samples")
    return [(v, r, c, len(xs), statistics.median(xs), min(xs), max(xs))
            for (v, r, c), xs in sorted(groups.items())]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: analyze.py measured.csv [measured.csv ...]")
    writer = csv.writer(sys.stdout)
    writer.writerow(["variant", "rows", "cols", "samples", "median_ms", "min_ms", "max_ms"])
    writer.writerows(summarize(sys.argv[1:]))
