import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

EXE = 'build/compact'
MODES = ('direct', 'warp')


def run(mode, n, threshold, repeats):
    raw = subprocess.check_output([EXE, mode, str(n), str(threshold), str(repeats)], text=True)
    elapsed, count = raw.strip().split(',')
    ms = float(elapsed)
    if not math.isfinite(ms) or ms <= 0:
        raise ValueError('invalid CUDA timing')
    return ms, int(count)


def verify():
    for mode in MODES:
        for n in (1, 31, 32, 33, 257, 65537):
            for threshold in (0, 17, 256):
                run(mode, n, threshold, 1)
    for args in (('bad', '1', '0', '1'), ('warp', '0', '1', '1'), ('direct', '1', '257', '1')):
        result = subprocess.run([EXE, *args], capture_output=True, text=True)
        if result.returncode != 1 or 'invalid' not in result.stderr:
            raise ValueError('invalid command accepted')
    print('PASS 36 CPU/GPU exact membership comparisons and three invalid inputs')


def benchmark(out):
    rows = []
    summary = []
    for n in (65537, 16777216):
        for threshold in (4, 64, 256):
            for pair in range(5):
                order = MODES if pair % 2 == 0 else MODES[::-1]
                for mode in order:
                    ms, count = run(mode, n, threshold, 3)
                    rows.append(dict(n=n, threshold=threshold, count=count, pair=pair, mode=mode, ms=ms))
            for mode in MODES:
                data = [r['ms'] for r in rows if (r['n'], r['threshold'], r['mode']) == (n, threshold, mode)]
                summary.append(dict(n=n, threshold=threshold, mode=mode,
                                    median_ms=statistics.median(data), min_ms=min(data), max_ms=max(data)))
    with (out / 'raw.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    if sys.argv[1] == 'verify':
        verify()
    else:
        benchmark(Path(sys.argv[1]))
