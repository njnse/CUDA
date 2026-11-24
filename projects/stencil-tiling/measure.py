import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

EXE = 'build/stencil'
MODES = ('direct', 'tiled')


def run(mode, w, h, repeats):
    result = subprocess.check_output([EXE, mode, str(w), str(h), str(repeats)], text=True)
    ms, checksum = result.strip().split(',')
    ms = float(ms)
    if not math.isfinite(ms) or ms <= 0:
        raise ValueError('invalid timing')
    return ms, int(checksum)


def verify():
    for args in (('bad', '2', '2', '1'), ('direct', '0', '1', '1'), ('tiled', '1', '1', '31')):
        result = subprocess.run([EXE, *args], capture_output=True, text=True)
        if result.returncode != 1 or 'invalid' not in result.stderr:
            raise ValueError('invalid input accepted')
    for w, h in ((1, 1), (1, 19), (67, 1), (32, 8), (33, 9), (127, 65)):
        for mode in MODES:
            run(mode, w, h, 1)
    print('PASS 12 full-output oracle and guard cases and three invalid-input checks')


def benchmark(out):
    rows, summary = [], []
    for w, h in ((257, 259), (2049, 2051), (8193, 8193)):
        for pair in range(5):
            for mode in (MODES if pair % 2 == 0 else MODES[::-1]):
                ms, checksum = run(mode, w, h, 6)
                rows.append(dict(width=w, height=h, pair=pair, mode=mode, ms=ms, checksum=checksum))
        for mode in MODES:
            v = [r['ms'] for r in rows if (r['width'], r['height'], r['mode']) == (w, h, mode)]
            summary.append(dict(width=w, height=h, mode=mode, median_ms=statistics.median(v),
                                min_ms=min(v), max_ms=max(v)))
    with (out / 'raw.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    if sys.argv[1] == 'verify':
        verify()
    else:
        benchmark(Path(sys.argv[1]))
