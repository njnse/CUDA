import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

EXE = 'build/batch'
MODES = ('loop', 'strided')
CASES = ((16, 1), (16, 64), (16, 1024), (32, 64), (32, 1024), (64, 64), (64, 1024))


def run(mode, command, shape, repeats):
    raw = subprocess.check_output([EXE, mode, command, *map(str, shape), str(repeats)], text=True)
    ms, checksum, probes, version = raw.strip().split(',')
    elapsed = float(ms)
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError('invalid elapsed time')
    return dict(ms=elapsed, checksum=int(checksum), probes=int(probes), cublas=int(version))


def verify():
    for args in (('bad', 'verify', 1, 1, 1, 1, 1), ('loop', 'verify', 0, 5, 5, 1, 1),
                 ('strided', 'verify', 2, 2, 2, 0, 1), ('strided', 'verify', 1, 1, 1, 9, 1)):
        result = subprocess.run([EXE, *map(str, args)], capture_output=True, text=True)
        if result.returncode != 1 or 'invalid' not in result.stderr and 'must be' not in result.stderr:
            raise ValueError('invalid call accepted')
    for shape in ((1, 1, 1, 1), (3, 5, 7, 3), (17, 19, 23, 5), (32, 32, 32, 8)):
        for mode in MODES:
            print(json.dumps(dict(mode=mode, shape=shape, **run(mode, 'verify', shape, 1))))
    print('PASS eight complete integer-oracle and guard comparisons, four invalid-input checks')


def benchmark(out):
    rows, summary = [], []
    for n, batch in CASES:
        for pair in range(5):
            for mode in (MODES if pair % 2 == 0 else MODES[::-1]):
                sample = run(mode, 'sample', (n, n, n, batch), 3)
                rows.append(dict(n=n, batch=batch, pair=pair, mode=mode, **sample))
        for mode in MODES:
            selected = [r['ms'] for r in rows if (r['n'], r['batch'], r['mode']) == (n, batch, mode)]
            summary.append(dict(n=n, batch=batch, mode=mode, median_ms=statistics.median(selected),
                                min_ms=min(selected), max_ms=max(selected)))
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
