import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

EXE = 'build/gemm'
MODES = ('fp32', 'tf32')


def run(mode, command, shape, iterations):
    raw = subprocess.check_output(
        [EXE, mode, command, *map(str, shape), str(iterations)], text=True)
    ms, error, rel_l2, probes, version = raw.strip().split(',')
    numeric = [float(ms), float(error), float(rel_l2)]
    if not all(math.isfinite(value) for value in numeric) or numeric[0] <= 0:
        raise ValueError('invalid timing or error')
    return dict(zip(('ms', 'max_abs', 'relative_l2', 'probes', 'cublas'), (*numeric, int(probes), int(version))))


def verify():
    for args in (('invalid', 'verify', '1', '1', '1', '1'),
                 ('tf32', 'verify', '0', '1', '1', '1'),
                 ('fp32', 'verify', '257', '1', '1', '1')):
        checked = subprocess.run([EXE, *args], text=True, capture_output=True)
        if checked.returncode != 1 or not checked.stderr.strip():
            raise ValueError('invalid arguments accepted')
    for mode in MODES:
        for shape in ((1, 1, 1), (31, 29, 33), (129, 127, 65)):
            print(json.dumps(dict(mode=mode, shape=shape, **run(mode, 'verify', shape, 1))))
    print('PASS six full-matrix oracle comparisons and three invalid inputs')


def benchmark(out):
    rows, summary = [], []
    for shape in ((256, 256, 256), (2048, 2048, 2048), (4096, 4096, 4096)):
        for pair in range(5):
            for mode in (MODES if pair % 2 == 0 else MODES[::-1]):
                row = run(mode, 'sample', shape, 6)
                rows.append(dict(m=shape[0], n=shape[1], k=shape[2], pair=pair, mode=mode,
                                 tflops=2 * math.prod(shape) / row['ms'] / 1e9, **row))
        for mode in MODES:
            group = [row for row in rows if row['m'] == shape[0] and row['mode'] == mode]
            timing = [row['ms'] for row in group]
            summary.append(dict(shape=shape, mode=mode, median_ms=statistics.median(timing),
                                min_ms=min(timing), max_ms=max(timing),
                                median_tflops=statistics.median(row['tflops'] for row in group),
                                max_relative_l2=max(row['relative_l2'] for row in group)))
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
