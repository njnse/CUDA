import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys

EXE = 'build/fusion'
MODES = ('split', 'fused')


def run(mode, size, repeats):
    output = subprocess.check_output([EXE, mode, str(size), str(repeats)], text=True)
    elapsed, checksum = output.strip().split(',')
    return float(elapsed), int(checksum)


def verify():
    for mode in MODES:
        for size in (1, 255, 256, 257, 65537):
            run(mode, size, 1)
    for args in (('invalid', '1', '1'), ('split', '0', '1'), ('fused', '1', '51')):
        result = subprocess.run([EXE, *args], capture_output=True, text=True)
        if result.returncode != 1 or 'invalid' not in result.stderr:
            raise RuntimeError('invalid input accepted')
    print('PASS 10 full-output oracle comparisons, 10 guard checks and 3 invalid inputs')


def measure(directory):
    rows = []
    for size in (65537, 4194304, 67108864):
        for pair in range(5):
            for mode in MODES[::1 if pair % 2 == 0 else -1]:
                elapsed, checksum = run(mode, size, 6)
                if not elapsed > 0:
                    raise ValueError('nonpositive timing')
                rows.append(dict(n=size, pair=pair, mode=mode, ms=elapsed, checksum=checksum))
    with (directory / 'raw.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for size in sorted({row['n'] for row in rows}):
        for mode in MODES:
            values = [r['ms'] for r in rows if r['n'] == size and r['mode'] == mode]
            summary.append(dict(n=size, mode=mode, median_ms=statistics.median(values),
                                min_ms=min(values), max_ms=max(values)))
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    if sys.argv[1] == 'verify':
        verify()
    else:
        measure(Path(sys.argv[1]))
