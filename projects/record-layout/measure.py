import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

EXE = 'build/project'
LAYOUTS = ('rows', 'columns')
SELECTIONS = ('one', 'all')


def run(layout, selection, n, repeats):
    output = subprocess.check_output([EXE, layout, selection, str(n), str(repeats)], text=True)
    ms, checksum = output.strip().split(',')
    latency = float(ms)
    if not math.isfinite(latency) or latency <= 0:
        raise ValueError('invalid timing')
    return latency, int(checksum)


def verify():
    for args in (('bad', 'one', '1', '1'), ('rows', 'one', '0', '1'),
                 ('columns', 'all', '1', '31')):
        result = subprocess.run([EXE, *args], capture_output=True, text=True)
        if result.returncode != 1 or 'invalid' not in result.stderr:
            raise ValueError('invalid input was accepted')
    for n in (1, 31, 257, 4099):
        for selection in SELECTIONS:
            for layout in LAYOUTS:
                run(layout, selection, n, 1)
    print('PASS 16 full-output oracle and guard checks and 3 invalid-input checks')


def benchmark(out):
    rows, summary = [], []
    for n in (4099, 1048579, 16777219):
        for selection in SELECTIONS:
            for pair in range(5):
                for layout in (LAYOUTS if pair % 2 == 0 else LAYOUTS[::-1]):
                    ms, checksum = run(layout, selection, n, 6)
                    rows.append(dict(n=n, selection=selection, pair=pair, layout=layout,
                                     ms=ms, checksum=checksum))
            for layout in LAYOUTS:
                data = [x['ms'] for x in rows
                        if (x['n'], x['selection'], x['layout']) == (n, selection, layout)]
                summary.append(dict(n=n, selection=selection, layout=layout,
                                    median_ms=statistics.median(data), min_ms=min(data), max_ms=max(data)))
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
