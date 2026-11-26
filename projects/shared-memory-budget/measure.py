import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

EXE = 'build/budget'
BUDGETS = (256, 32768, 65536, 114688, 196608)


def run(budget, n, repeats):
    result = subprocess.check_output([EXE, str(budget), str(n), str(repeats)], text=True)
    ms, checksum, blocks = result.strip().split(',')
    ms = float(ms)
    if not math.isfinite(ms) or ms <= 0 or int(blocks) < 1:
        raise ValueError('invalid timing or residency')
    return ms, int(checksum), int(blocks)


def verify():
    for args in (('oops',), ('2', '1', '1'), ('256', '0', '1'), ('256', '1', '31')):
        result = subprocess.run([EXE, *args], capture_output=True, text=True)
        if result.returncode != 1 or 'invalid' not in result.stderr:
            raise ValueError('invalid input was accepted')
    print(subprocess.check_output([EXE, 'reject'], text=True).strip())
    for budget in (BUDGETS[0], BUDGETS[-1]):
        for n in (1, 255, 256, 257, 65537):
            run(budget, n, 1)
    print('PASS 10 complete oracle and guard cases and four invalid CLI inputs')


def benchmark(out):
    rows, summary = [], []
    n = 4194307
    for sample in range(7):
        offset = sample % len(BUDGETS)
        for budget in BUDGETS[offset:] + BUDGETS[:offset]:
            ms, checksum, blocks = run(budget, n, 6)
            rows.append(dict(n=n, sample=sample, dynamic_bytes=budget, ms=ms,
                             checksum=checksum, resident_blocks_per_sm=blocks))
    for budget in BUDGETS:
        group = [row for row in rows if row['dynamic_bytes'] == budget]
        values = [row['ms'] for row in group]
        summary.append(dict(dynamic_bytes=budget, median_ms=statistics.median(values),
                            min_ms=min(values), max_ms=max(values),
                            resident_blocks_per_sm=group[0]['resident_blocks_per_sm']))
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
