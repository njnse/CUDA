import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys


def benchmark(destination):
    rows = []
    for pair in range(9):
        images = ('cubin', 'ptx') if pair % 2 == 0 else ('ptx', 'cubin')
        for order, image in enumerate(images):
            output = subprocess.check_output(
                [sys.executable, 'probe.py', f'build/kernel.{image}', '65537'], text=True)
            row = json.loads(output)
            rows.append(dict(pair=pair, order=order, image=image, **row))
    with (destination / 'raw.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for image in ('cubin', 'ptx'):
        observations = [row for row in rows if row['image'] == image]
        summary[image] = {}
        for metric in ('module_ns', 'lookup_ns', 'launch_sync_ns'):
            samples = [row[metric] for row in observations]
            summary[image][metric] = dict(median=statistics.median(samples),
                                          min=min(samples), max=max(samples))
    (destination / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    benchmark(Path(sys.argv[1]))
