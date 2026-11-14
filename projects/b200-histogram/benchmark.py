import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys
import unittest
from collections import Counter

EXE = 'build/histogram'


def reference(n, distribution):
    state = 2027
    mask = 2**32 - 1
    counts = Counter()
    for _ in range(n):
        state = (state ^ (state << 13)) & mask
        state ^= state >> 17
        state = (state ^ (state << 5)) & mask
        counts[7 if distribution == 'hot' and state % 16 else state // 2**24] += 1
    return [counts[i] for i in range(256)]


def measure(mode, distribution, n, repetitions):
    output = subprocess.check_output(
        [EXE, mode, distribution, str(n), str(repetitions)], text=True)
    ms, *counts = output.strip().split(',')
    counts = list(map(int, counts))
    if len(counts) != 256 or any(value < 0 for value in counts) or sum(counts) != n:
        raise ValueError('malformed histogram')
    return float(ms), counts


def verify():
    for arguments in [('bad', 'uniform', '1', '1'), ('global', 'uniform', '-1', '1'),
                      ('block', 'hot', '1', '0')]:
        result = subprocess.run([EXE, *arguments], capture_output=True, text=True)
        if result.returncode != 1 or not any(
            marker in result.stderr for marker in ('unknown mode', 'out of range')
        ):
            raise ValueError(f'expected argument error: {arguments}')
    for distribution in ('uniform', 'hot'):
        for n in (0, 1, 255, 256, 257, 1_048_579):
            expected = reference(n, distribution)
            for mode in ('global', 'block'):
                _, counts = measure(mode, distribution, n, 1)
                if counts != expected:
                    raise ValueError(f'oracle mismatch: {mode} {distribution} {n}')
    print('24 GPU oracle comparisons and 3 invalid-input checks passed')


def benchmark(destination):
    rows = []
    for distribution in ('uniform', 'hot'):
        for n in (1 << 20, 1 << 24, 1 << 27):
            for pair in range(5):
                modes = ('global', 'block') if pair % 2 == 0 else ('block', 'global')
                results = {}
                for mode in modes:
                    ms, counts = measure(mode, distribution, n, 5)
                    results[mode] = counts
                    if not (ms > 0):
                        raise ValueError('invalid timing')
                    rows.append(dict(distribution=distribution, n=n, mode=mode, pair=pair,
                                     ms=ms, giga_items_per_second=n / ms / 1e6))
                if results['global'] != results['block']:
                    raise ValueError('benchmark variants disagree')
    with (destination / 'raw.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, rows[0])
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for distribution, n, mode in sorted({(r['distribution'], r['n'], r['mode']) for r in rows}):
        values = [r['ms'] for r in rows if (r['distribution'], r['n'], r['mode']) == (distribution, n, mode)]
        summary.append(dict(distribution=distribution, n=n, mode=mode,
                            median_ms=statistics.median(values), min_ms=min(values), max_ms=max(values)))
    (destination / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


class ReferenceTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(reference(0, 'hot'), [0] * 256)

    def test_first_value(self):
        expected = [0] * 256
        expected[31] = 1
        self.assertEqual(reference(1, 'uniform'), expected)

    def test_seed_and_skew(self):
        uniform = reference(4096, 'uniform')
        hot = reference(4096, 'hot')
        self.assertEqual(sum(uniform), 4096)
        self.assertEqual(sum(hot), 4096)
        self.assertLess(max(uniform), 64)
        self.assertGreater(hot[7], 3700)


if __name__ == '__main__':
    if sys.argv[1] == 'verify':
        verify()
    else:
        benchmark(Path(sys.argv[1]))
