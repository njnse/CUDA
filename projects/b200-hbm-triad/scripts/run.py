import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess

project = Path(__file__).resolve().parents[1]
out = Path(os.environ['RESULTS_DIR'])
if not out.is_absolute():
    raise SystemExit('RESULTS_DIR must be absolute')
out.mkdir(parents=True, exist_ok=True)
if any(out.iterdir()):
    raise SystemExit('Refusing to overwrite existing evidence')
if os.environ.get('CUDA_ARCH', 'sm_100') != 'sm_100':
    raise SystemExit('Only native sm_100 is supported')

def run(args, name, expected=0):
    result = subprocess.run(args, text=True, capture_output=True)
    (out / (name + '.txt')).write_text(result.stdout)
    (out / (name + '.stderr')).write_text(result.stderr)
    if result.returncode != expected:
        raise RuntimeError(f'{name} failed: exit {result.returncode}')
    return result.stdout

device = run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], 'device')
if any('B200' not in line for line in device.strip().splitlines()):
    raise SystemExit('B200 devices required')
compiler = run(['nvcc', '--version'], 'nvcc')
build = project / 'build'
build.mkdir(exist_ok=True)
exe = str(build / 'triad')
flags = ['-std=c++17', '-O3', '-lineinfo', '-arch=sm_100']
run(['nvcc', *flags, str(project / 'src/triad.cu'), '-o', exe], 'build')
run([exe, 'invalid'], 'invalid-cli', expected=1)
run([exe, 'verify'], 'correctness')
raw = run([exe, 'bench'], 'benchmark')
(out / 'raw.csv').write_text(raw)
groups = {}
for row in csv.DictReader(raw.splitlines()):
    groups.setdefault((int(row['n']), row['variant']), []).append(float(row['ms']))
summary = []
for (n, variant), samples in sorted(groups.items()):
    if len(samples) != 9 or any(x <= 0 for x in samples):
        raise RuntimeError('Invalid benchmark samples')
    median = statistics.median(samples)
    summary.append(dict(n=n, variant=variant, samples=len(samples), median_ms=median,
                        min_ms=min(samples), max_ms=max(samples), logical_GBps=n*12/median/1e6))
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
status = {'correctness': 'passed', 'benchmark': 'passed', 'sanitizers': {}, 'profiler': 'pending'}
(out / 'source-sha256.json').write_text(json.dumps({str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest()
    for folder in ['src', 'scripts'] for p in sorted((project/folder).glob('*')) if p.is_file()}, indent=2)+'\n')
for tool in ['memcheck', 'initcheck', 'racecheck', 'synccheck']:
    if shutil.which('compute-sanitizer'):
        run(['compute-sanitizer', '--tool', tool, '--error-exitcode', '86', exe, 'verify'], tool)
        status['sanitizers'][tool] = 'passed'
    else:
        status['sanitizers'][tool] = 'unavailable'
if shutil.which('ncu'):
    # Profile separately. Correctness launches are sufficient for counters; skip them to capture large inputs.
    proc = subprocess.run(['ncu', '--csv', '--metrics',
        'gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed',
        '--launch-count', '2', exe, 'profile'], text=True, capture_output=True)
    (out/'ncu.csv').write_text(proc.stdout)
    (out/'ncu.stderr').write_text(proc.stderr)
    status['profiler'] = 'passed' if proc.returncode == 0 else f'failed_exit_{proc.returncode}'
else:
    status['profiler'] = 'unavailable'
(out/'environment.json').write_text(json.dumps(dict(device=device, compiler=compiler, flags=flags,
    seed='deterministic modular pattern', warmup=5, repetitions=10, samples=9,
    timing='CUDA events; kernel launches only; excludes transfers, allocation, correctness and profiling'), indent=2)+'\n')
(out/'checks.json').write_text(json.dumps(status, indent=2)+'\n')
if status['profiler'] != 'passed' or any(v != 'passed' for v in status['sanitizers'].values()):
    raise SystemExit('Incomplete sanitizer/profiler validation; inspect checks.json')
print('PASS B200 correctness, benchmark, sanitizers and profiling')
