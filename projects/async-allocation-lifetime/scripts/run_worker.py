#!/usr/bin/env python3
"""One allocated GPU, no scheduler/SSH, no credentials and no global changes."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time

PROJECT = Path(__file__).resolve().parents[1]


def source_manifest():
    return {str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('src', 'include', 'tests', 'scripts')
            for p in sorted((PROJECT / folder).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts}


def main():
    destination = os.environ.get('RESULTS_DIR', '')
    if not destination or not Path(destination).is_absolute():
        raise ValueError('RESULTS_DIR must be absolute')
    out = Path(destination).resolve()
    if out == PROJECT or out in PROJECT.parents:
        raise ValueError('output directory must not be the project or an ancestor')
    # Output belongs under the project results tree or outside the project.
    if PROJECT in out.parents and PROJECT / 'results' not in out.parents:
        raise ValueError('output inside project must be below results/')
    out.mkdir(parents=True, exist_ok=True)
    artifacts = ['validation.json', 'source-sha256.json', 'source-commit.txt', 'environment.json',
                 'cpu-build.log', 'cpu-oracle.log', 'analysis-tests.log', 'build.log',
                 'diagnose.log', 'verify.log', 'raw.csv', 'benchmark.stderr', 'summary.json',
                 'cli-tests.log', 'memcheck.log', 'initcheck.log', 'racecheck.log',
                 'synccheck.log', 'profile.log', 'profile-application.log', 'failure.log']
    if any((out / name).exists() for name in artifacts):
        raise ValueError('output contains run artifacts; choose a fresh RESULTS_DIR')
    build = Path(tempfile.mkdtemp(prefix='run-', dir=ensure_build()))
    relative_build = str(build.relative_to(PROJECT))
    identities = [(str(PROJECT), '<project>'), (str(out), '<results>'),
                  (str(Path.home()), '<home>'), (socket.gethostname(), '<host>')]
    # Do not inspect environment contents, SSH settings, key files or credentials.
    def safe(text):
        for value, replacement in identities:
            if value and value != '/': text = text.replace(value, replacement)
        return text
    def write(name, text):
        with (out / name).open('x', encoding='utf-8') as f: f.write(safe(text))
    def run(args, name, timeout=30, expect=0, stderr_name=None):
        try:
            proc = subprocess.run(args, cwd=PROJECT, text=True, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            def decoded(value):
                return value.decode('utf-8', errors='replace') if isinstance(value, bytes) else (value or '')
            if stderr_name:
                write(name, decoded(exc.stdout)); write(stderr_name, decoded(exc.stderr))
            else:
                write(name, decoded(exc.stdout)+decoded(exc.stderr)+'\nTIMEOUT\n')
            raise RuntimeError(name+' timed out') from exc
        if stderr_name:
            write(name, proc.stdout); write(stderr_name, proc.stderr)
        else:
            write(name, proc.stdout + proc.stderr)
        if proc.returncode != expect:
            raise RuntimeError(name + ' returned unexpected exit code ' + str(proc.returncode))
        return proc.stdout
    state = {'status': 'running', 'gpu_correctness': 'pending', 'benchmark': 'pending',
             'sanitizers': {}, 'profiling': 'pending', 'seed': 2027}
    started = time.monotonic()
    try:
        write('source-sha256.json', json.dumps(source_manifest(), indent=2)+'\n')
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=PROJECT,
                                capture_output=True, text=True, check=False, timeout=10)
        write('source-commit.txt', commit.stdout.strip()+'\n' if commit.returncode==0 else 'unavailable\n')
        arch = os.environ.get('CUDA_ARCH', 'sm_89')
        if not re.fullmatch(r'sm_[0-9]{2,3}', arch): raise ValueError('invalid CUDA_ARCH')
        cxx = os.environ.get('CXX', 'g++')
        # One executable name/path only, no shell expansion or injected arguments.
        common = ['-std=c++17', '-O3', '-Iinclude']
        run([cxx, *common, '-Wall', '-Wextra', '-Werror', 'tests/reference_test.cpp', '-o', relative_build+'/reference_test'], 'cpu-build.log')
        run(['./'+relative_build+'/reference_test'], 'cpu-oracle.log')
        run(['python3','-B','-m','unittest','discover','-s','tests','-p','test_*.py','-v'], 'analysis-tests.log')
        state['cpu_tests'] = 'passed'
        flags = ['-std=c++17','-O3','-lineinfo','-arch='+arch,'-Iinclude','-Xcompiler=-Wall,-Wextra']
        metadata = {'build_flags': flags,
                    'timing': {'samples':9,'repetitions':20,'warmup':3,'seed':2027,
                               'scope':'host wall allocation+fill kernel+free+synchronization; pool/context creation and correctness transfers excluded'}}
        missing = []
        for label, args in (
                ('cuda_compiler',['nvcc','--version']),
                ('host_compiler',[cxx,'--version']),
                ('gpu',['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'])):
            try:
                version = subprocess.run(args,capture_output=True,text=True,check=False,timeout=15)
                metadata[label] = version.stdout.strip() if version.returncode==0 else 'unavailable: '+version.stderr.strip()
                if version.returncode != 0: missing.append(args[0])
            except (OSError, subprocess.TimeoutExpired):
                metadata[label] = 'unavailable'
                missing.append(args[0])
        write('environment.json',json.dumps(metadata,indent=2)+'\n')
        if missing: raise RuntimeError('required GPU environment tool unavailable: '+', '.join(missing))
        run(['nvcc', *flags, '-ccbin', cxx, 'src/pool.cu','-o',relative_build+'/pool_lab'], 'build.log', timeout=180)
        binary = './'+relative_build+'/pool_lab'
        run([binary,'--bad-mode'], 'cli-tests.log', expect=1)
        run([binary,'--diagnose'], 'diagnose.log')
        run([binary,'--verify'], 'verify.log', timeout=180)
        state['gpu_correctness'] = 'passed'
        run([binary,'--benchmark'], 'raw.csv', timeout=180, stderr_name='benchmark.stderr')
        run(['python3','scripts/analyze.py',str(out/'raw.csv')], 'summary.json')
        state['benchmark'] = 'passed'
        for tool in ('memcheck','initcheck','racecheck','synccheck'):
            if shutil.which('compute-sanitizer'):
                run(['compute-sanitizer','--tool',tool,'--error-exitcode','1',binary,'--quick'], tool+'.log', timeout=120)
                state['sanitizers'][tool] = 'passed'
            else:
                state['sanitizers'][tool] = 'unavailable'
        if shutil.which('ncu'):
            # No binary report: avoid persisting host names or absolute process paths.
            args = ['ncu','--target-processes','application-only','--kernel-name','fill_pattern',
                    '--launch-count','1','--metrics','gpu__time_duration.sum',
                    '--page','details',binary,'--profile']
            try:
                proc = subprocess.run(args,cwd=PROJECT,capture_output=True,text=True,timeout=120,check=False)
                write('profile.log',proc.stdout+proc.stderr)
                state['profiling'] = 'completed_attempt' if proc.returncode==0 else 'unavailable_or_denied'
                state['profiler_exit_code'] = proc.returncode
            except subprocess.TimeoutExpired:
                write('profile.log','Profiler timed out; no completed counter capture.\n')
                state['profiling'] = 'timeout'
            # Validate the same focused path uninstrumented even if counters are denied.
            run([binary,'--profile'],'profile-application.log')
        else: state['profiling'] = 'unavailable'
        state['status'] = 'passed'
        return 0
    except Exception as exc:
        state['status'] = 'failed'
        state['failure'] = safe(type(exc).__name__ + ': ' + str(exc))
        if not (out/'failure.log').exists(): write('failure.log',state['failure']+'\n')
        return 1
    finally:
        state['elapsed_seconds'] = round(time.monotonic()-started,3)
        write('validation.json',json.dumps(state,indent=2)+'\n')
        # Keep all build and output artifacts for diagnosis; remove nothing.


def ensure_build():
    path = PROJECT/'build'
    path.mkdir(exist_ok=True)
    return path


if __name__ == '__main__':
    try: sys.exit(main())
    except Exception as exc:
        print('runner configuration error: '+str(exc), file=sys.stderr)
        sys.exit(2)
