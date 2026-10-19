# Validation record

## Executed cloud checks

GNU C++ 14.2.0 compiled the CPU tests with C++17 and `-O3 -Wall -Wextra -Werror`. Nine oracle/size checks passed: known values, an independent incremental wraparound calculation over six lengths, and invalid sizes. Seven Python analysis/runner tests passed, including incomplete/corrupt evidence rejection, invalid/live pool accounting, adverse performance, protected output paths and source-manifest scope.

Shell syntax and Python source compilation passed. An actual `run.sh` invocation in this CUDA-free cloud passed its CPU checks, returned exit 1 for unavailable GPU tools, retained environment/source/failure evidence and left GPU status pending. A repeated invocation rejected the existing output directory with exit 2 and preserved every artifact byte for byte. Missing, relative, project-root and source-tree output paths were rejected.

Logs, cloud environment and source SHA-256 manifest are in `cloud-cpu/`. Probe outputs remain in ignored local results storage. Synthetic parser fixtures are not benchmark results. CPU checks do not establish CUDA event ordering or target compiler compatibility.

## Executed NVIDIA L4 validation

Two independent GPU allocations ran the unmodified latest-main source `646ec50547f8291792de3f8f49082286764b083f` through `scripts/run.sh`. No application failure or source correction was required. Each requested one NVIDIA L4, two CPU cores, 8 GiB host RAM and a 25-minute limit on the GPU worker. CUDA module 12.4.1 supplied nvcc 12.4.131; GCC 11.5.0, driver 580.178.04 and sm_89 were recorded. Worker elapsed times were 6.411 and 5.617 seconds, excluding scheduling. Source SHA-256 manifests in both runs match the committed executable sources, scripts and tests; documentation was updated afterward.

Each run passed CPU tests, CUDA compilation, invalid-mode rejection and 30 full GPU shape/path/handoff cases repeated three times. Exact payload, guards, event handoff and zero live pool bytes passed. Each generated 81 measured CSV rows. Memcheck, initcheck, racecheck and synccheck each passed the 10-case quick suite repeated three times with zero reported errors. Sanitizer coverage uses the quick shapes, not every benchmark shape. Nsight Compute exited zero and captured the fill kernel duration; this focused capture does not measure allocator cost.

The [project README](../README.md#measured-l4-results) reports both runs' per-shape median latencies. Retain was 9.96–15.31× faster than legacy and 63.31–87.27× faster than release, while retaining 32 MiB for allocations from 1,280 to 4,194,560 bytes. Release reserved zero bytes and was slower than legacy. All async samples had zero used bytes. The reproducible tradeoff is reduced steady-state latency for retained memory; it is not a universal async API speedup. Pool/context setup, contention and multiple outstanding allocations remain outside this experiment.

## Evidence and privacy

[Run 1](repeated-nvidia-l4/attempt-1/) and [run 2](automated-nvidia-l4/latest/) contain actual raw CSV, summary JSON, environment/build flags, source commit and SHA-256 manifests, CPU/build/correctness/sanitizer logs, profiler output and validation state. The runner redacts project/result/home paths and hostname. Published profiler process IDs were additionally replaced with `<pid>`; numerical measurements were preserved. No scheduler identity/account/node listing, GPU UUID, credentials, binary reports or user home paths are included. Logs are explicitly tracked despite the repository's `*.log` ignore rule. No GPU-Computing files were modified.

## Independent artifact review

All 81 raw rows per execution were reprocessed with the committed analyzer; each recomputed summary exactly matched its archived JSON. All nine source SHA-256 values per execution matched the committed implementation, scripts and tests. Both full verification logs and the four sanitizer logs were checked. The two profiler logs contain an actual `gpu__time_duration.sum` observation of 2.05 microseconds; this is a separate instrumented kernel measurement and is not used as allocation-lifetime performance.

Publication uses neutral device-based paths. The second execution's files are the canonical `automated-nvidia-l4/latest/` snapshot; the first is retained under `repeated-nvidia-l4/attempt-1/`. The canonical validation JSON includes provenance fields added during review; raw CSV and numerical/tool evidence are unchanged. No new GPU execution occurred during this review.
