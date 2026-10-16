# Measurement and debugging protocol

Run on an allocated compute node. Build with `-O3 -lineinfo`; avoid debug compilation for timing. Record `nvidia-smi`, `nvcc --version`, build target, source identity and Slurm job ID. Check other activity and power/clock conditions when interpreting variability.

```bash
compute-sanitizer --tool memcheck --error-exitcode 1 build/transpose_lab --verify
compute-sanitizer --tool racecheck --error-exitcode 1 build/transpose_lab --verify
compute-sanitizer --tool synccheck --error-exitcode 1 build/transpose_lab --verify
compute-sanitizer --tool initcheck --error-exitcode 1 build/transpose_lab --verify
nsys profile --trace=cuda --output=transpose-trace build/transpose_lab --benchmark 4096 4096 5
# Inspect names supported by the installed Nsight Compute release first:
ncu --query-metrics > supported-metrics.txt
ncu --set full --launch-count 20 --export transpose-report build/transpose_lab --benchmark 4096 4096 3
```

These commands assume the project directory as the current directory. The full-set profile may be expensive and includes verification launches; inspect the report's kernel names and choose a kernel filter using your installed tool's help before drawing variant-specific conclusions. Re-profile only the relevant kernel when needed. Tool permissions may prohibit hardware counters; preserve the actual denial instead of replacing it with inferred metrics.

Investigate shared-memory transactions/conflicts, global load/store efficiency, achieved throughput and occupancy. Metric names vary by architecture/tool version. Do not hard-code a counter result from a different GPU. Profiler replay and sanitizers perturb execution; timing for comparison must come from a separate uninstrumented run.

The included analysis script reports per-launch median, minimum and maximum from real CSV. Its unit tests use synthetic fixtures only to verify statistics and rejection of invalid measurements; fixture numbers are never performance evidence.
