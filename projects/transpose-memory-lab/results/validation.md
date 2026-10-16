# Validation and measured configuration

Executed on 2026-10-16 using one NVIDIA L4, compute capability 8.9, 23,034 MiB visible memory. Driver: 580.178.04. CUDA compiler: 12.4.131; runtime: 12.4. GNU C++: 11.5.0. Compilation: `-std=c++17 -O3 -lineinfo -arch=sm_89`. Nsight Compute: 2026.1.1.0.

Allocation: one GPU, two CPU cores, 4 GiB host memory. Benchmark/verification completed with exit 0 in eight seconds; focused profiling completed with exit 0 in three seconds. Requested walltime limits were 15 minutes and five minutes respectively.

| Check | Result |
|---|---|
| Independent CPU known-answer, rectangular round-trip, invalid inputs | PASS |
| CUDA compilation | PASS |
| Python analysis tests | 3 PASS |
| Shell syntax, CLI help, malformed-input rejection | PASS |
| GPU correctness | 10 shapes x 4 paths; exact equality, poisoned outputs, guards, immutable input: PASS |
| Compute Sanitizer memcheck / initcheck / synccheck | 0 errors each |
| Compute Sanitizer racecheck | 0 errors and 0 warnings |
| Uninstrumented timing | 4 shapes x 4 paths x 15 samples; raw CSV preserved |
| Focused shared-memory counters | Both tiled paths captured; profiler exit 0 |

## Timing and interpretation

Seed 2027; ten warmups/path; default-stream CUDA events; 100 launches/sample for 31 by 33 and ten for the other measured shapes. Allocation, transfers, verification and CPU computation are excluded; launch gaps are included. Paths use matched dimensions.

See the README median table and `nvidia-l4-cuda12_4/benchmark/summary.csv` for dispersion. At 1024 by 1024, padded/naive acceleration was 9.82x and padded/unpadded-tiled acceleration was 2.27x. Padding did not improve the two larger tiled medians. Logical bandwidth can reflect cache traffic; it is not measured DRAM bandwidth.

Separate profiling recorded 1,015,808 shared-load bank conflicts for unpadded tiling and zero for padding at 1024 by 1024. Shared-store conflicts were zero for both. Profiled launches were the correctness pass preceding the benchmark with identical dimensions and mappings. Instrumented application timings in the profile log are **not** used as performance results.

One allocation, one GPU model, fixed path order and same-buffer replay limit generalization. Replicate with rotated order and other working-set sizes/devices before claiming broad acceleration.

## Provenance and publication

Measured source commit: `4a85f14da93dc03edc3d569f0ea85878b021e2fa`. The implementation is unchanged by the documentation/publication cleanup. SHA-256:

```text
dc73ee8d3306fee2bf4bf1e2cd194a9d808125a086837462264e32d14b76bd0b  src/transpose.cu
1a254b4d8060e7b388870c45a8f9187a72a0f7e8995526bf072a4221602673ad  include/reference.hpp
```

Logs in `nvidia-l4-cuda12_4/` are sanitized publication copies. Account/machine names, allocation identifiers and absolute user paths are omitted. Numerical measurements, hashes, versions and pass/fail evidence are preserved. Original operational logs remain in private execution storage. The profiling script was added after the measured source commit; binary/source hashes link the profile to the tested implementation.
