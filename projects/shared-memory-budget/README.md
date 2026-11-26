# Dynamic shared-memory budget and block residency

Sweep five dynamic shared-memory reservations while keeping the same CUDA kernel, input, block size and grid. Each block writes and reads one shared word for a fixed salt. The rest of the reservation changes resource availability without changing the logical work. A 256-byte reservation is the baseline. The other cases request 32, 64, 112 and 192 KiB per block.

The CUDA Runtime reports the default and opt-in dynamic limits, total shared capacity per SM and register count. The runner opts in to the maximum for the kernel before measurements. It also checks that requesting an opt-in limit one byte above the reported capacity returns `cudaErrorInvalidValue`. The occupancy API estimates resident blocks per SM at each request size. [CUDA Runtime API for function attributes](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-runtime-api/group__CUDART__EXECUTION.html)

## Reproduce

Use a Slurm allocation exposing one NVIDIA B200 with CUDA 12.8 or later, Python 3, Compute Sanitizer and Nsight Compute.

```sh
export RESULTS_DIR="$(mktemp -d)" CUDA_ARCH=sm_100
bash projects/shared-memory-budget/scripts/run.sh
```

The CPU oracle uses 64-bit arithmetic, division and modulo reduction to emulate the kernel's 32-bit rotates and adds. It checks every output and a guard beyond the buffer. Validation covers 1, 255, 256, 257 and 65,537 values at both ends of the reservation sweep, four invalid CLI inputs, and the runtime rejection case. The deterministic input offset and shared salt are both 2027.

The benchmark processes 4,194,307 elements in 256-thread blocks. Seven rounds rotate the order of reservations. After three warmup launches, each sample measures six launches in one CUDA event interval and divides by six. Allocations, copies, resource queries, attribute changes and host verification are outside that interval. Two separate Nsight runs capture kernel duration and achieved occupancy at the endpoints.

For a production kernel, use the reservation sweep to identify regime changes, then compare them with the reuse or latency benefit of genuinely using the extra memory. This experiment isolates resource reservation for one instruction mix and block size.

## Measured results

The November 25 run used one NVIDIA B200, driver 580.178.04, CUDA 12.8.93 and native `sm_100`. The runtime reported a default dynamic limit of 48 KiB per block, an opt-in limit of 227 KiB per block and 228 KiB total shared memory per SM. The above-limit rejection, native build, ten full-output oracle and guard cases, four invalid CLI tests, 35 timings, all four Compute Sanitizer tools at both endpoints and both Nsight captures passed.

| Dynamic bytes per block | Runtime estimate of resident blocks per SM | Median latency | Sample range |
|---:|---:|---:|---:|
| 256 B | 8 | 20.981 µs | 20.816–20.992 µs |
| 32 KiB | 6 | 22.357 µs | 22.187–22.357 µs |
| 64 KiB | 3 | 34.811 µs | 34.619–34.821 µs |
| 112 KiB | 2 | 48.981 µs | 48.949–49.152 µs |
| 192 KiB | 1 | 91.819 µs | 91.643–91.824 µs |

Nsight measured achieved occupancy of 84.86% at 256 B and 12.23% at 192 KiB. At the higher reservation, elapsed time was 4.38× the baseline. The same code and input ran in every case. The observations show a cost for reserving capacity without increasing useful shared-memory reuse. Changes in residency and available L1 capacity can occur together, so these measurements do not assign all the added time to occupancy.

[Raw timings, runtime limits, profiler captures, validation logs and source hashes](results/automated-nvidia-b200/latest/)
