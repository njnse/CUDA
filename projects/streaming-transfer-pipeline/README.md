# Streaming transfer pipeline

**Question:** With an identical chunked H2D → affine kernel → D2H workload, when do pinned host buffers and two independent streams improve synchronized end-to-end throughput?

Three concrete paths share the same data, kernel, chunk sizes and output contract:

| Path | Host memory | Active streams | Device staging buffers |
|---|---|---:|---:|
| `pageable_serial` | ordinary vector storage | 1 | 1 |
| `pinned_serial` | `cudaMallocHost` | 1 | 1 |
| `pinned_pipeline` | `cudaMallocHost` | 2 | 2 |

The two serial paths isolate host-memory choice. The pinned paths isolate the stream/staging topology. All paths use `cudaMemcpyAsync` followed by the same kernel and return copy; pageable calls may block or internally stage data. No speedup or overlap is asserted before measurement. The serial baseline is an ordinary CUDA implementation, not a deliberately slow CPU reference. A CPU oracle establishes correctness independently.

## Reproduce from the repository root

```bash
bash projects/streaming-transfer-pipeline/scripts/test_cpu.sh
# In an existing GPU allocation with CUDA already prepared:
RESULTS_DIR=/path/to/fresh/results CUDA_ARCH=sm_89 \
  bash projects/streaming-transfer-pipeline/scripts/run.sh
```

`RESULTS_DIR` must be absolute and must not contain existing run artifact names. The runner builds under the project's `build/` and writes evidence under `RESULTS_DIR` only. It does not schedule jobs, connect remotely, read credentials, delete user files or modify global settings. The caller prepares any `CUDA_MODULE`; the script does not change modules.

Target compatibility: CUDA Toolkit 12.4.1 (`nvcc` 12.4.131), GNU C++ 11.5, C++17, `CUDA_ARCH=sm_89`, one NVIDIA L4, two CPU cores and 8 GiB host memory. This is a target, not completed validation. Flags: `-std=c++17 -O3 -lineinfo -arch=sm_89 -Iinclude -Xcompiler=-Wall,-Wextra`. `CXX` selects the host compiler. Python helpers use only the standard library. Kernel blocks contain 256 threads. Benchmark buffers occupy substantially less than 1 GiB; the largest host input contains 4,194,307 32-bit elements.

## Correctness and boundaries

The kernel computes `y = 1664525*x + 1013904223` modulo 2^32. The CPU oracle uses 64-bit multiplication and masking; scalar 32-bit tests independently verify it. This integer workload makes comparison exact and avoids floating-point tolerance hiding scheduling or transfer errors. Inputs use a defined 32-bit LCG with seed 2027 and directed zero/high-bit/wraparound values.

Lengths 1, 31, 257, 65,537 and 1,048,579 are combined with chunk capacities 256, 4,096 and 65,536 across all three paths: 45 cases. Each case executes twice with freshly poisoned host outputs. CPU planning tests also include the maximum benchmark length and reject empty/excessive lengths, empty/excessive chunks and invalid stream counts. Outputs are initialized to the bitwise complement of their expected values, so an unwritten element always fails.

Each chunk is assigned one contiguous, nonoverlapping host output range. Serial scheduling uses slot 0; pipelined scheduling alternates slots. A slot's H2D, kernel and D2H operations share one stream, so later chunks cannot overwrite its device input before earlier work finishes. Pinned host arrays remain alive and unchanged until both streams synchronize. Verification checks every output, host input immutability, host/device prefix/suffix guards and the final device input for each used slot. Guards do not replace sanitizers or detect every within-capacity tail overwrite.

## Timing contract

The sweep uses total lengths 65,537, 1,048,579 and 4,194,307, each with chunk capacities 4,096 and 65,536. Each path receives three warmups, nine samples and three full-array repetitions per sample. Path order rotates by sample. There are 162 raw samples.

`host_ms` is synchronized host wall time per full-array repetition, including all chunk submissions, H2D transfers, kernels, D2H transfers and final synchronization. Each repetition synchronizes before the next begins. Allocation, CPU input generation, initial guard setup, output poisoning and correctness verification are excluded. Inputs/outputs and staging buffers are reused between repetitions. Host thread count is one. There is no separate device-only timing presented as end-to-end latency.

`logical_GBps = 8*n/(host_ms*1e6)` counts one host-to-device and one device-to-host payload. It is **not** measured PCIe bandwidth, DRAM bandwidth, or proof of simultaneous bidirectional transfers. Pinned allocation/registration costs are excluded; short-lived applications must include them separately before choosing a default. No transfer-free or fused comparator is needed here because every path must return the same full result to the host.

The analyzer rejects incomplete or duplicate samples, mismatched settings, nonfinite/nonpositive values and inconsistent logical bandwidth. `summary.json` records median/min/max/IQR, matched pinned-path ratios and paired saving ranges. A ratio below one is retained; unsuccessful pipelining is a valid result.

## Worker evidence and status

`run.sh` executes CPU compilation/oracle/tests, CUDA build, malformed-mode rejection, GPU correctness, the benchmark, analysis, available Compute Sanitizer tools and available focused Nsight Compute profiling. Core failure or an available sanitizer failure returns nonzero. Unavailable sanitizers and denied/unsupported profiling remain explicitly unavailable, not passing. Process timeouts are bounded to fit the 25-minute allocation; normal duration has not yet been measured on the target.

The output directory contains `validation.json`, source commit and SHA-256 manifest, environment/build flags, application/tool logs, `raw.csv` and `summary.json`. Captured tool messages replace known operational absolute paths and host identity with neutral markers; CSV contains no operational identifiers. Binary profiler reports are not published.

[Cloud validation](results/validation.md) records completed checks. CUDA compilation and GPU execution are pending until the actual worker publishes `results/automated-nvidia-l4/latest/validation.json` and matching raw CSV/logs for these sources. Earlier projects' results are not evidence for this one.

See [tutorial](docs/tutorial.md), [engineering/product analysis](docs/engineering-analysis.md) and [profiling protocol](docs/profiling.md).

References: [CUDA stream synchronization behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html), [CUDA API synchronization behavior](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html), [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html), [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html). These are background references, not evidence of target validation.
