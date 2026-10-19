# Async allocation lifetime lab

**Question:** How does a private CUDA memory pool's release threshold change the cost and retained-memory footprint of repeated allocation → kernel → free lifetimes, and how should an allocation safely cross streams?

The experiment implements three complete, synchronous-at-the-boundary lifetimes:

| Path | Allocation / release | Private-pool release threshold |
|---|---|---|
| `legacy` | `cudaMalloc`, kernel, stream synchronize, `cudaFree` | Not used |
| `async_release` | `cudaMallocFromPoolAsync`, kernel, `cudaFreeAsync`, stream synchronize | 0 |
| `async_retain` | Same stream-ordered APIs | UINT64_MAX |

The legacy path is the standard synchronous allocation baseline. The two async paths differ only in pool retention policy. Stream-ordered allocation is exercised through an explicitly owned pool, rather than changing the device's default pool or a global setting. Pool creation and CUDA context startup are outside measured lifetimes. Two HPG L4 runs measured the latency/memory tradeoff below.

## Reproduce

From the repository root:

```bash
bash projects/async-allocation-lifetime/scripts/test_cpu.sh
# Inside an existing GPU allocation, with the toolkit prepared externally:
RESULTS_DIR=/path/to/fresh/results CUDA_ARCH=sm_89 \
  bash projects/async-allocation-lifetime/scripts/run.sh
```

Use a fresh absolute `RESULTS_DIR`. The runner builds under the project's `build/`, writes evidence under `RESULTS_DIR`, rejects existing artifact names and never removes previous results. It performs no remote login, scheduling, credential reading or global configuration changes. The worker prepares `CUDA_MODULE`; the script does not change modules.

Target: CUDA Toolkit 12.4.1 (`nvcc` 12.4.131), GNU C++ 11.5, C++17, one NVIDIA L4, two CPU cores and 8 GiB host memory. Default flags are `-std=c++17 -O3 -lineinfo -arch=sm_89 -Iinclude -Xcompiler=-Wall,-Wextra`; `CXX` and `CUDA_ARCH` may select other explicit test configurations. Python helpers use the standard library. The executable requires `cudaDevAttrMemoryPoolsSupported`; absence fails with a clear diagnostic rather than silently substituting a baseline. Target compilation and execution passed on HPG.

## Correctness and allocation lifetime

A kernel writes `uint32(index)*2654435761 + 2027` modulo 2^32. An independent CPU oracle uses 64-bit multiplication and masking, with known-answer and incremental-wraparound tests. No floating-point tolerance is needed.

GPU tests cover payload lengths 1, 31, 256, 257, 65,537 and 1,048,576. Every allocation includes 32 prefix and 32 suffix guard elements. Pinned verification output is initialized with guards and the complement of each expected payload value, then copied to the allocation on its producer stream. Full-array comparison and both guards are checked after completion. No unwritten element can match its poison.

Each path runs same-stream correctness three times per shape. Both async paths additionally run a two-stream handoff three times: producer allocates/initializes → records ready event → consumer waits → writes/copies result → records done event → producer waits → queues free → synchronizes. The free therefore cannot execute before the consumer's final access. This produces 30 logical shape/path/handoff cases, each repeated three times. The quick sanitizer suite uses lengths 1 and 257 for ten cases, also repeated three times. Available pool `UsedMemCurrent` must be zero after synchronized release.

No intentionally racy or use-after-free kernel is executed as a negative example. CPU size validation rejects zero and excessive counts; unknown command modes fail before GPU initialization. CPU tests do not establish event ordering; the real GPU handoff checks and sanitizers supply that evidence when executed.

## Benchmark contract

Payload lengths are 256, 65,537 and 1,048,576. Allocated bytes are `(payload_elements+64)*4`, including the same guard padding as correctness allocations, so validation does not prime a retained pool with a different allocation size. All paths execute one identical 256-thread-block fill kernel over the payload; no transfers occur in measured lifetimes.

Each path receives an untimed correctness/context warmup and three full lifetime warmups. Nine samples contain 20 allocation/kernel/free lifetimes each; path order rotates per sample. Every lifetime returns after its completion boundary. Host wall milliseconds per lifetime include allocation, kernel submission, required synchronization and release. The legacy path explicitly synchronizes before freeing; async paths order the free in the producer stream and synchronize afterward. These are legal complete lifetimes, not identical internal synchronization sequences. Pool construction, data verification/copies, metadata queries and teardown are excluded.

`raw.csv` contains 81 samples, with reserved and used pool bytes queried after the measured batch. For `legacy`, pool byte fields are zero sentinels and do not describe all device memory usage. Zero used bytes for async paths is checked, not guessed. Reserved bytes describe the private pool's runtime accounting, not process RSS or total GPU memory. Threshold zero permits release at synchronization; exact physical release behavior remains runtime dependent. Pointer reuse is not assumed or used as proof.

`summary.json` records median/min/max/IQR, matched latency ratios, paired retention saving ranges, allocation bytes and reserved-byte ranges. Slow or neutral async results remain visible. These steady-state samples exclude pool creation; an application with a short lifetime must measure setup separately. Concurrent allocation pressure, multiple outstanding allocations, interprocess sharing and graph capture are outside scope.

## Worker execution and evidence

The worker runs CPU checks → CUDA compilation → invalid-mode rejection → GPU correctness → benchmark/analysis → available Compute Sanitizer modes → available focused profiling. Core failures and available sanitizer errors return nonzero. Missing tools and denied/unsupported profiling are explicitly unavailable, never passing. Subprocess limits fit a 25-minute allocation; the two HPG worker runs completed in 6.411 and 5.617 seconds (excluding allocation queue time).

Results contain source commit/hashes, environment/build flags, correctness and tool logs, raw CSV, summary and validation state. Known operational absolute paths and host identity in tool messages are replaced with neutral markers; numerical data and versions remain intact. No binary profiler report is published.

## Measured HPG L4 results

Two independent GPU allocations executed `scripts/run.sh` against source commit `646ec50547f8291792de3f8f49082286764b083f`. Both passed CPU tests, 30 GPU cases repeated three times, 81 benchmark samples, all four Compute Sanitizer modes and focused Nsight Compute capture. CUDA 12.4.131, GCC 11.5.0, driver 580.178.04 and `sm_89` were used.

Median complete-lifetime latency, in microseconds (run 1 / run 2):

| Payload elements | Legacy | Async release | Async retain |
|---|---:|---:|---:|
| 256 | 85.409 / 84.663 | 720.137 / 718.927 | 8.465 / 8.238 |
| 65,537 | 84.922 / 83.863 | 719.438 / 718.474 | 8.527 / 8.313 |
| 1,048,576 | 171.381 / 170.280 | 722.460 / 721.247 | 11.412 / 11.122 |

Retention reduced median latency by 9.96–15.31× versus legacy and 63.31–87.27× versus threshold zero across these runs and shapes. This required 33,554,432 reserved pool bytes (32 MiB), even for the 1,280-byte allocation. Threshold zero reported zero reserved bytes; both async paths reported zero live bytes after completion. Threshold-zero async lifetimes were slower than legacy here: API choice alone does not guarantee lower latency.

These are host-wall steady-state measurements with one outstanding allocation, excluding pool/context setup. Two allocations on one GPU model do not establish behavior under contention or on other devices. The focused kernel capture is separate from lifetime timing.

[Validation record](results/validation.md) documents the protocol and limitations. Actual logs, CSV, environment and source hashes are in [run 1](results/hpg-nvidia-l4/attempt-1/) and [run 2](results/hpg-nvidia-l4/attempt-2/). Previous projects' measurements are not evidence here.

Read the [tutorial](docs/tutorial.md), [engineering/product analysis](docs/engineering-analysis.md) and [profiling protocol](docs/profiling.md).

References: [CUDA stream-ordered memory allocator](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html), [CUDA events](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html), [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html). API references are background, not a claim of completed target validation.
