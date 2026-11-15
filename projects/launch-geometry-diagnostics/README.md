# Launch geometry diagnostics

**Question:** Does the block size recommended by CUDA's occupancy API minimize execution time, and what should a diagnostic distinguish between a legal launch, predicted occupancy and measured performance?

The same grid-stride kernel runs with four policies: fixed block sizes 128, **256 (baseline)** and 512, and the result of `cudaOccupancyMaxPotentialBlockSize`. Inputs, output contract, arithmetic work and stream are matched. Each label is timed separately even when the API selects a block size already represented in the sweep.

## Reproduce from the repository root

```bash
bash projects/launch-geometry-diagnostics/scripts/test_cpu.sh
# Inside an existing GPU allocation, with CUDA prepared externally:
RESULTS_DIR=/path/to/fresh/results CUDA_ARCH=sm_100 \
  bash projects/launch-geometry-diagnostics/scripts/run.sh
```

Use a fresh absolute `RESULTS_DIR`. The runner builds only beneath this project's `build/`, writes evidence beneath `RESULTS_DIR`, rejects existing artifact names and removes no previous outputs. It does not connect remotely, submit jobs, read credentials or change global settings. The caller prepares any `CUDA_MODULE`; the script does not load modules.

Run in a Slurm allocation with one NVIDIA B200 and CUDA 12.8 or later. Build flags are `-std=c++17 -O3 -lineinfo -arch=sm_100 -Iinclude -Xcompiler=-Wall,-Wextra`. `CXX` selects a host compiler. Helpers use Python's standard library.

## What the experiment isolates

For each element, the kernel repeats `value = value*1664525 + 1013904223` modulo 2^32 for either 1 or 64 rounds. One round emphasizes launch/data movement overhead; 64 rounds increases a dependency chain's arithmetic work without changing the kernel signature or correctness mapping. These are workload descriptions, not measured bottleneck classifications.

A CPU oracle composes the affine recurrence in 64-bit arithmetic and applies the resulting map independently of GPU lane assignment. Tests compare it with iterative modulo-32 arithmetic, known answers, zero/high-bit/wraparound inputs and invalid shapes. Output poisoning uses the complement of the expected result; unwritten elements therefore always fail. GPU tests check full output equality, prefix/suffix input/output guards and immutable input.

The grid is `min(ceil(n/block), 32*SM_count, maxGridSize.x)`. The cap limits launch count while permitting enough blocks for the target's per-SM resident-block limits; it is not a measured optimal grid. The CPU planner rejects zero/excessive lengths, zero/out-of-range blocks, zero SM/grid limits and an overflowing cap calculation. Its coverage test proves each element is visited exactly once for multiple capped grids and block sizes. The GPU executable records actual kernel/device limits and rejects invalid planned geometry.

`cudaOccupancyMaxActiveBlocksPerMultiprocessor` supplies a resource-based resident-block upper bound. The analyzer computes `active_blocks*block/maxThreadsPerMultiProcessor` for these warp-aligned block sizes. **Predicted occupancy is not achieved occupancy, useful lane utilization or a performance measurement.** Small grids may leave most SMs idle despite a high per-SM resource bound. The potential-block-size API's minimum-grid recommendation is recorded diagnostically; the experiment uses its own documented grid policy.

## Correctness and launch-error diagnosis

The normal GPU suite covers lengths 1, 31, 257, 65,537 and 1,048,576, two round counts and all four policies: 40 payload cases. It additionally launches a block one thread larger than the device limit using real allocated buffers. That launch must fail immediately with `cudaErrorInvalidConfiguration`. The test reads/clears the error and then executes a legal launch with full oracle verification, demonstrating recovery from this rejected configuration. It never creates an illegal-memory-access fault or claims every asynchronous failure is recoverable.

The `--quick` suite used by Compute Sanitizer covers lengths 1 and 257, both round counts and all policies (16 payload cases). It intentionally skips the expected invalid-API call so that sanitizer API-error reporting cannot confuse a handled negative test with a memory defect. Full error recovery remains covered by the separate normal verification log. Guards do not detect every possible in-bounds scheduling error and do not replace sanitizers.

## Fair timing

The sweep uses lengths 257, 65,537 and 1,048,576, both round counts and all four policies. Five warmups/policy precede nine samples, each containing 20 launches. Policy order rotates each sample; total raw rows are 216. Allocation, transfers, output poisoning, oracle/verification and occupancy API queries are outside the timed intervals. All variants use the same resident buffers.

`device_ms` is a CUDA-event stream interval per launch; it includes possible host submission gaps and is not an isolated instruction duration. `host_ms` is synchronized host wall time per launch, including event submission, launch error check and matching event synchronization amortized over the batch. No transfer-inclusive application speedup is inferred. Replayed buffers can reside in cache, so this project does not report DRAM bandwidth.

`raw.csv` includes actual block/grid sizes, SM count, maximum resident threads, resource-predicted active blocks and timing. The analyzer requires all matched rows, stable device/configuration metadata, legal occupancy bounds, exact grid policy and finite positive timings. `summary.json` reports median/min/max/IQR for both scopes, the best observed fixed policy, heuristic/fixed ratios and paired saving ranges. A slower recommendation is a valid result. Choosing the lowest sample median is descriptive, not a statistical guarantee.

## Worker and evidence

On the measured B200, the occupancy API selected 1,024 threads per block. The fastest fixed policy varied with input size and work per element. At 1,048,576 elements, the one-round kernel ran in 3.66 µs with a fixed block of 512 versus 3.81 µs with the API recommendation. With 64 rounds, a fixed block of 256 ran in 4.70 µs versus 5.12 µs. Use the API recommendation as a starting point, then check it against measured choices at representative shapes.

`run.sh` executes CPU compilation and oracle tests, CUDA build, CLI and launch-error checks, normal GPU correctness, benchmarks, four Compute Sanitizer tools, and a focused Nsight Compute counter capture. The runner returns nonzero when a required check or tool fails.

The results directory contains validation state, source commit/hashes, environment/build flags, application/tool logs, raw CSV and summary. Known operational absolute paths and host identity in tool output are replaced by neutral markers; numerical data is unchanged. No binary profiler report is published.

[Validation record](results/validation.md) lists the evidence and environment for this project.

See the [tutorial](docs/tutorial.md), [engineering/product analysis](docs/engineering-analysis.md) and [profiling protocol](docs/profiling.md).

References: [CUDA occupancy API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OCCUPANCY.html), [CUDA execution control](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html), [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html). API references are background, not proof of target execution.
