# CUDA Graph replay lab

When does replaying a captured two-kernel pipeline pay back graph setup, and when is kernel fusion a better choice?

The workload is `y = max(1.25 * x - 0.5, 0)` over resident FP32 arrays. Three implementations share identical inputs and outputs: two ordinary launches (eager), the same two kernels captured in a CUDA Graph, and one fused kernel. This is an established API/workload experiment, not a novel algorithm.

## Build and execute

```bash
bash projects/cuda-graph-replay/scripts/build.sh
python3 -m unittest discover -s projects/cuda-graph-replay/tests -p 'test_*.py' -v
# Run the following inside a GPU allocation:
projects/cuda-graph-replay/build/graph_lab --verify
RESULTS_DIR=/path/to/private/results bash projects/cuda-graph-replay/scripts/run.sh
```

Requires CUDA 12.4-compatible APIs, GNU C++ with C++17, and an NVIDIA GPU. Build defaults to `sm_89`; set `CUDA_ARCH` for another compiler-supported target. Runtime returns 77 when a usable GPU is absent. The run script expects an existing allocation and never submits a nested job.

## Correctness contract

Tests cover lengths 1, 31, 257, 4096 and 1048576. A CPU double-precision formula supplies the oracle. Dyadic inputs make these coefficients and intermediate results exactly representable, so equality is exact for this dataset. Each path must overwrite NaN-poisoned outputs, preserve prefix/suffix guards on output and temporary arrays, and leave input immutable. The captured graph must contain exactly two nodes.

CPU tests include hand-calculated values, a tail length and empty-input rejection. The execution script also runs memcheck, racecheck, initcheck and synccheck; guards alone do not establish memory safety. Exceptional floating-point inputs, graph updates, concurrent replay, changing shapes and pointer lifetimes are outside this executable's contract.

## Timing contract

For each length: capture, instantiate and explicit graph upload are measured separately with a host clock. Twenty warmups per path precede fifteen measured samples. Sample order rotates eager/graph/fused to reduce systematic order bias.

A sample contains 200 pipeline iterations below one million elements and 30 at the largest length. It records CUDA-event stream-interval time, synchronized host wall time, and host submission time per pipeline. Device intervals include host launch gaps; they are not isolated kernel durations. Allocation, H2D/D2H copies and correctness checks are outside the measured replay intervals. Host wall time includes matching event and synchronization overhead amortized over the batch.

An illustrative amortization count is `ceil(setup_us / (eager_wall_us - graph_wall_us))` when the measured difference is positive. Otherwise no finite measured break-even exists. Setup is one observation per shape, not a distribution. Compare against fusion too: graphs are not a replacement for eliminating work when fusion is possible.

## Evidence

GPU correctness, four sanitizer tools and actual timing passed. Sanitized evidence is available in `results/automated-nvidia-l4/latest/`.

See [developer tutorial](docs/tutorial.md) and [engineering analysis](docs/engineering-analysis.md).

Reference: [NVIDIA CUDA Runtime Graph API](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__GRAPH.html).

## Measured NVIDIA L4 results

NVIDIA L4, compute capability 8.9, 23,034 MiB visible device memory; driver 580.178.04; CUDA Toolkit 12.4.131; runtime 12.4. Build: C++17, -O3 -lineinfo -arch=sm_89. Block size: 256 threads. Inputs are deterministic dyadic values based on element index, with no random seed. Table values are medians from 15 rotated-order samples; host wall time is synchronized and normalized per pipeline.

| Elements | Eager wall us | Graph wall us | Fused wall us | Eager / Graph | Setup us | Illustrative break-even |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 3.275 | 2.427 | 1.652 | 1.35x | 195.427 | 231 |
| 31 | 3.319 | 2.435 | 1.683 | 1.36x | 26.100 | 30 |
| 257 | 3.334 | 2.475 | 1.678 | 1.35x | 20.502 | 24 |
| 4096 | 3.322 | 2.575 | 1.670 | 1.29x | 24.738 | 34 |
| 1048576 | 12.242 | 11.544 | 6.212 | 1.06x | 22.104 | 32 |

Graph replay reduced measured eager wall time at every tested length, but fusion was fastest at every length. The fused comparator removes temporary memory traffic and a launch. The two-kernel graph path cannot obtain that benefit merely by replaying the dependency graph.

Break-even is an illustration from one observed setup duration per length and the difference of measured median wall times. It is not a confidence interval or a guaranteed application crossover. First-shape setup includes one-time effects, and fixed-address replay benefits from a stable workload.

Compilation, the independent CPU test and three Python tests passed. Five input lengths times three implementations passed exact full-array comparison, output/temporary guards and input-immutability checks. memcheck, initcheck and synccheck reported zero errors; racecheck reported zero errors and warnings. An additional Nsight Systems node-level trace was collected separately; its timings are excluded from performance comparisons. See the repeated-execution report.

## Repeated execution and profiling

Two additional uninstrumented runs reproduced the result. Across all three runs, graph replay was 1.06x-1.41x faster than eager execution, while fusion remained fastest. [Detailed execution report](results/reproduction-report.md) records run-level variation, setup amortization, source identity, trace counts and reproduction commands.
