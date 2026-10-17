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

GPU validation and actual measurements are pending for this revision. No speedup is claimed until actual results are published. The server worker will place sanitized evidence in `results/automated-nvidia-l4/latest/`.

See [developer tutorial](docs/tutorial.md) and [engineering analysis](docs/engineering-analysis.md).

Reference: [NVIDIA CUDA Runtime Graph API](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__GRAPH.html).
