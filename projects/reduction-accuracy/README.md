# Reduction accuracy lab

**Question:** For finite FP32 inputs, how much numerical accuracy does FP64 accumulation buy over an FP32 reduction, and what execution cost should an application measure before choosing it?

Three paths sum the same resident array: a deterministic two-pass FP32 tree, the identical tree with FP64 accumulators and partials, and the toolkit's CUB `DeviceReduce::Sum` with FP32 output. CUB is the established library baseline. The two custom paths isolate accumulator precision; CUB also changes the reduction algorithm. No performance or accuracy superiority is asserted before GPU execution.

## Reproduce

From the repository root, CPU checks need a C++17 compiler and Python 3, with no third-party Python packages:

```bash
bash projects/reduction-accuracy/scripts/test_cpu.sh
```

Inside an existing GPU allocation, prepare CUDA on the worker and run:

```bash
RESULTS_DIR=/path/to/results CUDA_ARCH=sm_89 \
  bash projects/reduction-accuracy/scripts/run.sh
```

Use a fresh absolute `RESULTS_DIR`. The runner refuses existing run artifact names and never deletes previous outputs. It builds only beneath the project's `build/`, writes results beneath `RESULTS_DIR`, and performs no scheduling, remote login, credential access or global configuration changes. `CUDA_MODULE` is prepared externally; no module name is recorded. The target configuration is CUDA Toolkit 12.4.1 (`nvcc` 12.4.131), GNU C++ 11.5, C++17, NVIDIA L4, two CPU cores and 8 GiB host memory. This target is a compatibility requirement, not a claim of completed validation. Default flags: `-std=c++17 -O3 -lineinfo -arch=sm_89 -Iinclude -Xcompiler=-Wall,-Wextra`. Override `CUDA_ARCH` and `CXX` only when deliberately testing another configuration.

Build, CPU oracle/tests, malformed-mode rejection, GPU correctness, benchmark, CSV analysis, available Compute Sanitizer tools and an available focused Nsight Compute attempt run in sequence. Build, correctness, analysis or available sanitizer failures return nonzero. Missing sanitizers and denied/unavailable profiling are recorded explicitly; they never become passing checks. Core subprocess timeouts total less than 25 minutes; normal execution should be substantially shorter, but elapsed time is not yet measured on the target.

## Numerical contract

Inputs contain 1..1,048,576 finite FP32 values. Three reproducible datasets expose different conditions:

- `positive`: `1 + (index % 8)/8`, without cancellation.
- `cancellation`: repeated `[100000000, 1, -100000000, 1]`, including incomplete tails.
- `dyadic`: a defined 32-bit LCG starting at seed 2027, mapped to integers in [-1024, 1024] divided by 16.

A compensated long-double CPU sum supplies the independent oracle. CPU tests separately sum exact integer units of 1/16, check hand-calculated answers and reject empty, excessive, unknown and nonfinite cases. All chosen sums and intermediate FP64 additions are exactly representable: integer units fit comfortably below 2^53. Therefore the FP64 path must match the oracle exactly for this dataset. All paths must also match exactly on the positive dataset, whose positive partial sums are multiples of 1/8 below 2^21.

FP32 cancellation/dyadic paths report observed absolute error and error divided by the sum of input magnitudes. Their safety bound is `gamma_n * sum_abs`, where `gamma_n = (n*u)/(1-n*u)` and `u=2^-24`. This is deliberately conservative and **not an accuracy acceptance target** for applications; it admits expected cancellation error rather than calling rounding a memory bug. Application error budgets must be stricter and workload specific. NaN/Inf, overflow, subnormal behavior, arbitrary lengths beyond the cap and cross-device bitwise reproducibility are outside scope.

GPU checks cover lengths 1, 31, 255, 256, 257, 65,537 and 1,048,576 across three datasets and three paths (63 cases). Each output is NaN-poisoned; input and output guards plus input immutability are checked. Partial-buffer safety additionally depends on sanitizer execution. `--quick` reduces sanitizer checks to lengths 1 and 257 (18 cases).

## Fair timing and interpretation

Benchmark lengths are 257, 65,537 and 1,048,576 across all datasets and paths. Each path receives five warmups, nine samples, and 30 reductions per sample. Path order rotates each sample. Kernel blocks contain 256 threads. Custom first-pass grids use `min(ceil(n/256), 1024)` blocks; a single second-pass block completes the sum. CUB receives the same pointer, count and nonblocking stream. Temporary-storage query/allocation, allocation, data transfers, poisoning, oracle and verification are excluded from timing.

`raw.csv` records CUDA-event stream interval and synchronized host wall milliseconds per reduction, along with error and oracle magnitudes. The host interval includes event submission and synchronization, amortized over 30 reductions; stream intervals include host launch gaps and are not isolated kernel times. Buffers are replayed from the same addresses and can be cache resident. CUB may select an architecture/toolkit-specific implementation. Compare numerical error and timing together; do not treat a lower-precision result as equivalent merely because it is faster.

The analyzer requires all 243 samples, matching settings, finite metrics and matching oracle values. `summary.json` includes median, min, max and IQR for both timing scopes, worst observed absolute/scaled error, and the cancellation condition `sum_abs/abs(sum)` (null for zero sums). It does not silently average missing cases or fabricate speedups.

## Evidence

[Validation record](results/validation.md) distinguishes completed cloud checks from pending GPU execution. GPU completion requires an actual worker record at `results/automated-nvidia-l4/latest/validation.json` plus raw CSV and verification logs for these sources. No earlier project's evidence is reused.

See the [tutorial](docs/tutorial.md), [engineering and product analysis](docs/engineering-analysis.md) and [profiling protocol](docs/profiling.md).

References: [CUDA floating-point guide](https://docs.nvidia.com/cuda/floating-point/), [CUB DeviceReduce](https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceReduce.html), [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html). These are API/background references; target compilation remains pending.
