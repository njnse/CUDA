# B200 CUDA module load: cubin versus PTX

Measure how the choice of a native Blackwell cubin or driver-compiled PTX affects the first request to a CUDA module. Both images use the same integer transform and compiler optimization flags. Native `sm_100` cubin is the baseline. Generic `compute_100` PTX exercises driver JIT compilation. See the [Blackwell compatibility guide](https://docs.nvidia.com/cuda/archive/12.8.2/blackwell-compatibility-guide/index.html) and [Driver API module reference](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html).

## Reproduce

Inside an existing Slurm allocation exposing one NVIDIA B200, use CUDA 12.8 or newer, Python 3, Compute Sanitizer and Nsight Compute. From the repository root:

```sh
export CUDA_ARCH=sm_100
export RESULTS_DIR="$(mktemp -d)"
bash projects/b200-module-load/scripts/run.sh
```

The runner compiles both images, checks sizes 1, 255, 256, 257 and 65,537 against a Python integer oracle, verifies a sentinel past the logical output, and checks the driver error for a missing symbol. Inputs use a fixed integer sequence. It then alternates the order of nine cubin/PTX pairs. Each sample uses a fresh process and context with eager loading and the CUDA disk cache disabled. Read `raw.csv` for individual observations and `summary.json` for medians and ranges.

Module timing excludes process and context creation, file reading, transfers and allocation. Symbol lookup and launch plus synchronization are measured separately. The Nsight capture and four sanitizer passes run outside the timed pairs. Nsight reports device kernel duration separately from the host launch and synchronization interval. Compiler, device, driver API version, image sizes and hashes accompany the measurements.

## Engineering decision

For a B200 service with a tight first-request budget, compare module-load and lookup medians and ranges before choosing a deployment image. Set the acceptance threshold from the service startup budget, then repeat with its real module and normal cache settings. Favor including a native cubin if the PTX path spends material startup budget on JIT. Evaluate retaining PTX against the deployment's architecture coverage and artifact size requirements. This microbenchmark cannot establish warm-cache behavior, sustained kernel throughput, or compatibility with another GPU or driver.

## Validation status

The local CPU suite passed four tests covering integer wraparound and index terms, deterministic inputs, tail guard failures, and rejection of out-of-range lengths. CUDA compilation, B200 correctness, performance, sanitizers and Nsight captures are pending. No GPU measurements have been recorded for this project.
