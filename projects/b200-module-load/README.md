# CUDA module load: cubin versus PTX

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

For deployments that need low first-load latency on B200, include a native cubin when the startup budget cannot absorb the measured PTX compilation cost. Recheck with the production module and normal cache settings before deciding whether to ship PTX for architecture coverage. This run used a single small kernel, one device and one driver version with disk caching disabled.

## Validation status

Both images compiled with CUDA 12.8.93 and passed all size, guard and missing-symbol checks on NVIDIA B200. Four CPU tests, four Compute Sanitizer tools per image and both Nsight captures passed.

| Image | Median module load | Range across nine samples | Median launch and synchronization |
|---|---:|---:|---:|
| Native cubin | 79.4 µs | 73.4–82.9 µs | 34.3 µs |
| PTX | 12.005 ms | 11.298–12.398 ms | 35.8 µs |

The measured module-load median was about 151 times higher for PTX. Launch and synchronization were much closer after module loading. See the [raw data and tool logs](results/automated-nvidia-b200/latest/).
