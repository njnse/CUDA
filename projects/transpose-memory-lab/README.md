# Transpose memory lab

**Question:** For a row-major FP32 matrix, when does a tiled transpose justify its extra shared memory and synchronization, and what evidence should a CUDA example expose to developers?

Four paths isolate useful effects: coalesced copy (bandwidth context), naive transpose (strided stores), 32-by-32 shared-memory transpose, and a padded 32-by-33 shared-memory transpose. Copy is not a transpose implementation and is not a correctness competitor. All transpose variants implement `out[col * rows + row] = in[row * cols + col]` out of place. In-place operation, arbitrary strides and overlapping pointers are outside this executable's contract.

## Build and validate

Build with CUDA 12.4 or a compatible toolkit. Run GPU commands on a supported GPU:

```bash
bash projects/transpose-memory-lab/scripts/build.sh
python3 -m unittest discover -s projects/transpose-memory-lab/tests -p 'test_*.py' -v
# The following are GPU-allocation commands:
projects/transpose-memory-lab/build/transpose_lab --diagnose
projects/transpose-memory-lab/build/transpose_lab --verify
projects/transpose-memory-lab/build/transpose_lab --benchmark 4093 2051 15 > measured.csv
python3 projects/transpose-memory-lab/scripts/analyze.py measured.csv
```

The default target is `sm_89` for L4. Set `CUDA_ARCH=sm_XX` to a target supported by your compiler and GPU. The build script does not request an allocation.

## Slurm reproduction

From the repository root:

```bash
# Validate scheduling feasibility without creating a job:
sbatch --account=YOUR_ACCOUNT --test-only projects/transpose-memory-lab/scripts/slurm.sbatch
sbatch --account=YOUR_ACCOUNT projects/transpose-memory-lab/scripts/slurm.sbatch
```

Supply your site's account, QOS, GPU constraint and CUDA environment. This example requests one GPU, two CPU cores, 4 GiB memory and 15 minutes. Adapt the `l4` constraint to your site. It embeds no connection settings. The script builds, verifies, benchmarks four shapes and runs four Compute Sanitizer tools. Results go under `results/local/slurm-JOBID/`, excluded from Git until deliberately selected and sanitized.

## Evidence and scope

- CPU: hand-calculated known-answer test, rectangular round trips and invalid input rejection.
- GPU: exact element equality against the independent CPU oracle; fixed-seed signed finite data; NaN-poisoned outputs; input immutability; prefix/suffix guards; ten boundary/rectangular/square shapes.
- Memory safety: guarded output is useful but not a substitute for Compute Sanitizer.
- Timing: ten warmups, fifteen samples by default, 100 launches/sample below one million elements and ten otherwise; CUDA events on the default stream; raw samples plus median/min/max. Allocation, copies, verification and CPU work are excluded. Launch gaps remain part of the event interval.
- Effective GB/s counts one read plus one write (`2 * rows * cols * sizeof(float)`). It is logical traffic, not measured DRAM traffic; cache-resident results can exceed external-memory bandwidth.
- Fixed variant order and same-buffer replay can bias clock/cache conditions. Rotate order and use a working set larger than cache before making stronger hardware claims. No library baseline or cross-device claim is provided.

See [validation](results/validation.md), [tutorial](docs/developer-tutorial.md), [product brief](docs/product-brief.md) and [profiling protocol](docs/profiling.md).

## Measured configuration and results

NVIDIA L4 (compute capability 8.9, 23,034 MiB visible memory), driver 580.178.04, CUDA Toolkit 12.4.131, runtime 12.4, GNU C++ 11.5.0; `-std=c++17 -O3 -lineinfo -arch=sm_89`. Block shape: 32 by 8. Shared tiles: 32 by 32 or 32 by 33. Medians below are milliseconds per launch from 15 uninstrumented samples.

| Rows x columns | Naive | Tiled | Padded | Naive / padded |
|---|---:|---:|---:|---:|
| 31 x 33 | 0.002486 | 0.002171 | 0.001792 | 1.39x |
| 1024 x 1024 | 0.053533 | 0.012362 | 0.005453 | 9.82x |
| 4093 x 2051 | 0.426669 | 0.285491 | 0.286106 | 1.49x |
| 4096 x 4096 | 0.856986 | 0.585114 | 0.585728 | 1.46x |

Ten verification shapes and four sanitizer modes passed. Separate Nsight Compute profiling of 1024 by 1024 measured 1,015,808 shared-load bank conflicts in the unpadded tiled kernel and zero in the padded kernel. Larger measured shapes showed essentially tied tiled medians. Fixed order, cache residency and one-device coverage limit generalization. See [sanitized evidence](results/nvidia-l4-cuda12_4/).
