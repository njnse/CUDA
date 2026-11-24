# Shared-memory tiling for a five-point stencil

Compare direct global loads with explicit shared-memory staging for one sweep of a two-dimensional stencil. Both variants use 32×8 threads per block and give the center a weight of four and its four axial neighbors a weight of one. Arithmetic wraps modulo 2^32. At the boundary, a missing neighbor repeats the center value.

The tiled kernel stages the block interior and its four halo strips. Every thread reaches the barrier, including threads outside a partial tile. Those threads load clamped source coordinates to supply valid neighbors of edge outputs. The corners of the shared array are unused. The direct baseline can benefit from hardware cache reuse across overlapping reads. [CUDA best practices for shared memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory)

## Reproduce

Use a Slurm allocation exposing one NVIDIA B200 with CUDA 12.8+, Python 3, Compute Sanitizer, and Nsight Compute.

```sh
export RESULTS_DIR="$(mktemp -d)" CUDA_ARCH=sm_100
bash projects/stencil-tiling/scripts/run.sh
```

The CPU oracle calculates each output using flat neighbor indices and a 64-bit sum before modulo reduction. Verification includes single-row and single-column grids, an exact tile, partial tiles, an output guard, and invalid command arguments. Both implementations use identical inputs generated with seed 2027.

The sweep uses 257×259, 2,049×2,051, and 8,193×8,193 cells. Five pairs alternate the variant order for each shape. Each CUDA event observation averages six identical sweeps after three warmups. Allocation, initialization, host verification and data transfers are excluded. Nsight captures kernel duration and DRAM read bytes separately for the largest shape.

Choose a variant using medians and sample ranges on representative grids. The comparison covers one tile size, one boundary rule, and single sweeps of a static buffer. Iterative time stepping, fusion across sweeps, and different tile sizes can change the balance between cache reuse, halo loading, and synchronization.

## Measured results

The November 24 run used one NVIDIA B200, driver 580.178.04, CUDA 12.8.93 and native `sm_100`. CUDA build, 12 full-output and guard cases, three invalid-input checks, 30 benchmark samples, all four Compute Sanitizer tools on both variants, and both Nsight captures passed.

| Grid | Direct median | Tiled median | Tiled / direct |
|---|---:|---:|---:|
| 257×259 | 3.077 µs | 3.093 µs | 1.01× |
| 2,049×2,051 | 11.269 µs | 13.653 µs | 1.21× |
| 8,193×8,193 | 222.720 µs | 241.488 µs | 1.08× |

On the largest grid, Nsight measured 268,510,976 DRAM read bytes for direct loads and 268,514,816 bytes for shared-memory tiling. Both figures are close to the 268,501,252-byte logical input. The tiled kernel took 8.4% longer in the CUDA event sweep. The DRAM byte counts support the interpretation that direct loads already benefit from reuse, but do not isolate the costs of barriers, address arithmetic or shared-memory accesses. The smallest sample ranges overlap. Use the direct kernel for this single-sweep workload unless representative grids show a repeatable improvement from staging.

[Raw timings, variability, profiler captures, validation logs and source hashes](results/automated-nvidia-b200/latest/)
