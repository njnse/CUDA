# Warp-aggregated stream compaction

Filter a deterministic array and return the indices that pass a threshold. The baseline uses a plain CUDA `atomicAdd` expression to reserve output positions. The handmade variant votes on selected lanes, reserves a contiguous range once per nonempty warp, broadcasts the base, and derives each lane's rank from the selection mask. Output order is unspecified for both variants.

## Reproduce

Run in a Slurm allocation that exposes a single NVIDIA B200 with CUDA 12.8 or later, Python 3, Compute Sanitizer and Nsight Compute.

```sh
export RESULTS_DIR="$(mktemp -d)" CUDA_ARCH=sm_100
bash projects/warp-compaction/scripts/run.sh
```

The host oracle uses wide arithmetic to recreate the predicate and checks full membership, duplicate indices and the guard after the output buffer. Correctness covers warp and block tails, zero and full selections, and malformed arguments. The data generator uses an initial offset of 2027.

The sweep uses 65,537 and 16,777,216 elements at target selectivities of 1.56%, 25% and 100%. Five pairs per case alternate variant order. Each observation averages three CUDA event intervals after three warmups. Counter reset, transfers, allocation and host verification are outside the timed interval. The Nsight duration captures run separately for one million elements at full selection.

NVCC can apply warp aggregation to a plain atomic expression on its own. This comparison tests the two source implementations under the same compiler flags. Instruction counts and hardware atomic traffic are outside the measured scope. [NVIDIA's warp aggregation guide](https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/)

## Measured results

The November 16 run used an NVIDIA B200, CUDA 12.8.93 and native `sm_100`. The build, 36 exact CPU/GPU membership comparisons, three invalid-input checks, four Compute Sanitizer tools per variant and separate Nsight kernel-duration captures all passed.

| Elements | Target selectivity | Plain atomic median | Manual aggregation median |
|---:|---:|---:|---:|
| 65,537 | 1.56% | 6.976 µs | 6.795 µs |
| 65,537 | 25% | 7.371 µs | 7.061 µs |
| 65,537 | 100% | 6.976 µs | 7.317 µs |
| 16,777,216 | 1.56% | 177.451 µs | 177.216 µs |
| 16,777,216 | 25% | 347.339 µs | 347.136 µs |
| 16,777,216 | 100% | 346.976 µs | 346.763 µs |

The sample ranges overlap in all six cases. At the larger size, median differences are below 0.14% at every selectivity. Keep the simpler atomic expression for this workload unless production measurements show a repeatable benefit from explicit aggregation. The duration data cannot establish whether the compiler generated warp aggregation for the baseline.

[Raw samples, ranges, validation logs and source hashes](results/automated-nvidia-b200/latest/)
