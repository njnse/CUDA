# Elementwise kernel fusion

Fuse an affine transform and a second operation that combines the intermediate with a bias. The baseline uses two kernels on the same stream and materializes the intermediate. Both variants use a 256-thread block and implement the same unsigned 32-bit output for every input. The fused variant saves one intermediate write and read per element and removes one launch.

## Reproduce

Use a Slurm allocation exposing one NVIDIA B200, CUDA 12.8+, Python 3, Compute Sanitizer and Nsight Compute. From the repository root:

```sh
export CUDA_ARCH=sm_100 RESULTS_DIR="$(mktemp -d)"
bash projects/elementwise-fusion/scripts/run.sh
```

The runner verifies both variants with a host oracle that uses wide arithmetic and explicit modulo reduction. It checks all returned values and a word past the logical output for sizes 1, 255, 256, 257 and 65,537. It also checks rejection of invalid modes, lengths and repetition counts. Inputs are deterministic with offset 2027. The same oracle runs for every benchmark process.

The benchmark alternates the order of the two implementations across five pairs per size. Each CUDA event interval spans six repetitions after three warmups. Transfers, allocation and host verification are outside the interval. Inspect the median and range alongside the raw pairs. Nsight captures the measured kernels in separate processes and records device duration and DRAM read/write bytes. For the two-kernel baseline, add the per-kernel byte counters.

Fusion is a deployment candidate when both operations run back-to-back on the same input and no other consumer needs the intermediate. Compare against representative workloads and check reuse or resource pressure before replacing independently callable operations.

## Measured results

On NVIDIA B200 with CUDA 12.8.93, native `sm_100` compilation, ten full-output oracle comparisons, guard and invalid-input checks, four Compute Sanitizer tools per variant, 30 timing samples and Nsight Compute captures passed.

| Elements | Split median | Fused median | Speedup |
|---:|---:|---:|---:|
| 65,537 | 4.09 µs | 2.46 µs | 1.66× |
| 4,194,304 | 20.99 µs | 10.92 µs | 1.92× |
| 67,108,864 | 387.43 µs | 206.34 µs | 1.88× |

At the largest size, Nsight recorded 1.250 GB of combined DRAM reads and writes across the two baseline kernels and 0.779 GB for the fused kernel, about 38% less traffic. These are separate profiler captures, and the observed reduction depends on cache and replay behavior. At small sizes, fewer launches may contribute to the reduction, but the experiment does not isolate that contribution.

[Raw timings, tool logs and source hashes](results/automated-nvidia-b200/latest/) record the run.
