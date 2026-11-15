# Histogram under atomic contention

Compare a 256-bin global atomic baseline with block-local shared memory and a global merge. Both kernels read the same byte stream with 256 threads per block and eight blocks per SM. The shared-memory variant reduces global updates for skewed inputs, while its initialization, barriers and merge can cost more on smaller workloads.

From a Slurm allocation exposing one NVIDIA B200, with CUDA 12.8+, Python 3, Compute Sanitizer and Nsight Compute:

```sh
export RESULTS_DIR="$(mktemp -d)" CUDA_ARCH=sm_100
bash projects/b200-histogram/scripts/run.sh
```

The fixed seed is 2027. The uniform scenario uses pseudorandom bytes. The hot scenario routes approximately 15/16 of inputs directly to bin 7. CPU checks cover empty inputs, the first seeded value and skew. GPU checks compare all bins against a Python reference at zero, tiny, block boundary and nonmultiple sizes. The executable checks an output guard and rejects invalid arguments.

The benchmark alternates variant order across five pairs at 1, 16 and 128 MiB, for each distribution. Each observation averages five kernels after two warmups. CUDA events cover kernel execution including the block merge. Host transfers, allocation and output clearing are outside the interval. The variants must agree on every timed histogram. Review raw observations, median and range before drawing a performance conclusion. The four sanitizer tools and separate Nsight captures run outside benchmark timing. Nsight captures kernel duration for both distributions under a separate run.

A product decision should use the observed input distribution and a latency budget. The shared variant won every tested batch size and distribution on this GPU, so it is the candidate to take into end-to-end tests. Repeat with smaller batches, representative production inputs and transfers included before setting a dispatch threshold or latency target.

## Measured results

| Distribution and input size | Global atomic median | Block-local median | Ratio |
|---|---:|---:|---:|
| Uniform, 1 MiB | 0.241 ms | 0.0167 ms | 14.4× |
| Hot bin, 1 MiB | 0.661 ms | 0.0151 ms | 43.8× |
| Uniform, 128 MiB | 30.09 ms | 0.202 ms | 148.7× |
| Hot bin, 128 MiB | 86.07 ms | 0.202 ms | 426.5× |

On NVIDIA B200 with CUDA 12.8.93, the direct global-atomic kernel slowed substantially under the hot-bin workload, while the block-local kernel had similar medians at 128 MiB for both distributions. These ratios compare this direct baseline with this block-local implementation. CUDA build, 24 GPU oracle comparisons, three argument checks, four sanitizer tools per mode, and four Nsight captures passed. See [raw measurements and validation evidence](results/automated-nvidia-b200/latest/).
