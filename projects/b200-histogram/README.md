# B200 histogram under atomic contention

Compare a 256-bin global atomic baseline with block-local shared memory and a global merge. Both kernels read the same byte stream with 256 threads per block and eight blocks per SM. The shared-memory variant reduces global updates for skewed inputs, while its initialization, barriers and merge can cost more on smaller workloads.

From a Slurm allocation exposing one NVIDIA B200, with CUDA 12.8+, Python 3, Compute Sanitizer and Nsight Compute:

```sh
export RESULTS_DIR="$(mktemp -d)" CUDA_ARCH=sm_100
bash projects/b200-histogram/scripts/run.sh
```

The fixed seed is 2027. The uniform scenario uses pseudorandom bytes. The hot scenario routes approximately 15/16 of inputs directly to bin 7. CPU checks cover empty inputs, the first seeded value and skew. GPU checks compare all bins against a Python reference at zero, tiny, block boundary and nonmultiple sizes. The executable checks an output guard and rejects invalid arguments.

The benchmark alternates variant order across five pairs at 1, 16 and 128 MiB, for each distribution. Each observation averages five kernels after two warmups. CUDA events cover kernel execution including the block merge. Host transfers, allocation and output clearing are outside the interval. The variants must agree on every timed histogram. Review raw observations, median and range before drawing a performance conclusion. The four sanitizer tools and separate Nsight captures run outside benchmark timing. Nsight captures kernel duration for both distributions under a separate run.

A product decision should use the observed input distribution and a latency budget. Adopt the shared variant if its improvement exceeds sample spread on the production distributions and it stays within the latency budget on the smallest supported batch. For mixed inputs, measure a representative trace before adding dispatch logic. This experiment measures kernel time and cannot establish request latency with transfers or explain contention solely from the duration counter.

CPU tests and shell syntax check passed locally. CUDA compilation, device correctness, sanitizer, profiler and B200 timings are pending.
