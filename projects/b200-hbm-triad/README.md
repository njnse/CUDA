# B200 streaming triad

Question: does aligned float4 access improve `c = a + 2*b` over scalar coalesced access on B200, and how does the answer change beyond cache capacity?

Both kernels use the same 256-thread blocks and 16 blocks per SM. float4 handles tails explicitly. A deterministic host FP64 oracle checks every output and four guard elements for each variant. Inputs are binary fractions, so exact comparison is appropriate. Tests cover tiny sizes, vector boundaries, odd tails and a million-element case. Invalid CLI input must fail. This project measures memory access, not Tensor Core throughput.

Inside a pre-existing B200 allocation, with a compatible CUDA 12.8+ environment:

```bash
RESULTS_DIR=/absolute/new/results CUDA_ARCH=sm_100 bash projects/b200-hbm-triad/scripts/run.sh
```

The runner rejects other GPUs/architectures and existing evidence. It builds native sm_100, verifies correctness, collects 9 samples with 10 repetitions after 5 warmups, runs four Compute Sanitizer tools, and captures Nsight Compute counters separately. Missing/failed sanitizer or profiler stages return nonzero. Host memory consumption at the largest shape is approximately 3 GiB plus overhead; device arrays total approximately 3 GiB.

Sizes span 1M to 256M floats. Logical bandwidth is `12*n / time` (two reads and one write); cache reuse can make small-shape bandwidth exceed device-memory bandwidth. This is not a DRAM byte counter. Rotated variant order reduces ordering bias. CUDA event timing excludes allocation, transfers, oracle and profiler. Report median and min/max rather than a single best sample. Inputs are deterministic; no random seed is used.

Tutorial: first inspect correctness and guard checks, then compare scalar and float4 medians at each size, and finally inspect DRAM/SM throughput counters. At 64M floats the profile uses about 768 MiB of logical traffic per kernel. Use device-reported L2 size to assess cache effects. A faster vectorized kernel can reduce instruction overhead without increasing the theoretical memory ceiling. Nsight replay/cache control and clock policy can affect counters; profiler timings must not replace benchmark timings.

Product decision: recommend float4 only if large-shape medians improve consistently beyond observed run variation, correctness and all sanitizer checks pass, and profiling supports the claimed bottleneck. Otherwise retain the simpler scalar path. Limitations: one GPU, one deterministic input family, fixed launch geometry, no power/clock control, no end-to-end transfer measurement. A second independent run is required before making a robust speedup recommendation.

Validation: GPU execution pending at implementation commit. Read `results/automated-nvidia-b200/latest/` for actual evidence when published. Historical results from other projects are not evidence for this experiment.
