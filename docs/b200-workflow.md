# B200 experiment workflow

New GPU experiments exclusively target NVIDIA B200, native `sm_100`, with CUDA 12.8 or newer. Historical L4 data is preserved and is not evidence for B200 performance.

The intended server worker polls the main branch and runs one changed project inside a B200 allocation. Migration of the existing persistent worker is pending specific approval; its configuration has not changed. The first B200 experiment was run manually within an existing allocation. New scripts must honor `RESULTS_DIR` and `CUDA_ARCH=sm_100`, reject other devices, return nonzero on validation failures, and retain raw evidence. CPU-only development can proceed without a GPU; mark GPU results pending until executed.

Rotate concrete questions across HBM bandwidth/cache effects, Tensor Core GEMM with library baselines, precision/accuracy tradeoffs, asynchronous movement and pipelining, graph/launch overhead, and compiler/runtime/driver compatibility. Use problem sizes large enough to expose the intended bottleneck, within available memory. GPU utilization alone is not proof of useful work or peak performance. Use measured bandwidth, arithmetic intensity, throughput, latency and Nsight counters to explain limits. Never promise 100% of every resource on every workload.

Each daily topic includes source, independent oracle, tail/error tests, baseline, benchmark, tutorial, product decision/acceptance criteria and validation record. Check the daily ledger and current main to avoid duplicate daily projects. Publish implementation first only if the GPU stage is clearly pending; update with measured evidence when available.

Public evidence lives under `projects/<topic>/results/automated-nvidia-b200/`. Preserve logs, CSV, compiler/device metadata and source provenance; remove account names, hostnames and private paths before publication. Profiler runs are separate from benchmark samples.

References: [Blackwell compatibility](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/contents.html), [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).
