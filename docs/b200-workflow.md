# GPU validation and publication workflow

New GPU experiments exclusively target NVIDIA B200, native `sm_100`, with CUDA 12.8 or newer. Historical L4 data is preserved and is not evidence for B200 performance.

Use an existing B200 allocation or submit a Slurm job from an isolated HPG checkout. A typical allocation requests one B200, four CPUs and 32 GiB host RAM with a 30-minute wall limit. Check for an active job before submitting. The server dispatcher polls published commits, so prepublication validation should use the unpublished checkout. Project scripts honor `RESULTS_DIR` and `CUDA_ARCH=sm_100`, reject other devices, return nonzero on validation failures, and retain raw evidence. Keep job logs outside RESULTS_DIR until the project finishes because runners may require an initially empty evidence directory.

Rotate concrete questions across HBM bandwidth/cache effects, Tensor Core GEMM with library baselines, precision/accuracy tradeoffs, asynchronous movement and pipelining, graph/launch overhead, and compiler/runtime/driver compatibility. Use problem sizes large enough to expose the intended bottleneck, within available memory. GPU utilization alone is not proof of useful work or peak performance. Use measured bandwidth, arithmetic intensity, throughput, latency and Nsight counters to explain limits. Never promise 100% of every resource on every workload.

Each daily topic includes source, independent oracle, tail/error tests, baseline, benchmark, tutorial, product decision/acceptance criteria and validation record. Check the daily ledger and current main to avoid duplicate daily projects. Run the complete HPG GPU validation from a local or HPG checkout before publishing the project and its measured evidence together. Put the GPU model in the execution environment and omit it from the project title.

Public evidence lives under `projects/<topic>/results/automated-nvidia-b200/`. Preserve logs, CSV, compiler/device metadata and source provenance; remove account names, hostnames and private paths before publication. Profiler runs are separate from benchmark samples.

References: [Blackwell compatibility](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/contents.html), [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).
