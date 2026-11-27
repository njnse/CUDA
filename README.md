# CUDA engineering portfolio

Reproducible CUDA experiments for developer education and evidence-based product decisions.

| Project | Engineering question | Evidence |
|---|---|---|
| [Batched small matrix multiplication](projects/batched-small-gemm/) | How much latency does a strided batched call save versus launching small GEMMs one by one? | GPU validation passed. A batch of 1,024 order-32 GEMMs fell from 5.453 ms to 8.971 µs on prepacked data. |
| [Dynamic shared-memory budget and block residency](projects/shared-memory-budget/) | How does shared-memory reservation affect residency and latency when the kernel does the same work? | GPU validation passed. At 192 KiB per block, achieved occupancy was 12.23% and latency was 4.38× the 256 B baseline. |
| [Shared-memory tiling for a five-point stencil](projects/stencil-tiling/) | Does staging neighboring cells reduce latency or DRAM reads for a single stencil sweep? | GPU validation passed. At 8,193×8,193, tiling took 8.4% longer with nearly identical profiled DRAM reads. |
| [Record layout and field selection](projects/record-layout/) | How do interleaved records and contiguous columns compare when reading one field or all eight? | GPU validation passed. At 16.8M records, a one-field column scan was 1.72× faster with 87.5% fewer profiled DRAM read bytes. |
| [cuBLAS GEMM precision tradeoff](projects/gemm-precision/) | How do strict FP32 and TF32-enabled GEMM trade latency for numerical error? | GPU validation passed. At order 4,096, TF32 was 12.09× faster with sampled relative L2 error of 2.49e-4. |
| [Warp-aggregated stream compaction](projects/warp-compaction/) | Does explicit warp aggregation improve unordered filtering at different selectivities? | GPU validation passed. Large-input medians differed by less than 0.14% across all tested selectivities. |
| [Elementwise kernel fusion](projects/elementwise-fusion/) | What does removing an intermediate buffer save for two consecutive transforms? | GPU validation passed. At 67.1M elements, fusion was 1.88× faster with 38% less profiled DRAM traffic. |
| [Histogram under atomic contention](projects/b200-histogram/) | How do skew and batch size affect global versus shared-memory atomics? | GPU oracle, benchmark, sanitizer and Nsight pass. Block-local median 0.202 ms on 128 MiB hot input. |
| [CUDA module load](projects/b200-module-load/) | How does the deployment image affect first module load? | GPU validation passed. Median module load was 79.4 µs for cubin and 12.0 ms for PTX with disk caching disabled. |
| [Streaming triad](projects/b200-hbm-triad/) | Does vectorized access improve streaming bandwidth beyond cache capacity? | Native sm_100, independent oracle, tail guards, rotated samples and profiler; see project validation |
| [Launch geometry diagnostics](projects/launch-geometry-diagnostics/) | Does occupancy-selected block size minimize measured latency? | GPU oracle, launch-error recovery, sanitizer and Nsight pass. Fixed block sizes led the measured sweep. |
| [Async allocation lifetime](projects/async-allocation-lifetime/) | How do pool retention and event-ordered lifetimes affect allocation cost and reserved memory? | Two NVIDIA L4 runs, exact oracle, event handoff, four sanitizers and 162 raw samples |
| [Streaming transfer pipeline](projects/streaming-transfer-pipeline/) | When do pinned buffers and two streams improve complete H2D/kernel/D2H throughput? | Worker-reported GPU pass and raw CSV; detailed tool logs not archived |
| [Reduction accuracy lab](projects/reduction-accuracy/) | What accuracy and execution cost follow from FP32 versus FP64 accumulation? | Worker-reported GPU pass, raw CSV and source hashes; detailed tool logs not archived |
| [CUDA Graph replay](projects/cuda-graph-replay/) | When does graph replay amortize setup, and when is fusion better? | Three implementations, exact oracle, rotated-order timing and reproducible GPU checks |
| [Transpose memory lab](projects/transpose-memory-lab/) | How do coalescing and shared-memory padding affect rectangular transpose? | Runnable tutorial, independent oracle, timing protocol, diagnostics and product acceptance criteria |

Each project separates source/build validation from actual GPU correctness and performance evidence. Never interpret an unexecuted benchmark as a result. The older `GPU-Computing` repository is separate and has not been migrated here.

## Working agreement

New GPU execution exclusively uses **NVIDIA B200** with native `sm_100` and a compatible CUDA 12.8+ environment. Previous L4 results remain historical. See the [B200 workflow](docs/b200-workflow.md) and [daily ledger](docs/daily-ledger.md).

- Preserve existing work; check the index before choosing the next topic.
- Pair implementation with correctness tests, a fair baseline, reproduction instructions, a developer tutorial and a product brief.
- GPU execution belongs on a Slurm compute allocation, not a login node.
- Publish only observed measurements with device, toolkit, input, timing scope and source provenance.

Next candidates: runtime/driver diagnostics, compiler regression minimization, JIT-versus-AOT compatibility, shared-memory resource limits. Avoid repeating a topic without a new engineering question.

The [CPU CI example](docs/cpu-ci-example.yml) is documentation, not an active workflow. CPU and GPU checks were executed directly; no GitHub Actions success is claimed.
