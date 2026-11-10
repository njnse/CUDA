# CUDA engineering portfolio

Reproducible CUDA experiments for developer education and evidence-based product decisions.

| Project | Engineering question | Evidence |
|---|---|---|
| [B200 streaming triad](projects/b200-hbm-triad/) | Does vectorized access improve streaming bandwidth beyond cache capacity? | Native sm_100, independent oracle, tail guards, rotated samples and profiler; see project validation |
| [Launch geometry diagnostics](projects/launch-geometry-diagnostics/) | Does occupancy-selected block size minimize measured latency? | Fixed baselines, exact oracle, launch-error recovery and worker runner; GPU validation pending |
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
