# Repeatability and execution evidence

The result survived two additional uninstrumented process runs: graph replay reduced eager wall time, while fusion remained fastest at every tested length. The three runs span two GPU allocations on the same NVIDIA L4 configuration. They are not a cross-device study.

## Execution configuration

- NVIDIA L4, compute capability 8.9, 23,034 MiB visible memory.
- Driver 580.178.04; CUDA Toolkit 12.4.131; runtime API 12.4.
- GNU C++ 11.5.0; C++17; CUDA flags: -O3 -lineinfo -arch=sm_89.
- One GPU, two CPU cores, 8 GiB host-memory allocation.
- Original full validation allocation: COMPLETED, exit 0, eight seconds.
- Supplementary allocation: COMPLETED, exit 0, seven seconds; includes two unprofiled benchmarks and one instrumented trace.
- Job elapsed time includes setup/build/validation; it is not a per-pipeline performance metric.
- Nsight Systems 2023.4.4.54; trace and CSV export both exited 0.

Account, server and connection identifiers are intentionally omitted from publication copies. Slurm completion and original private logs were re-read before creating this report.

## Repeated timing

There are 675 uninstrumented raw samples: 3 runs x 5 lengths x 3 implementations x 15 samples. The table is the range of the **three run-level medians**, not a min/max across all individual samples or a confidence interval. Speedup and break-even are paired within each run.

| Elements | Eager wall us | Graph wall us | Fused wall us | Paired Eager / Graph | Setup us | Break-even iterations |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 3.275-3.376 | 2.400-2.427 | 1.652-1.705 | 1.35-1.41x | 195.006-195.978 | 201-231 |
| 31 | 3.319-3.367 | 2.416-2.443 | 1.683-1.698 | 1.36-1.39x | 19.690-26.100 | 21-30 |
| 257 | 3.334-3.409 | 2.461-2.475 | 1.678-1.708 | 1.35-1.39x | 14.702-20.502 | 16-24 |
| 4096 | 3.322-3.399 | 2.546-2.575 | 1.670-1.689 | 1.29-1.33x | 16.255-24.738 | 20-34 |
| 1048576 | 12.147-12.242 | 11.472-11.544 | 6.191-6.215 | 1.06-1.06x | 18.418-22.104 | 28-32 |

Each run rotates implementation order across samples. Device intervals, synchronized host wall time and host submission time remain separate. All comparison values above use synchronized host wall time normalized per pipeline. Twenty warmups precede measurement; a sample batches 200 pipeline iterations for smaller lengths and 30 at 1,048,576 elements.

The largest input has only about a 6% graph advantage over eager execution, while fusion remains about twice as fast as eager. That is consistent with the fused path removing both a launch and temporary memory traffic. Hardware memory-transaction counters were not collected, so the latter mechanism is an implementation-level explanation rather than a measured counter result.

## Setup and amortization

Capture, instantiate and explicit graph upload costs are preserved in each run's checks log. The break-even estimate is ceil(total_setup_us / (eager_median_wall_us - graph_median_wall_us)). Each setup value is one observation per length per process. The first graph's setup includes one-time graph initialization effects; later shapes reuse the process context. Expected application replay count, shape/pointer changes and integration overhead must be considered before adopting graphs.

The interval above is an observed range of estimates, not a statistical interval. No inference about cold-process startup or application-wide amortization is made.

## Actual execution trace

The separate instrumented run captured node-level CUDA Graph activity. Its exported CUDA API and GPU kernel summaries show:

| Observation | Count |
|---|---:|
| Graph launch API calls | 12,555 |
| Ordinary kernel launch API calls, including capture calls | 37,675 |
| Executed affine kernels | 25,110 |
| Executed ReLU kernels | 25,110 |
| Executed fused kernels | 12,555 |
| Total executed kernels | 62,775 |

The counts include correctness passes, warmups and benchmark repetitions. There are 12,555 pipeline invocations per implementation across the five lengths. Eager execution accounts for 25,110 ordinary two-kernel launches, fusion for 12,555, and the five captures contribute ten additional kernel launch API calls. Graph replay still executes both captured kernels; it reduces host submission calls rather than fusing GPU work.

The trace supports that the intended execution paths actually ran. Instrumentation changes API costs and synchronization behavior. **No timings from profile-application-output.txt or the profiler summaries are used in the performance table.** The binary trace and SQLite export remain in private execution storage; sanitized text/CSV exports are versioned here.

## Accuracy and source identity

Every supplementary benchmark also checked all five input lengths against the CPU oracle, including full-array comparisons at 1,048,576 elements. Guards and input-immutability checks passed. The original memcheck, racecheck, initcheck and synccheck results remain available and report no errors; supplementary runs did not repeat these unchanged-source sanitizer tests.

Source hashes for main.cu, reference.hpp and build.sh match between the original and supplementary runs. Documentation and published evidence changed between these commits; measured source hashes match. See source-commit.txt and source-sha256.txt in each evidence directory. The earlier default-stream/nonblocking-stream ordering failure and correction are documented in ../docs/stream-ordering.md.

## Reproduce

Inside an existing compatible GPU allocation, from the repository root:

```bash
bash projects/cuda-graph-replay/scripts/build.sh
for repeat in 1 2; do
  projects/cuda-graph-replay/build/graph_lab --benchmark > "repeat-$repeat.csv" 2> "repeat-$repeat-checks.txt"
  python3 projects/cuda-graph-replay/scripts/analyze.py "repeat-$repeat.csv" > "repeat-$repeat-summary.csv"
done
nsys profile --trace=cuda --sample=none --cpuctxsw=none --cuda-graph-trace=node --output launch-trace projects/cuda-graph-replay/build/graph_lab --benchmark
nsys stats --report cuda_api_sum,cuda_gpu_kern_sum --format csv --output nsys-summary launch-trace.nsys-rep
```

Use uninstrumented CSV for performance and the separate trace for execution diagnostics. For the full sanitizer protocol, run scripts/run.sh with RESULTS_DIR pointing to private execution storage.

## Evidence index

- [Original timing and correctness](automated-nvidia-l4/latest/)
- [Additional raw samples, setup logs and profiler exports](repeated-nvidia-l4/)
- [Supplementary execution metadata](repeated-nvidia-l4/execution-metadata.json)
- [CUDA API trace summary](repeated-nvidia-l4/nsys-summary_cuda_api_sum.csv)
- [GPU kernel trace summary](repeated-nvidia-l4/nsys-summary_cuda_gpu_kern_sum.csv)
