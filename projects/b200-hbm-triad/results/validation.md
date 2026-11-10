# B200 validation

Executed 2026-11-10 in one existing GPU compute allocation, native sm_100, CUDA 12.8.93, driver 580.178.04, NVIDIA B200 (183359 MiB, 148 SMs, 132644864-byte L2). This was an individual run, not a persistent-worker run. Source commit: implementation commit `217580d`; the exact source-file SHA256 values are archived.

- Native CUDA compilation passed.
- Invalid CLI handling passed.
- Both GPU variants matched the independent FP64 host arithmetic oracle exactly, including tiny inputs and odd tails. Output guards remained intact. Benchmark shapes also passed the oracle before timing.
- Compute Sanitizer memcheck, initcheck, racecheck and synccheck returned success and zero errors/hazards.
- 72 raw timing samples across 4 sizes and 2 variants were collected. Nine samples per pair, each averaging ten launches after five warmups; variant order rotates. All eight archived medians were recomputed from raw CSV and matched. Source hashes were independently checked after retrieval.
- Nsight Compute captured both kernels separately from benchmark runs and returned success.

| Elements | Scalar median ms | float4 median ms | float4 logical TB/s |
|---:|---:|---:|---:|
| 1,048,576 | 0.0033856 | 0.003616 | 3.480 |
| 16,777,216 | 0.0450464 | 0.034 | 5.921 |
| 67,108,864 | 0.16775 | 0.127674 | 6.308 |
| 268,435,456 | 0.667408 | 0.508131 | 6.339 |

At the largest size, float4 was 1.313x faster. Its observed range was 0.507398–0.508813 ms; scalar ranged 0.667005–0.668845 ms. At the profiled 64M size, Nsight DRAM throughput was 80.41% for float4 and 61.09% for scalar, with SM throughput 7.88% and 21.91%, respectively. This supports a memory-access/instruction-efficiency explanation, but does not establish all hardware resources are saturated. The logical bandwidth calculation is not a hardware DRAM byte measurement.

Limitations: one run, fixed clock policy inherited from environment, no isolated clock/power study, no second-run recommendation, no Tensor Core work, no transfer timing. Local publication preserves sanitized raw logs and exact source hashes. The directory name denotes the B200 evidence convention and does not imply that the persistent worker was migrated.
