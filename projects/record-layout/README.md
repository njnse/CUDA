# Record layout and field selection

Compare interleaved records with one contiguous array per field. Both layouts store identical logical records of eight 32-bit integers. One kernel reads field 3 only. The other reads all eight fields and forms a weighted sum modulo 2^32. One thread computes one output in both layouts.

When neighboring threads read one field, the interleaved layout spaces those loads 32 bytes apart. The column layout makes the same loads adjacent. When all eight fields are consumed, cache lines fetched by the interleaved layout can supply several loads. DRAM bytes and latency can therefore tell different parts of the story. [CUDA guide to coalesced memory access](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)

## Reproduce

Use a Slurm allocation exposing a single NVIDIA B200, CUDA 12.8 or later, Python 3, Compute Sanitizer and Nsight Compute.

```sh
export RESULTS_DIR="$(mktemp -d)" CUDA_ARCH=sm_100
bash projects/record-layout/scripts/run.sh
```

The independent host oracle uses a closed-form expression from the logical row index, field constants and seed offset 2027, without reading the staged input. Every run checks each output and a guard immediately beyond the array. The verification cases cover four sizes including block tails and reject malformed command lines.

The sweep contains 4,099, 1,048,579 and 16,777,219 records with five alternating layout pairs per selection at each size. CUDA events time six launches after three warmups. Allocation, layout construction, transfers and host verification are outside the interval. Nsight captures duration and DRAM bytes read separately at the largest size. Compare medians and sample ranges from the raw CSV.

Before changing a pipeline layout, account for the cost of converting upstream rows, downstream consumers' access patterns, and how many kernels can reuse the converted representation. The experiment isolates GPU consumption of two already materialized layouts.

## Measured results

The November 19 run used one NVIDIA B200, driver 580.178.04, CUDA 12.8.93 and native `sm_100`. The build, 16 oracle and guard cases, three invalid-input tests, 60 timing observations, four sanitizer tools on each kernel variant and four Nsight captures passed.

| Records | Fields read | Interleaved median | Column median | Interleaved / column |
|---:|---:|---:|---:|---:|
| 4,099 | 1 | 3.067 µs | 2.907 µs | 1.06× |
| 4,099 | 8 | 3.280 µs | 3.072 µs | 1.07× |
| 1,048,579 | 1 | 4.752 µs | 4.267 µs | 1.11× |
| 1,048,579 | 8 | 10.581 µs | 4.949 µs | 2.14× |
| 16,777,219 | 1 | 85.845 µs | 49.840 µs | 1.72× |
| 16,777,219 | 8 | 127.147 µs | 87.547 µs | 1.45× |

At the largest size, Nsight measured 536,926,976 DRAM read bytes for interleaved records and 67,139,072 bytes for columns when reading one field, an 87.5% reduction. With all fields, both transferred approximately 537 MB from DRAM. The difference in all-field kernel time occurred with similar DRAM byte counts. These counters alone cannot separate instruction cost, cache behavior and scheduling effects.

The smallest-case ranges overlapped. For repeated single-field scans of large buffers, the measured latency saving is about 36 µs per invocation. Use the conversion cost and the number of scans to decide whether that saving can pay for an alternate layout in an end-to-end pipeline.

[Raw timings, variability, profiler captures, validation logs and source hashes](results/automated-nvidia-b200/latest/)
