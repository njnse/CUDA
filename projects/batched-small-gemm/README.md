# Batched small matrix multiplication

Compare two ways to multiply many independent small FP32 matrices on one CUDA stream. The baseline calls `cublasSgemm` once for each matrix pair. The batched variant calls `cublasSgemmStridedBatched` once for the same inputs. Both use tightly packed column-major matrices, constant strides, nonoverlapping output matrices, the same handle configuration, and pedantic math. [cuBLAS strided batched GEMM reference](https://docs.nvidia.com/cuda/archive/12.8.1/cublas/index.html#cublas-t-gemmstridedbatched)

## Reproduce

Use a Slurm allocation exposing one NVIDIA B200 with CUDA 12.8 or later, cuBLAS, Python 3, Compute Sanitizer and Nsight Compute.

```sh
export RESULTS_DIR="$(mktemp -d)" CUDA_ARCH=sm_100
bash projects/batched-small-gemm/scripts/run.sh
```

The host oracle computes integer dot products from the original inputs. Both APIs must return exactly the same answer because the generated inputs and the maximum sum of absolute terms fit within exact FP32 integer range. The seeds are 2027 and 2028. Verification checks all outputs for four cases, including non-square and odd dimensions, and checks a canary after the batch output. Four invalid inputs must fail. Timed cases verify up to 257 evenly spaced outputs per call sequence and the end canary.

Each CUDA event interval covers three complete batches after three batch warmups, then divides by three. Five pairs per case alternate variant order. The timing includes stream execution and any gaps while the host submits work. CUDA memory allocation, transfers, handle setup and correctness checks are outside that interval. Nsight captures the first GPU kernel at order 32 and batch 64 separately to show dispatch geometry. A single captured kernel in the loop is one part of that baseline, so compare full batches using the event timings.

The strided API is a candidate when the workload can present uniform matrix shapes and strides. Evaluate any packing or layout conversion cost and batching delay in the application. This benchmark does not cover irregular sets, multiple submission streams, graph replay, or time waiting to form a batch.

## Measured results

The November 26 run used one NVIDIA B200, driver 580.178.04, CUDA 12.8.93, cuBLAS 12.8.4, and native `sm_100`. CUDA build, eight complete CPU-oracle and canary comparisons, four invalid-input checks, 70 benchmark observations, all four Compute Sanitizer tools on both modes, and both Nsight captures passed.

| Square matrix order | Matrices per batch | Loop batch median | Strided batch median | Loop / strided |
|---:|---:|---:|---:|---:|
| 16 | 1 | 6.560 µs | 6.816 µs | 0.96× |
| 16 | 64 | 285.056 µs | 8.021 µs | 35.54× |
| 16 | 1,024 | 4,305.344 µs | 8.789 µs | 489.84× |
| 32 | 64 | 358.699 µs | 8.619 µs | 41.62× |
| 32 | 1,024 | 5,452.555 µs | 8.971 µs | 607.82× |
| 64 | 64 | 357.376 µs | 8.437 µs | 42.36× |
| 64 | 1,024 | 5,472.565 µs | 20.821 µs | 262.83× |

The two API calls had similar medians at batch one. At order 32 and batch 64, Nsight captured the same kernel symbol with grid `(1, 1, 1)` for the first looped GEMM and `(1, 1, 64)` for the batched GEMM. The profiled durations were 4.928 µs and 5.056 µs, respectively. The large batch speedups compare a single-stream loop of tiny calls with the strided API on prepacked inputs. They do not estimate the speedup for a single GEMM or across a pipeline with layout conversion.

[Raw timings, sample ranges, profiler captures, validation logs and source hashes](results/automated-nvidia-b200/latest/)
