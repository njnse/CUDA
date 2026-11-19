# cuBLAS GEMM precision tradeoff

Compare strict FP32 and TF32-enabled computation for dense matrix multiplication with FP32 input and output. Both paths call `cublasGemmEx` on the same column-major matrices with the same leading dimensions and default algorithm selection. The baseline uses `CUBLAS_COMPUTE_32F_PEDANTIC`. The other path uses `CUBLAS_COMPUTE_32F_FAST_TF32`, which permits the library to select TF32 Tensor Core kernels. [cuBLAS compute types and GemmEx](https://docs.nvidia.com/cuda/archive/12.8.1/cublas/index.html#cublasgemmex)

## Reproduce

Run inside a Slurm allocation exposing one NVIDIA B200 with CUDA 12.8 or later, cuBLAS, Python 3, Compute Sanitizer and Nsight Compute.

```sh
export RESULTS_DIR="$(mktemp -d)" CUDA_ARCH=sm_100
bash projects/gemm-precision/scripts/run.sh
```

The C++ host oracle accumulates dot products in FP64 directly from the original FP32 inputs. It checks every output on 1×1×1, 31×29×33 and 129×127×65 shapes and 64 evenly spaced positions in each benchmark matrix. Both paths check a guard immediately beyond the output and the runner checks rejection of malformed and out-of-range inputs. Inputs use deterministic seeds 2027 and 2028. Each dot product must satisfy an absolute tolerance of `1e-5 + s × sum(abs(A_i × B_i))`, where `s` is `1e-6` for strict FP32 and `1e-3` for the TF32-enabled path. Raw records include maximum absolute error and relative L2 error over the checked positions.

The benchmark uses square matrices of order 256, 2,048 and 4,096. Five pairs per shape alternate mode order. Each sample averages six GEMMs after three warmups. CUDA event timing excludes allocation, transfers, host verification and library initialization. Effective throughput uses `2 × M × N × K` operations. Nsight records one kernel duration separately at order 2,048, and sanitizer checks run on the 31×29×33 case.

For a deployment decision, compare latency and numerical error at production shapes. Set the error limit from the application before tuning and repeat with representative input distributions. Errors on the sampled outputs do not bound errors on all large-matrix outputs or on ill-conditioned inputs.

## Measured results

The November 18 run used one NVIDIA B200, native `sm_100`, driver 580.178.04, CUDA 12.8.93 and cuBLAS 12.8.4. Native build, six full-matrix oracle comparisons, three invalid-input checks, 30 benchmark observations, all four sanitizer tools for both modes and both Nsight captures passed.

| Square matrix order | Strict FP32 median | TF32-enabled median | Speedup | Strict FP32 relative L2 | TF32-enabled relative L2 |
|---:|---:|---:|---:|---:|---:|
| 256 | 8.12 µs | 6.06 µs | 1.34× | 2.19e-7 | 2.13e-4 |
| 2,048 | 307.20 µs | 29.53 µs | 10.40× | 7.09e-7 | 2.76e-4 |
| 4,096 | 2,160.98 µs | 178.69 µs | 12.09× | 8.93e-7 | 2.49e-4 |

Errors are the largest relative L2 value across the five observations for each mode and shape, computed on the same 64 output positions. At order 4,096, measured effective throughput was 63.6 TFLOP/s in strict FP32 and 769.2 TFLOP/s with TF32 enabled. At order 2,048, Nsight captured a SIMT kernel for strict FP32 and a TF32 tensor-op kernel for the permissive mode. The latency and error differences make the fast mode a candidate for large GEMMs when representative validation meets the application error budget. The data cover one GPU and one deterministic input distribution.

[Raw timings, variability, tool logs and source hashes](results/automated-nvidia-b200/latest/)
