# From strided stores to a padded tile

## 1. State the data contract

Take a row-major matrix with `rows * cols` FP32 elements. Output has logical shape `cols * rows`. For the 2-by-3 input `[1,2,3; 4,5,6]`, the output is `[1,4; 2,5; 3,6]`. A transpose only rearranges values, so exact equality is appropriate for this finite dataset.

## 2. Inspect the naive path

The block has 32 threads along x and eight along y. A warp reads adjacent input columns. Its output addresses differ by `rows`, so stores are generally strided. Four iterations let the block cover a 32-by-32 tile without using 1,024 threads. Rectangular boundary predicates protect both dimensions.

## 3. Exchange ownership using shared memory

Coalesced input loads populate a shared tile. After every thread reaches the barrier, block coordinates swap for the output. Threads now read a transposed shared tile and write adjacent output columns. The barrier must remain unconditional: threads outside an edge tile's input bounds still participate.

For tail tiles, every shared element read by a valid output thread has a corresponding valid input load. The predicates are intentionally different before and after the block-coordinate swap. Tests include 31-by-33, 33-by-31 and 65-by-97 to catch accidental square-only logic.

## 4. Change only the padding

The two tiled kernels instantiate the same template with PAD=0 or PAD=1. In the common 32-bank, 4-byte-word model, reading a column of a 32-wide float tile maps lanes to the same bank. A stride of 33 distributes those accesses across banks. This is an explanatory model; only profiler counters on the actual device establish the observed transaction behavior. Shared memory grows from 4,096 to 4,224 bytes per block. The padded version is not guaranteed to win for every shape or GPU.

## 5. Run the experiment

Use the commands in the README inside an authorized GPU allocation. Run correctness first. Preserve raw CSV, GPU model, driver, toolkit, source hash and compilation target. Compare transpose paths at the same dimensions and timing scope. Use copy as bandwidth context. Report tiny-matrix latency separately from large-matrix throughput.

## 6. Explain a result honestly

If padding helps, check whether shared-memory conflict metrics support the explanation. If it does not, examine launch overhead, cache residency, timing variability and occupancy. A faster kernel is evidence for one workload and environment, not proof of broad application acceleration. Measure a real caller's end-to-end time before recommending a product change.

Reference: [NVIDIA CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html), sections on coalesced global access, shared-memory banks and effective bandwidth. This project implements the established transpose teaching pattern; it does not claim a novel algorithm.
