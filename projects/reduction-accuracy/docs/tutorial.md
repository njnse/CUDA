# Understand summation before tuning it

Start with `[100000000, 1, -100000000, 1]`. Its mathematical sum is 2. FP32 cannot retain a unit increment next to 100000000, so changing addition order can change the result without any memory error. Relative error against a nearly zero total can be misleading. Keep absolute error, the sum of magnitudes and cancellation condition alongside latency.

Run `scripts/test_cpu.sh` from the repository-root command in the README. The hand-calculated cancellation test and a separate exact integer-units calculation establish the CPU oracle independently of a GPU tree. Inspect `include/reference.hpp` before assuming higher precision alone guarantees an oracle.

Read `reduce_tree` in `src/reduction.cu`. Each thread accumulates a grid-stride subsequence, writes a shared-memory lane and participates in a synchronized tree. The first pass produces one partial per block; the second sums those partials. Every thread reaches every barrier, including tail threads initialized to zero. Instantiating the same template with `double` changes registers, shared memory, partial storage and arithmetic throughput without changing the mapping.

The CUB path queries temporary storage once, allocates it once and uses the same stream as the custom paths. Its FP32 output is an appropriate production baseline for the FP32 tree. Its internal order need not equal the custom tree, so bitwise equality between FP32 implementations is not the contract.

On a GPU worker, run the README command with a fresh results directory. Read `verify.log` first; then inspect `raw.csv` and `summary.json`. Separate `positive` from `cancellation`: a small error on a well-conditioned input does not establish robustness under cancellation. FP64 equality here is specific to exact dyadic datasets and their bounded magnitudes, not a general promise of exact floating-point summation.

Only after uninstrumented correctness/timing should you open the sanitizer and profile logs. Guard checks detect some overruns; they do not detect every out-of-bounds partial access or synchronization fault. A profiler may be unavailable even when kernels execute. Its measurements must not replace the uninstrumented benchmark.
