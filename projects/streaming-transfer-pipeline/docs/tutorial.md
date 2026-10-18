# Build a pipeline without assuming overlap

A CUDA stream orders its own work. Two streams permit independent work, but they do not promise that copies and kernels overlap on a particular device or allocation. Start from `pinned_serial`, whose one stream orders H2D → kernel → D2H for every chunk. Its measured end-to-end time is the direct reference for `pinned_pipeline`.

Read `plan` in `include/reference.hpp`. Chunk offsets cover the input once; the last chunk uses `min(capacity, remaining)`. The slot number alternates between two independent device-buffer pairs. Both streams write different regions of the full host output. There is no host staging-buffer refill race because the full pinned input stays alive and unchanged throughout execution.

Read `Work::run` in `src/transfer.cu`. For a slot, reuse is safe because its next H2D follows its previous D2H in the same stream. The two streams have no dependency between them. Final stream synchronizations establish when the CPU may inspect the output. Removing the final synchronization would measure submission latency and could race verification; reusing one device buffer across both streams would introduce a cross-stream hazard.

The kernel uses unsigned integer arithmetic with exact modulo behavior. The independent CPU oracle widens multiplication to 64 bits and masks the result. Each output is poisoned with its expected value's complement. An omitted chunk, incorrect tail size or early result read should therefore fail exact full-array comparison rather than slip through a tolerance.

Run the CPU script before a GPU allocation. After the worker executes, read `verify.log` and `validation.json`, then inspect `raw.csv` and `summary.json`. Compare pinned serial and pipeline at the same total/chunk sizes. Compare pageable and pinned serial separately; otherwise a memory-type improvement can be mistaken for a streaming improvement.

Read the profiling protocol before attributing a measured improvement to overlap. A faster pipeline could reflect queueing, staging or cache effects. A timeline is needed to identify concurrent H2D, kernel and D2H activity. Re-run the uninstrumented benchmark to quantify variability after tracing, because instrumentation changes timings.
