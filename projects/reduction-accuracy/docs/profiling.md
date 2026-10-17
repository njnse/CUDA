# Profiling protocol

Keep benchmark and instrumented executions separate. The worker records uninstrumented `raw.csv` before any sanitizer/profiler execution. Five warmups, nine rotated samples, 30 reductions/sample, seed 2027 and both timing scopes are fixed. Re-run the entire sweep in a fresh output directory to assess allocation-level variability; sample IQR alone is not a confidence interval across machines or runs.

When Compute Sanitizer is installed, the worker runs memcheck, initcheck, racecheck and synccheck with `--error-exitcode 1` over 18 directed cases. Any tool failure makes the run fail. An unavailable tool is recorded as unavailable. Partial arrays have no explicit guard padding, making actual sanitizer checks especially relevant.

When `ncu` is installed, the worker requests one custom `reduce_tree` launch with `gpu__time_duration.sum`, using the small `--profile` mode and text details output. The application separately checks all three paths without instrumentation. Profiling permission/version failures are recorded with their exit code; they do not establish successful counter collection and do not invalidate otherwise passing core checks. This focused capture is instrumentation readiness evidence, not a complete explanation of FP32/FP64 throughput differences.

To investigate a measured difference, collect both passes of each custom path and the CUB launches separately, keeping count, input and build fixed. Inspect arithmetic instruction mix, register use, occupancy, shared memory and memory traffic using metrics supported by the installed Nsight Compute version. Attribute bottlenecks only after counters support the claim. Never use instrumented durations in the uninstrumented latency table.

The runner records queried GPU model/driver/memory, toolkit and compiler versions, exact build flags, runtime/driver API versions, source SHA-256 hashes and source commit when available. It omits host/account identities and replaces operational absolute paths in captured tool messages with neutral markers. Numerical data is unchanged. Raw application CSV has no operational identifiers; binary profiler reports are not published.
