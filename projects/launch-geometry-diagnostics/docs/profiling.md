# Compare predictions with observations

Keep uninstrumented timing authoritative: five warmups, nine rotated samples, 20 launches/sample, seed 2027, matched length and round count. Stream-event intervals include possible host submission gaps. Synchronized host intervals include submission and completion overhead. Allocations, transfers, poisoning, checks and occupancy queries are excluded from both. Same-buffer replay can favor cache residency.

When available, the worker runs memcheck, initcheck, racecheck and synccheck with a nonzero error exit code over `--quick`: 16 positive payload cases. The expected invalid launch is intentionally absent from this mode. Its immediate error classification/clearing/recovery is tested separately by normal `--verify`; do not infer that the negative API call passed a sanitizer test.

An available Nsight Compute attempt captures `gpu__time_duration.sum` for one `transform` launch from the small `--profile` mode, separately from timing. Exit zero is a completed attempt; actual metric output must exist before claiming collection. Denied/unsupported/timed-out profiling is recorded explicitly. This readiness probe does not measure achieved occupancy.

To investigate actual policy differences, profile identical shapes/rounds with each block/grid choice separately. Use installed-version-supported launch statistics, registers, active warps and instruction/cache metrics. Compare achieved occupancy with the recorded resource-predicted bound, accounting for too few blocks and tail lanes. Do not invent hardware metrics or mix instrumented durations into the uninstrumented summary.

The diagnostic log records kernel registers, local/static shared bytes, recommended block/minimum grid and actual device/runtime information. The CSV records launch geometry and active-block predictions. Repeat the entire uninstrumented run in a fresh results directory to assess run-level variability. Sample IQR is not a confidence interval across devices or driver versions.

Publish sanitized text artifacts with shape, seed, versions, build options, source hashes and timing scope. Keep binary profiler reports private and omit host/account names and personal absolute paths. A focused counter capture does not establish causal attribution for every timing difference.
