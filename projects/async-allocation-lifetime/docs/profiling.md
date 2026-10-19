# Inspect API cost and lifetime ordering separately

Authoritative samples are uninstrumented host wall lifetimes: allocation, one fill kernel, release and required completion. Context/pool creation, verification copies and pool-accounting queries are excluded. Three warmups, nine rotated samples and 20 lifetimes/sample are fixed. Repeat the entire worker run in a fresh output directory to assess allocation-level variation; sample IQR is not a cross-device confidence interval.

When available, the worker runs Compute Sanitizer memcheck, initcheck, racecheck and synccheck with `--error-exitcode 1` over the quick suite. Any tool failure fails the run. Unavailable tools remain unavailable. Event-handoff tests supply an actual ordered path; guards and zero pool usage do not establish absence of all lifetime bugs or replace sanitizer execution. Tool/runtime compatibility errors should be investigated, not relabeled success.

An available Nsight Compute attempt profiles one `fill_pattern` kernel from `--profile`, separately from timing. It is an instrumentation-readiness probe, not a measurement of allocator API overhead. Exit zero is recorded as a completed attempt, and actual counter output must be checked before claiming a capture. Denied/unsupported/timed-out profiling is recorded explicitly.

Use a separate Nsight Systems CUDA timeline to examine the stream-ordered allocation/free APIs and both event dependencies in `pool_lab --profile`. Keep binary reports in private build/result storage. Inspect producer ready, consumer wait, kernel/result copy, consumer done, producer wait and free ordering. API duration summaries can help explain an observed latency difference, but instrumented application timing must not replace the uninstrumented CSV.

Publish sanitized text summaries only: source hashes, shape, API options, GPU model, toolkit/driver/compiler, timing scope and variation. Omit host/account names and personal absolute paths. Pool reserved/used bytes are runtime accounting snapshots; they do not show page migration, external process memory pressure or total board utilization.
