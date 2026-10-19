# Expose retention as a bounded policy

The product question is whether a runtime example should enable retained allocator backing by default or make the policy explicit. The implementation establishes ordering, accounting and a reproducible measurement protocol. It does not yet establish a speedup, memory-pressure behavior or developer preference. No customer interviews have been conducted.

There are two distinct tradeoffs. Stream ordering can avoid a global-style allocation/free lifecycle but requires correct ownership across streams. Retaining backing can reduce repeated setup work but consumes reserved device memory even after all logical allocations are freed. Treating used and reserved bytes as interchangeable would hide that cost.

Acceptance criteria for this example are exact payload and guard checks, legal producer/consumer event dependencies, zero live pool bytes after synchronization, available sanitizer success and complete matched benchmark evidence. A default recommendation additionally needs latency dispersion and retained-memory ranges across representative lifetime lengths, multiple outstanding allocations, memory pressure and device/toolkit versions.

Compare `async_release` with `async_retain` to isolate the threshold. Compare both with `legacy` to evaluate legal complete-lifetime workflows. The legacy synchronization occurs before `cudaFree`; async free is queued before the completion synchronization. That difference is intrinsic to the APIs and is explicitly part of the question. The identical kernel, payload, padded allocation size and warmup/sample protocol prevent unrelated work reduction from being mistaken for allocator improvement.

Current recommendation: retain an explicit policy choice and defer a default until worker evidence exists. If retention helps, quantify its reserved-memory cost and separately measure pool construction for short-lived applications. If improvement disappears under pressure or another device, narrow the recommendation accordingly. Do not promise cross-stream safety without both dependency edges or interpret a single allocator address reuse as an API guarantee.
