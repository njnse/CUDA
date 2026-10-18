# Should an example default to pinned buffers and two streams?

This experiment asks a concrete API/workflow question: does a two-stream chunked pipeline justify its additional pinned memory, buffer lifetime requirements and synchronization complexity for a workload that must copy complete outputs back to the CPU?

The source establishes ordering and storage requirements. It does not establish measured overlap or a performance gain. Pinned host memory supports the intended asynchronous transfer behavior, but allocation/registration can be expensive and pinned memory consumes a constrained system resource. The two-stream path also doubles device staging buffers relative to the active serial topology. There is no evidence yet about application adoption or developer preferences; no customer interviews have been conducted.

Use two separate comparisons. Pageable versus pinned serial evaluates host-memory choice with the same stream ordering. Pinned serial versus pipeline evaluates the scheduling topology with the same memory type. Both comparisons require matching total length, chunk size, kernel, host thread count, measurement scope and build. This avoids attributing all differences to streams.

A candidate acceptance rule for a developer example is exact correctness under tail/slot reuse, passing available sanitizers, complete raw samples, timing dispersion, and explicit lifetime guidance. A performance default additionally requires repeated allocation-level measurements showing useful end-to-end gains for intended workload sizes, including setup cost when buffers are short lived. The present samples do not provide a confidence interval across allocations or devices.

If pipelining helps only large batches or one chunk size, expose it as a measured option rather than a universal default. If no improvement is observed, retain the simpler serial path as the documented recommendation for that tested regime. If a timeline shows no overlap, report it directly and investigate queue topology, copy-engine capability, transfer size and host submission overhead before adding more streams.

Current recommendation is deferred until real worker evidence exists. A later recommendation must cite the matched raw samples and report setup exclusions, variability and device/toolkit scope. No universal chunk size, speedup, PCIe utilization claim or cross-device default follows from the implementation alone.
