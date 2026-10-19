# Make the last access happen before free

Start with the same-stream path. `cudaMallocFromPoolAsync`, initialization, kernel, result copy and `cudaFreeAsync` are ordered in the producer stream. A CPU return from the allocation API is not permission for arbitrary streams to access the pointer. The synchronization at the end establishes completion before the CPU inspects verification output or queries live allocation bytes.

The handoff test makes both edges explicit. Recording `ready` after producer initialization and waiting on it in the consumer ensures that allocation and initialization precede the kernel. Recording `done` after the consumer's D2H copy and waiting in the producer ensures that free follows every consumer access. The pointer's numerical address is irrelevant. Even a reused address does not replace these dependencies.

Read `Allocator::correctness` in `src/pool.cu`. Pinned host output remains alive through the final producer synchronization. The consumer's copy completes before its done event, so producer completion also establishes when the CPU can read the result. Removing either event edge would violate the intended lifetime. This project does not run those invalid variants and mistake an occasional success for safety.

Next read `Pool`. It creates a private device pool and sets a release threshold for that pool only. Threshold zero allows the runtime to release unused backing at synchronization. A high threshold can preserve backing for reuse. Both policies should still report zero live allocation bytes after a synchronized free. Reserved and used memory answer different questions.

Run the CPU script to check known answers, exact wraparound, tails and invalid sizes. On a GPU worker, read `verify.log` before `raw.csv`. Compare the two async paths first to isolate retention, then compare with legacy complete lifetimes. Do not call the host-wall result a kernel duration or assume pool creation is amortized in a short-lived application.

Read reserved-byte ranges beside latency/IQR. A faster retained allocator can intentionally keep unused memory. Decide whether that tradeoff fits the application only after actual measurements. The benchmark's single outstanding allocation does not model a memory-intensive service with competing streams or processes.
