# Accumulator precision as an API decision

The product question is whether an application should expose accumulator precision explicitly, choose it automatically, or keep an FP32 default. The implementation provides a controlled experiment and a library comparator; no workload interviews, performance measurements or user preferences have been collected for this project.

The established facts are mathematical and structural. Addition is order dependent in finite precision. FP64 partials double their storage footprint relative to FP32, and the two custom kernels use the same grid and shared-memory reduction structure. The target device's arithmetic throughput, launch overhead, cache residency and CUB's internal choices determine the observed cost; the source alone does not quantify that cost.

A candidate requirement is an explicit accumulation policy with documented output precision, supported input range and an error budget. Acceptance evidence should include exact known answers, cancellation cases, tails, sanitized runtime metadata, raw matched latency samples and sanitizer results. A generic `gamma_n` bound is a correctness envelope, not a useful application accuracy promise. Downstream applications must provide representative distributions and tolerances before selecting a policy.

Use the following decision process after worker evidence exists:

1. Reject paths that fail the input/memory/numerical contract; do not widen tolerances to hide a defect.
2. Compare FP32 tree and CUB at the same shape/dataset, acknowledging different summation order.
3. Compare FP64 with both FP32 choices using absolute and scaled error plus both timing scopes and IQR.
4. If a workload exceeds its error budget in FP32, assess whether FP64 meets that budget at acceptable measured cost. If not, evaluate compensated or reproducible summation as a separate experiment.
5. Repeat across relevant devices and toolkit versions before setting a universal default.

Current recommendation: retain an explicit precision choice in the experiment and defer a default recommendation until real worker data arrives. No cross-device accuracy/performance claim is justified. The experiment does not evaluate training convergence, scientific-simulation error propagation, mixed-precision inputs or library implementation stability across releases.
