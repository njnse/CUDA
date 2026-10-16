# Product brief: make CUDA memory examples diagnosable

## Product question

Can a small CUDA sample help a developer distinguish indexing bugs, unsupported environments and workload-dependent performance without requiring an expert to interpret a single speedup number?

This is a technical experiment. No customer interviews, adoption metrics or competitive benchmarks have been collected.

## Implemented evidence

- Three transpose paths plus a separately labeled copy context path.
- Independent exact oracle, edge shapes, poisoned outputs, write guards and input-immutability checks.
- `--diagnose` reports the device, compute capability, runtime version and driver API version. Driver API version is not a complete compatibility assessment; retain `nvidia-smi` and the toolkit version too.
- Missing GPU / driver is distinguished by exit 77 and a diagnostic message; invalid CLI input and CUDA errors return 1.
- Raw timing samples, summary dispersion, correctness and four sanitizer reports are preserved alongside focused hardware counters.

## Observations versus hypotheses

Observed on NVIDIA L4 with CUDA 12.4.131 and driver 580.178.04: ten verification shapes and four sanitizer tools passed. At 1024 by 1024, padded transpose took a median 0.005453 ms versus 0.012362 ms for unpadded tiling and 0.053533 ms for naive transpose. Separate profiling measured 1,015,808 shared-load bank conflicts for unpadded tiling and zero for padding. The tiled paths were essentially tied at the two larger shapes.

The counters support the bank-conflict explanation for the profiled shape. Hypotheses still to test include cache and launch-overhead effects on other shapes/devices, and whether explicit diagnostics reduce support ambiguity. Developer usability has not been measured.

## Alternatives

| Alternative | Benefit | Cost / limitation |
|---|---|---|
| Single optimized snippet | Easy to read | Conceals baseline, tails and diagnostic failure modes |
| This executable experiment | Exposes causal comparison and testable failures | More code; intentionally narrow contiguous FP32 contract |
| Production framework / library operation | Appropriate application integration baseline | Requires a caller, layout agreement and dependency/version control |

No measured library comparison exists. A useful next evaluation could add a pinned-version library transpose baseline with identical input/output layouts and timing scope.

## Requirements and acceptance criteria

1. Reject malformed dimensions before device allocation; document limits and out-of-place semantics.
2. All GPU paths must match the oracle on every verification shape with intact guards and immutable inputs.
3. All four Compute Sanitizer modes must report success before labeling memory/synchronization validation complete.
4. GPU-unavailable behavior must return 77; successful build alone must never become a GPU correctness claim.
5. Every performance report must include raw samples, shape, timing scope, device, toolkit, build target, source identity and dispersion.
6. Any recommendation to change a public sample needs an actual GPU run and at least one developer usability check; no fabricated feedback.

## Recommendation and next decision

Keep padding as an explained teaching example and preserve the unpadded comparator. Report shape-specific measurements rather than a universal speedup. Rotate benchmark order, repeat allocations, evaluate more architectures and conduct a developer usability check before broadening the recommendation.
