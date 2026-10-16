# Product brief: make CUDA memory examples diagnosable

## Product question

Can a small CUDA sample help a developer distinguish indexing bugs, unsupported environments and workload-dependent performance without requiring an expert to interpret a single speedup number?

This is a portfolio experiment, not NVIDIA user research. The internship responsibilities were supplied by the user. No customer interviews, adoption metrics or competitive benchmarks have been collected.

## Implemented evidence

- Three transpose paths plus a separately labeled copy context path.
- Independent exact oracle, edge shapes, poisoned outputs, write guards and input-immutability checks.
- `--diagnose` reports the device, compute capability, runtime version and driver API version. Driver API version is not a complete compatibility assessment; retain `nvidia-smi` and the toolkit version too.
- Missing GPU / driver is distinguished by exit 77 and a diagnostic message; invalid CLI input and CUDA errors return 1.
- Raw timing samples and summary dispersion are preserved. GPU evidence remains pending until a compute-node run succeeds.

## Observations versus hypotheses

Observed in the current session: Windows SSH with user-approved Duo authentication reaches HiPerGator; CUDA 12.4.131 and GNU C++ 11.5.0 are available. Slurm recognizes the user's `eel4930` account and two QOS entries. These observations establish tooling access, not GPU performance.

Hypotheses to test: padding will reduce bank-conflict overhead in appropriate larger matrices; tiny shapes will be dominated by launch costs; explicit diagnostics and reproduction metadata may reduce support ambiguity. The latter is a product hypothesis requiring user testing, not an outcome measured here.

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

Keep the explicit diagnostic and evidence contract in the portfolio. Defer claims about measured speedups or developer productivity. After an authorized L4 run, evaluate correctness and profiler evidence before deciding whether padding should be the recommended tutorial default. If build compatibility fails on another GPU generation, capture a minimal compiler/runtime reproduction before changing target-selection behavior.
