# CUDA portfolio workflow

All new GPU execution must use NVIDIA B200. Never fall back to L4 or another GPU. Existing L4 evidence is historical and must remain labeled as such.

Read docs/b200-workflow.md and the project index before every daily run. Add at most one new topic per America/New_York calendar day; inspect the latest commits and docs/daily-ledger.md first. Fix failed validation before adding a topic. Preserve remote work and never force push.

Every project needs an independent CPU oracle, boundary/error checks, fair baseline, reproducible scripts/run.sh, English tutorial and evidence-based engineering/product analysis. Public content must omit recruiting context and operational identifiers. Report only measured results. run.sh runs within an existing allocation, takes absolute RESULTS_DIR, and must not call SSH or sbatch.

Target native sm_100 with a verified CUDA 12.8+ compiler and compatible driver. Check the actual device before GPU work and fail on a non-B200. Never run GPU workloads on login nodes. Record environment, source hashes, shapes, seeds, warmups, repetitions, timing scope, variability, sanitizer and profiler status separately. Profile separately from timed benchmarks.

The existing server GPU worker polls main and publishes sanitized B200 results. Its presence does not mean a new project passed. Verify actual logs and remote files. Do not create additional schedulers or workers when an existing one is active.
