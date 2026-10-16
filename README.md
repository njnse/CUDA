# CUDA engineering portfolio

Reproducible CUDA experiments for developer education and evidence-based product decisions.

| Project | Engineering question | Evidence |
|---|---|---|
| [Transpose memory lab](projects/transpose-memory-lab/) | How do coalescing and shared-memory padding affect rectangular transpose? | Runnable tutorial, independent oracle, timing protocol, diagnostics and product acceptance criteria |

Each project separates source/build validation from actual GPU correctness and performance evidence. Never interpret an unexecuted benchmark as a result. The older `GPU-Computing` repository is separate and has not been migrated here.

## Working agreement

- Use descriptive topic directories, English code and documentation, and Korean chat reports.
- Preserve existing work; check the index before choosing the next topic.
- Pair implementation with correctness tests, a fair baseline, reproduction instructions, a developer tutorial and a product brief.
- GPU execution belongs on a Slurm compute allocation, not a login node.
- Publish only observed measurements with device, toolkit, input, timing scope and source provenance.

Next candidates: streaming transfers, CUDA Graph break-even, runtime/driver diagnostics, reduction accuracy, compiler regression minimization. Avoid repeating a topic without a new engineering question.

The [CPU CI example](docs/cpu-ci-example.yml) is documentation, not an active workflow. CPU and GPU checks were executed directly; no GitHub Actions success is claimed.
