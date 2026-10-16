# Transpose memory lab

**Question:** For a row-major FP32 matrix, when does a tiled transpose justify its extra shared memory and synchronization, and what evidence should a CUDA example expose to developers?

Four paths isolate useful effects: coalesced copy (bandwidth context), naive transpose (strided stores), 32-by-32 shared-memory transpose, and a padded 32-by-33 shared-memory transpose. Copy is not a transpose implementation and is not a correctness competitor. All transpose variants implement `out[col * rows + row] = in[row * cols + col]` out of place. In-place operation, arbitrary strides and overlapping pointers are outside this executable's contract.

## Build and validate

On HiPerGator, build on the login node but run GPU commands inside a compute allocation:

```bash
module load cuda/12.4.1
bash projects/transpose-memory-lab/scripts/build.sh
python3 -m unittest discover -s projects/transpose-memory-lab/tests -p 'test_*.py' -v
# The following are GPU-allocation commands:
projects/transpose-memory-lab/build/transpose_lab --diagnose
projects/transpose-memory-lab/build/transpose_lab --verify
projects/transpose-memory-lab/build/transpose_lab --benchmark 4093 2051 15 > measured.csv
python3 projects/transpose-memory-lab/scripts/analyze.py measured.csv
```

The default target is `sm_89` for L4. Set `CUDA_ARCH=sm_XX` to a target supported by your compiler and allocated GPU. CUDA 12.4 should not be assumed to support every newer HiPerGator GPU. The build script does not select a GPU or request an allocation.

## Slurm reproduction

From the repository root:

```bash
# Validate scheduling feasibility without creating a job:
sbatch --test-only projects/transpose-memory-lab/scripts/hipergator.sbatch
# Submit only when authorized:
sbatch --constraint=l4 projects/transpose-memory-lab/scripts/hipergator.sbatch
```

The account and default QOS are `eel4930`. The checked `eel4930-b` QOS has a group GPU limit of zero and is not a GPU fallback. No partition is hard-coded. The `l4` feature was verified on `hpg-turin`; recheck site features before submitting. Submission does not guarantee immediate allocation. The full script builds, verifies, benchmarks four shapes and runs four Compute Sanitizer tools. Missing sanitizer tools fail the script rather than silently claim coverage. Results go under `results/local/slurm-JOBID/`, which is excluded from Git until deliberately selected and reviewed.

## Evidence and scope

- CPU: hand-calculated known-answer test, rectangular round trips and invalid input rejection.
- GPU: exact element equality against the independent CPU oracle; fixed-seed signed finite data; NaN-poisoned outputs; input immutability; prefix/suffix guards; ten boundary/rectangular/square shapes.
- Memory safety: guarded output is useful but not a substitute for Compute Sanitizer.
- Timing: ten warmups, fifteen samples by default, 100 launches/sample below one million elements and ten otherwise; CUDA events on the default stream; raw samples plus median/min/max. Allocation, copies, verification and CPU work are excluded. Launch gaps remain part of the event interval.
- Effective GB/s counts one read plus one write (`2 * rows * cols * sizeof(float)`). It is logical traffic, not measured DRAM traffic; cache-resident results can exceed external-memory bandwidth.
- Fixed variant order and same-buffer replay can bias clock/cache conditions. Rotate order and use a working set larger than cache before making stronger hardware claims. No library baseline or cross-device claim is provided.

See [validation](results/validation.md), [tutorial](docs/developer-tutorial.md), [product brief](docs/product-brief.md) and [profiling protocol](docs/profiling.md).
