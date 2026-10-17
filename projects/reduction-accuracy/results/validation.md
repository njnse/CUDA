# Validation record

## Completed in the cloud

- GNU C++ 14.2.0, C++17, `-O3 -Wall -Wextra -Werror`: independent oracle executable compiled and passed 29 checks.
- Python 3.12.14: five analysis tests passed, including corrupt/incomplete evidence rejection, exactness enforcement, zero-reference handling and FP32 error reporting.
- Shell syntax and Python source compilation: recorded in `cloud-cpu/checks.log`.
- No GPU benchmark measurements were generated.

Cloud CPU checks are not target compiler compatibility or GPU correctness evidence. The target worker uses GNU C++ 11.5 and CUDA 12.4.131. CUDA compilation, GPU correctness, benchmark, Compute Sanitizer and profiling are **pending**. The cloud has no usable CUDA compiler or GPU; a Docker socket access check was denied and no container compilation was performed.

A missing-tool worker probe is exercised only to verify nonzero failure propagation and pending GPU state. Its files remain in ignored local results storage, not in the automated GPU evidence directory. No synthetic analysis fixture is published as a benchmark result.

Only actual `automated-nvidia-l4/latest/validation.json`, matching source hashes, raw CSV and GPU verification/tool logs may establish completion. There is no such worker record for this project at publication time. Prior projects' measurements are not reused.
