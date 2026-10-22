# Validation record

## Executed in the cloud

- GNU C++ 14.2.0 compiled the independent CPU tests with C++17 and `-O3 -Wall -Wextra -Werror`. All 47 checks passed: composed oracle versus iterative arithmetic, known answers, wraparound, exact grid-stride coverage, grid-cap behavior and invalid shapes/limits.
- Python 3.12.14 ran seven analysis/runner tests. They cover complete and corrupt evidence, geometry/occupancy metadata, adverse heuristic performance, invalid output locations, artifact protection and manifest scope.
- Shell syntax and Python source compilation passed.
- A real `run.sh` invocation without CUDA passed CPU checks, exited 1 for unavailable GPU tools and retained source/environment/failure evidence with GPU pending.
- Reusing that output returned exit 2 and preserved every artifact byte for byte. Missing, relative, project-root and source-tree output locations were rejected.

Public logs, environment and source SHA-256 manifest are under `cloud-cpu/`. Probe outputs remain in ignored local results storage. Synthetic parser fixtures are not measurements. CPU checks do not establish CUDA compilation, GPU occupancy or launch-error recovery.

## Pending GPU execution

No CUDA compiler or GPU is available in this cloud. CUDA 12.4.131/GNU C++ 11.5 compilation, 40 normal GPU payload cases plus the separate rejected-launch recovery check, 216 benchmark samples, sanitizer execution and focused profiling are pending. No speedups, occupancy values or profiler measurements have been generated for this project.

Normal verification tests the expected invalid configuration outside sanitizer. Sanitizer quick mode runs only 16 positive payload cases. Optional tools must be reported individually; a core pass does not imply they all executed.

Completion requires actual `automated-nvidia-l4/latest/validation.json`, matching source hashes, raw CSV and verification/tool logs. Previous projects' GPU results are not reused.
