# Validation record

## Executed cloud checks

GNU C++ 14.2.0 compiled the CPU tests with C++17 and `-O3 -Wall -Wextra -Werror`. Nine oracle/size checks passed: known values, an independent incremental wraparound calculation over six lengths, and invalid sizes. Seven Python analysis/runner tests passed, including incomplete/corrupt evidence rejection, invalid/live pool accounting, adverse performance, protected output paths and source-manifest scope.

Shell syntax and Python source compilation passed. An actual `run.sh` invocation in this CUDA-free cloud passed its CPU checks, returned exit 1 for unavailable GPU tools, retained environment/source/failure evidence and left GPU status pending. A repeated invocation rejected the existing output directory with exit 2 and preserved every artifact byte for byte. Missing, relative, project-root and source-tree output paths were rejected.

Logs, cloud environment and source SHA-256 manifest are in `cloud-cpu/`. Probe outputs remain in ignored local results storage. Synthetic parser fixtures are not benchmark results. CPU checks do not establish CUDA event ordering or target compiler compatibility.

## Pending GPU work

No CUDA compiler or GPU is available in this cloud. Compilation with CUDA 12.4.131 and GNU C++ 11.5, 30 GPU correctness cases repeated three times, 81 benchmark samples, Compute Sanitizer and profiling are pending. No timings, speedups, counter values or allocator memory measurements have been generated for this project.

An actual worker record at `automated-nvidia-l4/latest/validation.json`, matching source hashes, raw CSV and detailed correctness/tool logs is required to establish those outcomes. Missing tools remain unavailable, and an overall core pass does not imply that every optional tool executed. Previous projects' results are not reused as evidence.
