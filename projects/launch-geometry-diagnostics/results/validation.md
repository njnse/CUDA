# Validation record

## Executed in the cloud

- GNU C++ 14.2.0 compiled the independent CPU tests with C++17 and `-O3 -Wall -Wextra -Werror`. All 47 checks passed: composed oracle versus iterative arithmetic, known answers, wraparound, exact grid-stride coverage, grid-cap behavior and invalid shapes/limits.
- Python 3.12.14 ran seven analysis/runner tests. They cover complete and corrupt evidence, geometry/occupancy metadata, adverse heuristic performance, invalid output locations, artifact protection and manifest scope.
- Shell syntax and Python source compilation passed.
- A real `run.sh` invocation without CUDA passed CPU checks, exited 1 for unavailable GPU tools and retained source/environment/failure evidence with GPU pending.
- Reusing that output returned exit 2 and preserved every artifact byte for byte. Missing, relative, project-root and source-tree output locations were rejected.

Public logs, environment and source SHA-256 manifest are under `cloud-cpu/`. Probe outputs remain in ignored local results storage. Synthetic parser fixtures are not measurements. CPU checks do not establish CUDA compilation, GPU occupancy or launch-error recovery.

## GPU validation

The experiment passed in a Slurm allocation with one NVIDIA B200, driver 580.178.04, CUDA 12.8.93, GCC 11.5.0 and native `sm_100`. Native compilation, the 40 GPU oracle cases, guards, input immutability, rejected-launch recovery, 216 benchmark samples, four Compute Sanitizer tools and the Nsight duration capture passed.

At 1,048,576 elements, median CUDA-event intervals for the one-round kernel were 3.66 µs with a fixed 512-thread block and 3.81 µs with the occupancy recommendation. For 64 rounds, the fixed 256-thread block measured 4.70 µs and the recommendation measured 5.12 µs. The differences describe this run and include launch gaps in the stream interval. Inspect the archived ranges and paired observations before selecting a deployment policy.

[Evidence](automated-nvidia-b200/latest/) contains raw samples, summary, environment, source hashes and verification logs. The source commit file identifies the checkout base. The source manifest identifies the exact files tested, including the updated B200 runner.
