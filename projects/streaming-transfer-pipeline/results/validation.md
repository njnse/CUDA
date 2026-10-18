# Validation record

## Executed in the cloud

- GNU C++ 14.2.0 compiled the independent CPU tests with C++17, `-O3 -Wall -Wextra -Werror`; 44 contract/oracle/chunk-planning checks passed.
- Python 3.12.14 ran seven analysis/runner tests. Corrupt/incomplete samples, nonpositive/NaN timings, mismatched settings, inconsistent bandwidth, adverse pipeline results, invalid output paths and manifest scope are covered.
- Shell syntax and Python source compilation passed.
- An actual run of `scripts/run.sh` without CUDA returned exit 1 after passing CPU tests. Its validation state reported GPU pending and retained environment/failure/source information.
- Reusing that output directory returned exit 2 and preserved every existing artifact byte for byte. Missing, relative, project-root and source-tree output paths were rejected.
- Public cloud logs and source hashes are archived under `cloud-cpu/`. Probe outputs remain in ignored `results/local/`; no probe is presented as a GPU run.

## Pending on the GPU worker

CUDA compilation with Toolkit 12.4.131 and GNU C++ 11.5, 45 GPU correctness cases, 162 benchmark samples, Compute Sanitizer and profiling have not been executed by this cloud task. No CUDA compiler or GPU is available here. Cloud CPU compilation does not establish target compiler compatibility.

Completion requires actual `results/automated-nvidia-l4/latest/validation.json`, matching source hashes, raw CSV and verification/tool logs for this project. No such worker record exists at publication time. Prior project measurements and synthetic analyzer fixtures are not reused as evidence.

Unavailable or denied optional tools must remain explicitly unavailable. An overall core pass does not imply every profiler or sanitizer ran. Numerical timing comparisons and recommendations remain pending until matching evidence is uploaded.
