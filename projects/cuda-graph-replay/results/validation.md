# Validation record

GPU execution and publication completed successfully. The tested source is identified in automated-nvidia-l4/latest/source-commit.txt and source-sha256.txt; execution metadata is in validation.json.

- CUDA compilation and compiled CPU known-answer/tail/invalid-input tests: PASS.
- Python analysis tests: 3 PASS.
- GPU exact full-array comparisons: 5 lengths x 3 implementations, PASS.
- Output/temporary guards and input immutability: PASS.
- memcheck, initcheck, synccheck: 0 errors each.
- racecheck: 0 errors, 0 warnings.
- Benchmark: 225 raw samples, 15 per length and implementation, rotating path order.
- Capture, instantiate and upload durations recorded separately for each length.
- Nsight profiling: not executed.

An initial attempt failed exact comparison at the largest input. Its input/poison copies used the default stream while kernels used a nonblocking stream. The fix queues H2D copies on the execution stream and synchronizes initial input before capture. The same large-input check and all sanitizers pass after the fix. Tolerance was not widened.

Public evidence omits connection identifiers, absolute personal paths and machine/account names. Measurements, source hashes, GPU/driver/toolkit information and pass/fail output are preserved. Performance scope and limitations are described in the README; results apply to this workload and device.
