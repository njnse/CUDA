# Validation status

Source and documentation are published. The Python analysis tests were executed and passed: median aggregation, empty-input rejection and invalid/non-finite timing rejection (3 tests).

CUDA compilation, the compiled CPU reference test, GPU correctness, all four sanitizer modes and performance measurement have not yet been completed for this project. No GPU speedup or setup amortization result is claimed.

The run script is compatible with the repository's GPU worker contract: it uses an existing allocation, honors RESULTS_DIR, records source hashes and environment data, and fails on unsuccessful checks. A passing worker run will publish sanitized evidence under results/automated-nvidia-l4/latest/.

Measure the eager, graph and fused paths on the same inputs. Report graph capture, instantiate and upload cost separately. Recompute conclusions from raw samples; do not reuse the transpose project's measurements.
