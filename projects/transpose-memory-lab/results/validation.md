# Validation status

Validation performed on `login10.ufhpc` through an authenticated Windows SSH session on 2026-10-16. No GPU work was run on the login node and no Slurm job was submitted.

| Check | Result |
|---|---|
| Independent CPU known-answer / rectangular round-trip / invalid-input tests | PASS |
| CUDA compilation, `nvcc -std=c++17 -O3 -lineinfo -arch=sm_89` | PASS; CUDA 12.4.131 |
| CPU compiler | GNU C++ 11.5.0, C++17, `-O2 -Wall -Wextra -Werror` |
| Python analysis tests | 3 tests PASS; median, invalid values and empty data |
| Shell syntax checks | PASS for all three shell scripts |
| CLI help | exit 0 |
| Zero benchmark dimension / unknown mode | expected exit 1 before GPU initialization |
| Slurm `sbatch --test-only` with account/QOS `eel4930`, 1 GPU, `l4` constraint | exit 0; controller found an L4 candidate in `hpg-turin` |
| Actual GPU correctness / four sanitizer modes / profiler / timings | NOT RUN; GPU submission awaits user authorization |

The initial CUDA compilation failed on a deduced braced range; an explicit vector of dimension pairs fixed compatibility with this compiler. The final build passed. CPU tests are not GPU validation.

## Scheduling evidence

Slurm 25.11.8 responded to read-only queries. The user association allows `eel4930` and `eel4930-b`; the former had `GrpTRES=cpu=32,gres/gpu=8,mem=250G`, while the latter had `gres/gpu=0`. The L4 feature is advertised by `hpg-turin`. These are snapshots, not reservations or guarantees of future allocation.

The controller's test-only message referred to prospective job 46154256 and node c0605a-s9. That identifier was part of the feasibility test, not evidence of an actual submitted or executed job. This project does not contain a measured GPU performance result.

## Source identity at validation

```text
dc73ee8d3306fee2bf4bf1e2cd194a9d808125a086837462264e32d14b76bd0b  src/transpose.cu
1a254b4d8060e7b388870c45a8f9187a72a0f7e8995526bf072a4221602673ad  include/reference.hpp
```

Reproduce the outstanding checks using the checked-in Slurm script after GPU submission is authorized. The script records the actual source commit, hashes, device, driver and toolkit for that future run.
