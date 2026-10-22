# Ask three separate launch questions

First ask whether the launch is legal. The device has block/grid limits, and the compiled kernel has its own thread/resource requirements. `DeviceInfo` queries both. The CPU planner validates dimensions before launching and uses a grid-stride mapping, so the capped grid still covers the array. CPU tests count visits to expose skipped or repeated elements independently of the kernel output formula.

Next ask what can reside on an SM. `cudaOccupancyMaxActiveBlocksPerMultiprocessor` evaluates resource constraints for the compiled kernel and a particular block size. The predicted fraction says how many threads could reside relative to the SM limit. It does not tell you how many SMs receive blocks, how many tail lanes do useful work or how long a dependent arithmetic chain stalls.

Then measure the work. Read `benchmark` in `src/geometry.cu`: identical input/output buffers and a single kernel, rotating policy order, fixed warmups/repetitions and both stream-event and synchronized host clocks. If the recommended block equals a fixed choice, similar results are expected and timing differences may be noise. A high predicted fraction is a hypothesis about resources, not an objective function for latency.

The CPU oracle composes the full affine recurrence, while the GPU repeatedly updates one value per visited element. Both implement exact unsigned modulo arithmetic. Poison, guard and input-immutability checks prevent an early or incomplete launch from appearing correct. There is no floating-point tolerance to adjust after a failure.

Read the normal verification log for the expected invalid-configuration test. The oversized block is rejected before kernel execution. Reading the last error clears that launch error; a legal launch then verifies context usability. This is different from an asynchronous memory access fault, which this project never deliberately triggers. The sanitizer quick mode contains only positive launches, keeping tool API-error summaries interpretable.

After worker execution, inspect `diagnose.log`, `verify.log`, `raw.csv` and `summary.json` together. Compare the recommended policy with fixed256 and the best observed fixed choice at the same length and round count. Check IQR and repeat complete runs before turning a small median difference into a default recommendation.
