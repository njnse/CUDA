# A launch advisor should show evidence and limits

The product question is whether an example or launch advisor should present an occupancy-selected block size as a performance recommendation. This project separates legal geometry, compiler resource metadata, resource-predicted occupancy and measured timing. No timing or developer preference evidence exists for this new project yet; no customer interviews were conducted.

The implementation establishes that all policies execute the same exact operation with the same buffers and verification contract. It does not establish which policy wins. The runtime occupancy API sees compiled resource requirements; it does not optimize application transfer cost, underfilled grids or every arithmetic/cache behavior. The recorded minimum-grid recommendation and the experiment's fixed grid policy must not be confused.

A candidate diagnostic requirement is to display the proposed block/grid, relevant device/kernel limits, compiled registers/local/shared bytes and an explicitly labeled predicted occupancy value. An immediate configuration error should identify the failed launch; the example verifies that a rejected oversized block can be cleared and followed by a legal kernel. It does not claim recoverability from arbitrary device faults.

Acceptance evidence consists of independent exact oracle tests, full-array/tail/guard/input checks, the separate launch-error recovery test, available sanitizer logs and complete matched raw timing. A proposed performance default also requires repeated uninstrumented runs on representative sizes, instruction workloads and supported device/toolkit combinations. One sample sweep cannot establish a universal best block size.

After real worker data exists, compare fixed256 and the recommendation first, then compare the recommendation with the best observed fixed candidate. Report adverse and neutral cases. If the recommendation duplicates a fixed policy, report that identity instead of attributing noisy differences to a new optimization. If a difference is material, use achieved-occupancy, launch statistics and instruction/cache counters to investigate it separately.

Current recommendation: expose occupancy selection as a documented starting point and keep measured evidence separate. Defer a latency default until GPU execution. The project does not evaluate dynamic shared memory tuning, cooperative launch, cross-kernel fusion, divergent workloads, persistent threads or automated search; none of those capabilities is implied by these results.
