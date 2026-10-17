# Engineering decision: repeated GPU work

Question: should a small repeated pipeline use ordinary launches, graph replay, or fusion?

Acceptance criteria:
1. Every implementation matches the independent oracle for every test length, with intact guards and immutable input.
2. All four sanitizer tools complete without reported errors.
3. A report preserves raw samples, setup stages, dimensions, iteration counts, build target, device, driver and toolkit.
4. Setup is excluded from steady-state replay and included explicitly in the amortization calculation.
5. The single-kernel fused path is a required comparator; an eager-only speedup is insufficient evidence for recommending graphs.

Observations: all five lengths and three implementations passed full-array correctness and all four sanitizer tools. Graph replay improved measured eager wall time by about 1.06x to 1.41x across three process runs in two allocations. Fusion was fastest at every tested length. See the README table and raw samples.

Interpretation: the measurements support reduced submission overhead for graph replay and lower overall cost for fusion in this specific pointwise pipeline. Cross-device behavior, workload changes, concurrent replay and application-wide impact remain untested.

Alternatives: ordinary launches minimize setup and shape-management complexity; graphs amortize stable dependency structures but retain lifetime/update obligations; fusion removes launches and intermediate traffic but may be impractical across library boundaries or independently maintained operations.

Recommendation: choose based on measured synchronized wall time and expected replay count, while retaining setup, accuracy and integration costs. One simple pointwise pipeline on one device does not represent an application-wide acceleration. No user-study or customer evidence is claimed.
