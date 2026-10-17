# Engineering decision: repeated GPU work

Question: should a small repeated pipeline use ordinary launches, graph replay, or fusion?

Acceptance criteria:
1. Every implementation matches the independent oracle for every test length, with intact guards and immutable input.
2. All four sanitizer tools complete without reported errors.
3. A report preserves raw samples, setup stages, dimensions, iteration counts, build target, device, driver and toolkit.
4. Setup is excluded from steady-state replay and included explicitly in the amortization calculation.
5. The single-kernel fused path is a required comparator; an eager-only speedup is insufficient evidence for recommending graphs.

Observations: implementation provides these comparisons and tests. Actual GPU results are pending in this revision.

Hypotheses: graph replay can reduce repeated host submission cost; fusion can outperform both two-kernel paths; for larger inputs, memory traffic can dominate. These are hypotheses until measured.

Alternatives: ordinary launches minimize setup and shape-management complexity; graphs amortize stable dependency structures but retain lifetime/update obligations; fusion removes launches and intermediate traffic but may be impractical across library boundaries or independently maintained operations.

Recommendation: choose based on measured synchronized wall time and expected replay count, while retaining setup, accuracy and integration costs. One simple pointwise pipeline on one device does not represent an application-wide acceleration. No user-study or customer evidence is claimed.
