# Task 023c non-macOS host benchmark summary — 2026-07-18

## Environment and sweep

BYU ORC node `m12-1-3`, AMD EPYC 7763 64-Core Processor, Linux
5.14.0-570.120.1.el9_6.x86_64, Julia 1.11.7, ILP64 OpenBLAS through
libblastrampoline. Separate processes reported actual BLAS thread counts 1 and
64; Julia used one thread. The synchronized tree was based on commit
`4d5ea4ee1f3a303e4df1dba19f8e160766ac65b1` with task changes in a dirty
worktree. Both jobs ran the focused 023c suite (73/73) before benchmarking.

The full sweep covered `P=4,8,12`, `N=150,2000,20000`, LH off/on, three timing
repetitions, and the 023a factored reference, scalar precomputed-y, forced GEMM,
production threshold, and column thresholds 4, 8, 12, 16, and 24.

## Crossover decision

Job 12799879 measured the provisional threshold 12. Across its 18 regimes per
thread count, threshold 16 was the only focused global candidate that remained
within 5% of the best candidate for both the M2L stage and full step in both
thread modes. Its worst gaps were 3.4% stage / 3.1% full step with one BLAS
thread and 1.7% / 3.7% with 64 threads. Threshold 12 was up to 15.6% behind the
best 64-thread M2L-stage candidate. Therefore production uses
`PRECOMPUTED_Y_GEMM_MIN_COLS = 16`. A single column rule met the acceptance
criterion, so a block-dimension rule would add complexity without measured need.

Job 12801369 is the final threshold-16 confirmation. Its one-thread production
rows were no more than 4.9% behind the per-regime stage best and 2.8% behind the
full-step best. The individual 64-thread confirmation had one 5.3% stage gap
and one 10.8% full-step outlier at `P=12`; at the latter point the M2L-stage gap
was only 1.5%, and the selection job had the opposite full-step ordering (16 was
faster than 12), identifying lifecycle timing noise rather than a crossover
change. Combining both independent jobs by geometric mean, threshold 16 stayed
within 3.4% of the best stage candidate and 4.8% of the best full-step candidate
over all 36 thread/regime combinations. All final production rows recorded
threshold 16, and both final processes exited zero.

## Representative performance and storage

Candidate-selection data at `P=8`, `N=2000`, LH off:

- One thread: scalar precomputed-y 613.9 ms M2L / 649.6 ms full step; threshold
  16 took 486.0 / 521.9 ms. The 023a reference took 1037.9 / 1085.6 ms.
- 64 threads: scalar precomputed-y 611.6 ms M2L / 650.0 ms full step; threshold
  16 took 499.1 / 536.9 ms. The 023a reference took 1055.5 / 1093.3 ms.
- Precomputed-y invariant operators occupied 15.6 MB and capacity-sized scratch
  64.9 MB, versus 72.4 MB operators and 1.4 MB scratch for 023a. Warmed measured
  allocations were 27,088 bytes for M2L and 43,568 bytes for the full step.

At the largest `P=12`, `N=20000`, LH-on point, 222,648 routes occupied 387 angle
classes (maximum angle occupancy 22,240) versus 3,252 offset classes (maximum
280). Threshold 16 took 2995.5 ms for M2L and 3824.2 ms for the full step with
one BLAS thread, versus 6081.8 and 6920.4 ms for 023a. Precomputed-y used 86.0 MB
of operators plus 331.0 MB scratch; 023a used 143.5 MB plus 2.5 MB. This records
the intended operator-memory reduction and the honest occupancy-dependent
scratch tradeoff.

Across the candidate-selection sweep, threshold 16 improved M2L over the scalar
precomputed-y baseline by 1.04–1.38x with one BLAS thread and 1.02–1.28x with 64.
Construction ranged from roughly 4.5 to 428 ms, depending on order, LH mode, and
capacity; it is paid once per resident cache. Occupancy, construction, allocation,
operator, and scratch columns remain in the raw CSVs for every regime.
