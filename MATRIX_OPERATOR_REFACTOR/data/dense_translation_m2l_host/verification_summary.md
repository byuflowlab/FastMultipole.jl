# Task 023e dense-translation host verification summary

## Provenance and coverage

The non-macOS matrix contains `N = 150, 2000, 20000`, `P = 4, 8, 12`,
Float32/Float64, Lamb--Helmholtz off/on, and one- and 64-thread OpenBLAS runs.
Every configuration ran in a fresh Julia process. The four-strategy captures cover
dense, materialized concat, factored, and precomputed-y; the post-optimization
dense-only repeat covers all 72 dense configurations.

- CPU: AMD EPYC 7763 64-Core Processor.
- Julia: 1.11.7; BLAS: libblastrampoline with ILP64 OpenBLAS.
- Functional host/node: `m12-1-7` (one recovered row from the identical pre-fix
  capture on `m12-1-18`); final dense host/node: `m12-4-30`.
- Git commit: `4d5ea4ee1f3a303e4df1dba19f8e160766ac65b1`.
- Git tree: `37ea598b9100c7a617c334b60736e5f45114f9bd`; worktree status: dirty.
- Functional baseline: job `12840355` (287 rows) plus the missing configuration
  recovered from the identical pre-fix kernel in job `12840356` (288 unique rows).
  Job `12848113` separately reran that configuration after the retained fix and is
  not used in the functional aggregate.
- Four-strategy optimized/no-regression capture: job `12840356` (288 rows).
- Final dense capture after the retained allocation specialization: job `12848432`
  (72 rows).

The dirty status is intentional: these jobs benchmark the task implementation in
the working tree. Each CSV repeats the provenance and all strategy options.

## Functional versus final dense result

Matched per-configuration ratios use all 72 dense configurations. Medians across a
heterogeneous sweep are useful as regression summaries, not as a recommendation
between strategies (that decision remains task 024).

| Metric | Functional median | Final median | Median final / functional |
|---|---:|---:|---:|
| Construction | 18,672.6 ms | 18,499.9 ms | 0.9984 |
| Warmed M2L stage | 103.23 ms | 93.13 ms | 0.9649 |
| Warmed full step | 391.66 ms | 345.90 ms | 0.9833 |
| Warmed M2L allocation | 77,520 bytes | 1,104 bytes | 0.0142 |
| Warmed full-step allocation | 197,248 bytes | 118,776 bytes | 0.6156 |

Final M2L allocation was exactly 1,104 bytes in every ORC row; the maximum final
full-step allocation was 123,032 bytes. Both are comfortably below the 64 KiB and
512 KiB gates, respectively. Construction timing is unchanged within run-to-run
noise, as expected: the retained specialization affects only steady-state dispatch.

## Dense persistent memory

The following are maximum-occupancy (`N=20000`) persistent totals. They include
complete operators, both application slabs, and route metadata; construction peak
is reported separately in every CSV.

| Precision | LH | P=4 | P=8 | P=12 |
|---|---:|---:|---:|---:|
| Float32 | off | 8.82 MiB | 83.42 MiB | 363.07 MiB |
| Float32 | on | 24.47 MiB | 395.84 MiB | 1,653.63 MiB |
| Float64 | off | 8.04 MiB | 161.76 MiB | 721.06 MiB |
| Float64 | on | 43.91 MiB | 786.61 MiB | 3,312.05 MiB |

The worst measured construction peak was 3,320.52 MiB (Float64, LH on, P=12),
below the default 4 GiB persistent limit. Chunking reduces application or builder
scratch, not the dominant operator storage.

## Optimization decisions

- Retained: concrete host `Vector{Int}`/`Matrix{TF}` function barriers for dense
  gather/scatter and prefix GEMM. This removed per-class boxed arguments and drove
  P=12 launch allocation from 78--104 KiB down to 1.1 KiB.
- Rejected: scalar multiplication for narrow classes. Locally, P=8 Float64 LH-on
  rose from about 0.68 s to 2.13 s when width-one classes used the scalar path.
- Rejected: apply chunks 8/16/32. The local P=4/N=2000 M2L stage measured about
  2.73/2.37/2.23 ms versus 2.17--2.19 ms at full class width.
- Rejected: smaller builder chunks as a speed optimization; widths 1/4/8/16/full
  were within construction noise. The control remains useful for peak memory.
- Rejected: removal of stable packing. It saved only about 0.07 ms at 34,512 routes
  and would introduce an additional route-emission ordering dependency.
- Not pursued: occupancy reordering or operand transposition, because profiling did
  not identify an ordering/layout bottleneck.

64-thread BLAS was faster than one thread in only 13 of 36 matched dense M2L-stage
configurations and was 2.2% slower at the median (with wide configuration-dependent
variation). No global BLAS thread setting is hard-coded; callers should choose it
for their workload.

## Verification

- Dense focused suite: 70/70 tests after the retained optimization.
- Resident buffer/M2L/M2M/L2L suite: 923/923 tests.
- Precomputed-y regression: 73/73 tests.
- Radix integration: 63/63; time stepping: 51,759/51,759 combined assertions.
- Full `Pkg.test()`: passed twice, including the final post-optimization run (one
  existing broken test, no failures).
- Local macOS results are smoke tests only; the EPYC rows above are definitive for
  task 023e CPU measurement.
