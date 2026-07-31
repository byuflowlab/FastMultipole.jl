# 015 axis-swap (M2L variant) benchmarks -- environment

- date: 2026-06-24T07:48:16.173
- hostname: tmpfac-126-17.et.byu.edu
- julia: 1.12.5
- cpu_model: Apple M2
- physical/logical cores: 4
- Threads.nthreads(): 1
- BLAS.get_num_threads(): 8
- BLAS libs: libopenblas64_p-r0.3.31.dylib
- BLAS optimized?: true
- git HEAD: 2ded52692dfccbd404cb96ee77cb5a3a31406c41
- git dirty: true
- P_LIST: [4, 8, 12, 20]
- BATCH_LIST: [1, 8, 64, 512, 4096]
- PREC_LIST: DataType[Float64]
- SAMPLES: 50
- RECURRENCE_BATCH_CAP: 2048
- COMP_P_LIST: [8, 20]
- COMP_BATCH_LIST: [64, 4096]
- DISTINCT_LIST: [1, 8, 64]
- COMP_SAMPLES: 15

> GPU: no GPU operator path exists yet (task 022 deferred). Actual GPU
> benchmarks are staged for tasks 022 and 024; the 015 GPU recommendation is
> analytical (see 015-results.md), reasoning from the 008c GPU baseline.
