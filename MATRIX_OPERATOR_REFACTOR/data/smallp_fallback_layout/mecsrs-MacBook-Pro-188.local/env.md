# 019b small-P fallback + chi layout exploratory benchmarks -- environment

- date: 2026-07-11T21:50:34.784
- hostname: mecsrs-MacBook-Pro-188.local
- julia: 1.12.5
- cpu_model: Apple M2
- physical/logical cores: 4
- Threads.nthreads(): 1
- BLAS.get_num_threads(): 1
- BLAS libs: libopenblas64_p-r0.3.31.dylib
- BLAS optimized?: true
- git HEAD: 4d5ea4ee1f3a303e4df1dba19f8e160766ac65b1
- git dirty: true
- P_LIST: [1, 2, 3, 4, 5, 6, 8, 12]
- BATCH_LIST: [1, 2, 4, 8, 16, 32, 64, 256, 1024, 4096]
- SAMPLES: 30
- RECURRENCE_BATCH_CAP: 2048
- STAGE_P_LIST: [1, 2, 3, 4, 6, 8]
- LAYOUT_P_LIST: [2, 4, 8, 12, 20]
- LAYOUT_BATCH_LIST: [64, 4096]
