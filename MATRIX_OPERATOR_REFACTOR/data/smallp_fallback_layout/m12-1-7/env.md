# 019b small-P fallback + chi layout exploratory benchmarks -- environment

- date: 2026-07-11T23:01:32.302
- hostname: m12-1-7
- julia: 1.11.7
- cpu_model: AMD EPYC 7763 64-Core Processor
- physical/logical cores: 128
- Threads.nthreads(): 1
- BLAS.get_num_threads(): 64
- BLAS libs: libopenblas64_.so
- BLAS optimized?: true
- git HEAD: unknown
- git dirty: false
- P_LIST: [1, 2, 3, 4, 5, 6, 8, 12]
- BATCH_LIST: [1, 2, 4, 8, 16, 32, 64, 256, 1024, 4096]
- SAMPLES: 30
- RECURRENCE_BATCH_CAP: 2048
- STAGE_P_LIST: [1, 2, 3, 4, 6, 8]
- LAYOUT_P_LIST: [2, 4, 8, 12, 20]
- LAYOUT_BATCH_LIST: [64, 4096]
