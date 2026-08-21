# 008c GPU baseline -- environment

- date: 2026-06-17T14:21:06.615
- hostname: m13h-1-1
- julia: 1.11.7
- gpu: NVIDIA H200
- cuda runtime: 12.8.0
- total mem (GiB): 139.8
- P_DENSE_LIST: [2, 3, 4, 5, 6, 7, 10, 14, 20]
- BATCH_LIST: [1, 8, 64, 512, 4096, 32768, 262144]
- PREC_LIST: DataType[Float64, Float32]
- SAMPLES: 50

Compare seconds_per_expansion against dense_vs_loop_blas<N>.csv (CPU),
form=recurrence and form=dense, same P and batch. The production
recurrence is Float64; the precision=Float32 GPU rows quantify the
speedup from dropping to single precision.
