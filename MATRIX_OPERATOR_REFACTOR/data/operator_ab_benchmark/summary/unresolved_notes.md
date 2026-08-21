# Task 024 unresolved and noisy regimes

- Infeasible: cpu dense clustered Float64 LH=true P=12 N=20000: ArgumentError: DenseTranslationM2L persistent footprint exceeds max_persistent_bytes: D=365; classes=29644; apply_width=3072; build_width=365; operators=31594575200 bytes (30130.93490600586 MiB); slabs=17940480 bytes (17.109375 MiB); route metadata=336255784 bytes; persistent=31948771464 bytes (30468.722785949707 MiB); estimated construction peak=31957652528 bytes (30477.192428588867 MiB); limit=12884901888 bytes. Raise the limit; lower P; disable Lamb-Helmholtz; or reduce apply_chunk when slabs are material; chunking does not reduce operator storage.
- Infeasible: cpu dense uniform Float32 LH=false P=12 N=150: unsupported Float32/P=12 dense materialization
- Infeasible: cpu dense uniform Float32 LH=false P=12 N=20000: unsupported Float32/P=12 dense materialization
- Infeasible: cpu dense uniform Float32 LH=false P=12 N=2000: unsupported Float32/P=12 dense materialization
- Infeasible: cpu dense uniform Float32 LH=true P=12 N=150: unsupported Float32/P=12 dense materialization
- Infeasible: cpu dense uniform Float32 LH=true P=12 N=20000: unsupported Float32/P=12 dense materialization
- Infeasible: cpu dense uniform Float32 LH=true P=12 N=2000: unsupported Float32/P=12 dense materialization
- Infeasible: cuda dense clustered Float64 LH=true P=12 N=20000: ArgumentError: DenseTranslationM2L device persistent payload exceeds max_persistent_bytes (=12884901888 bytes (12288.0 MiB)): dense operators=31594575200 bytes (30130.93490600586 MiB); route metadata=67227440 bytes; expansion buffers=0 bytes (0.0 MiB); packing/application slabs=95682560 bytes (91.25 MiB); other scratch=0 bytes (0.0 MiB); persistent total=31757485200 bytes (30286.2979888916 MiB); estimated device peak=32807292896 bytes (31287.472625732422 MiB); D=365; classes=29644; chunk=16384. Lower P; disable Lamb-Helmholtz; reduce the applicable scratch chunk (DENSE_CUDA_CHUNK); or select PrecomputedFactoredYM2L / FactoredRotationM2L / ConcatenatedFixedZM2L (5-30x smaller footprint); chunking does not reduce dense operator storage.
- Infeasible: cuda dense uniform Float32 LH=false P=12 N=150: unsupported Float32/P=12 dense materialization
- Infeasible: cuda dense uniform Float32 LH=false P=12 N=20000: unsupported Float32/P=12 dense materialization
- Infeasible: cuda dense uniform Float32 LH=false P=12 N=2000: unsupported Float32/P=12 dense materialization
- Infeasible: cuda dense uniform Float32 LH=true P=12 N=150: unsupported Float32/P=12 dense materialization
- Infeasible: cuda dense uniform Float32 LH=true P=12 N=20000: unsupported Float32/P=12 dense materialization
- Infeasible: cuda dense uniform Float32 LH=true P=12 N=2000: unsupported Float32/P=12 dense materialization
- Noisy M2L (>10% IQR/median): cpu concat uniform Float64 LH=true P=8 N=20000 ratio=0.113.
- Noisy M2L (>10% IQR/median): cpu dense uniform Float32 LH=true P=8 N=2000 ratio=0.195.
- Noisy M2L (>10% IQR/median): cpu dense uniform Float64 LH=true P=12 N=20000 ratio=0.713.
